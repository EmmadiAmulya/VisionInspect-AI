import os
import shutil
import uuid
import logging
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    Query,
)
from fastapi.responses import FileResponse

from pydantic import BaseModel, Field
from typing import Optional
from sqlalchemy.orm import Session

try:
    from database import get_db
    from models import Inspection, User, SystemSettings
    from auth import (
        get_current_user,
        require_inspector,
        require_supervisor
    )
    from ml_inference import (
        inspect_image,
        get_supported_categories,
        normalize_category,
        get_anomaly_visualization,
    )
    from image_quality import analyze_image_quality
    from severity import calculate_severity
    from quality_assessment import assess_quality
    from thresholds import UNIFIED_DEFAULTS, validate_severity_ordering
except ImportError:
    try:
        from app.database import get_db
        from app.models import Inspection, User, SystemSettings
        from app.auth import (
            get_current_user,
            require_inspector,
            require_supervisor
        )
        from app.ml_inference import (
            inspect_image,
            get_supported_categories,
            normalize_category,
            get_anomaly_visualization,
        )
        from app.image_quality import analyze_image_quality
        from app.severity import calculate_severity
        from app.quality_assessment import assess_quality
        from app.thresholds import UNIFIED_DEFAULTS, validate_severity_ordering
    except ImportError:
        from backend.app.database import get_db
        from backend.app.models import Inspection, User, SystemSettings
        from backend.app.auth import (
            get_current_user,
            require_inspector,
            require_supervisor
        )
        from backend.app.ml_inference import (
            inspect_image,
            get_supported_categories,
            normalize_category,
            get_anomaly_visualization,
        )
        from backend.app.image_quality import analyze_image_quality
        from backend.app.severity import calculate_severity
        from backend.app.quality_assessment import assess_quality
        from backend.app.thresholds import (
            UNIFIED_DEFAULTS,
            validate_severity_ordering,
        )

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB DoS guard



# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/inspections",
    tags=["Inspections"]
)


# ============================================================
# UPLOAD DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "uploads"
)

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


# ============================================================
# UPLOAD + INSPECT
# ============================================================

@router.post(
    "/upload"
)
def upload_and_inspect(

    file: UploadFile = File(...),

    # Phase 2: product category via form field (primary) or ?category=
    # query (fallback for webcam/legacy clients). Defaults to bottle.
    category: Optional[str] = Form(default=None),

    current_user: User = Depends(
        require_inspector
    ),

    db: Session = Depends(
        get_db
    ),

    category_query: Optional[str] = Query(default=None, alias="category"),
):

    # ========================================================
    # VALIDATE FILE TYPE
    # ========================================================

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    }

    allowed_mime = {
        "image/jpeg",
        "image/png",
        "image/bmp",
    }

    original_filename = (
        file.filename
        or ""
    )

    extension = os.path.splitext(
        original_filename
    )[1].lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Use JPG, JPEG, PNG or BMP."
            )
        )

    if file.content_type and file.content_type not in allowed_mime:
        raise HTTPException(
            status_code=400,
            detail="Unsupported media type."
        )


    # ========================================================
    # RESOLVE + VALIDATE PRODUCT CATEGORY (Phase 2)
    # ========================================================

    _raw_category = category if category else category_query
    try:
        product_category = normalize_category(_raw_category or "bottle")
    except Exception:
        product_category = "bottle"

    try:
        _available = {c.get("name") for c in get_supported_categories()}
        _avail_entry = next(
            (c for c in get_supported_categories()
             if c.get("name") == product_category),
            None,
        )
        if product_category not in _available:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown product category: '{product_category}'. "
                    f"Available: {sorted(_available)}"
                ),
            )
        if _avail_entry and not (
            _avail_entry.get("autoencoder_available")
            and _avail_entry.get("classifier_available")
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No trained models for category '{product_category}'. "
                    f"Train via ml/train_category.py --category "
                    f"{product_category} first."
                ),
            )
    except HTTPException:
        raise
    except Exception:
        # Category listing must never block uploads (e.g. models dir
        # unreadable); fall back to bottle default.
        product_category = normalize_category(_raw_category or "bottle")


    # ========================================================
    # CREATE UNIQUE FILE NAME
    # ========================================================

    unique_filename = (
        f"{uuid.uuid4()}"
        f"{extension}"
    )

    file_path = os.path.join(
        UPLOAD_DIR,
        unique_filename
    )


    # ========================================================
    # SAVE IMAGE
    # ========================================================

    try:

        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        if os.path.getsize(file_path) > MAX_UPLOAD_BYTES:
            os.remove(file_path)
            raise HTTPException(
                status_code=413,
                detail="Image exceeds 10MB size limit."
            )

    except HTTPException:
        raise

    except Exception:

        logger.exception("Could not save image")
        raise HTTPException(
            status_code=500,
            detail="Could not save image."
        )


    # ========================================================
    # IMAGE QUALITY + AI INSPECTION (DB thresholds wired)
    # ========================================================

    def _load_runtime_thresholds():
        vals = dict(UNIFIED_DEFAULTS)
        try:
            rows = db.query(SystemSettings).all()
            for r in rows:
                try:
                    vals[r.setting_key] = float(r.setting_value)
                except (TypeError, ValueError):
                    continue
        except Exception:
            pass
        # Validate ordering, fallback if misordered
        validate_severity_ordering(vals)
        return vals

    try:

        runtime_thresholds = _load_runtime_thresholds()

        # ----------------------------------------------------
        # IMAGE QUALITY ANALYSIS
        # ----------------------------------------------------

        quality_result = analyze_image_quality(
            file_path,
            blur_threshold=runtime_thresholds["blur_threshold"],
        )


        # ----------------------------------------------------
        # REJECT POOR QUALITY IMAGE
        # ----------------------------------------------------

        if not quality_result.get("quality_ok", False):

            if os.path.exists(file_path):

                os.remove(file_path)

            return {

                "message":
                    "Image quality is poor. "
                    "Please upload a better image.",

                "inspection_id":
                    None,

                "status":
                    "REJECTED",

                "image_quality":
                    quality_result
            }


        # ----------------------------------------------------
        # AI INSPECTION
        # ----------------------------------------------------

        ai_result = inspect_image(
            file_path,
            category=product_category,
            anomaly_threshold=runtime_thresholds["anomaly_threshold"],
            confidence_threshold=runtime_thresholds.get(
                "classification_confidence_threshold", 0.50
            ),
        )


    except Exception:

        # ----------------------------------------------------
        # REMOVE IMAGE IF PROCESSING FAILS
        # ----------------------------------------------------

        if os.path.exists(file_path):

            os.remove(file_path)

        logger.exception("AI inspection failed")
        raise HTTPException(
            status_code=500,
            detail="AI inspection failed."
        )


    # ========================================================
    # AI RESULT
    # ========================================================

    status = ai_result.get(
        "status",
        "PASS"
    )

    defect_detected = ai_result.get(
        "defect_detected",
        False
    )

    defect_type = ai_result.get(
        "defect_type",
        "good"
    )

    classification_confidence = (
        ai_result.get(
            "classification_confidence",
            0.0
        )
    )

    reconstruction_error = (
        ai_result.get(
            "reconstruction_error",
            0.0
        )
    )

    local_anomaly_score = (
        ai_result.get(
            "local_anomaly_score",
            0.0
        )
    )

    defect_region = ai_result.get(
        "defect_region"
    )

    # Normalize rescaled original-size coords back to 224 space for
    # severity scoring (thresholds tuned for 224x224).
    severity_region = defect_region
    if defect_region:
        try:
            ow = defect_region.get("orig_width")
            oh = defect_region.get("orig_height")
            if ow and oh:
                sx = 224.0 / float(ow)
                sy = 224.0 / float(oh)
                severity_region = {
                    "x": defect_region.get("x", 0) * sx,
                    "y": defect_region.get("y", 0) * sy,
                    "width": max(1.0, defect_region.get("width", 0) * sx),
                    "height": max(1.0, defect_region.get("height", 0) * sy),
                }
        except (TypeError, ValueError, ZeroDivisionError):
            severity_region = defect_region


    # ========================================================
    # SEVERITY (DB cutoffs wired)
    # ========================================================

    try:
        _rt = runtime_thresholds
    except NameError:
        _rt = dict(UNIFIED_DEFAULTS)

    severity_result = calculate_severity(

        defect_detected,

        defect_type,

        classification_confidence,

        severity_region,

        critical_cutoff=_rt["critical_severity_cutoff"],
        high_cutoff=_rt["high_severity_cutoff"],
        medium_cutoff=_rt["medium_severity_cutoff"],
    )

    severity_score = severity_result[
        "severity_score"
    ]

    severity_level = severity_result[
        "severity_level"
    ]


    # ========================================================
    # QUALITY ASSESSMENT
    # ========================================================

    quality_assessment = assess_quality(

        status,

        defect_detected,

        defect_type,

        classification_confidence,

        severity_score,

        severity_level,

        quality_result
    )


    # ========================================================
    # DATABASE PATH (posix for cross-OS consistency)
    # ========================================================

    relative_image_path = "uploads/" + unique_filename


    # ========================================================
    # CREATE INSPECTION RECORD
    # ========================================================

    inspection = Inspection(

        image_path=relative_image_path,

        product_category=product_category,

        defect_type=defect_type,

        confidence=classification_confidence,

        status=status,

        inspected_by=current_user.id,

        severity_score=severity_score,

        severity_level=severity_level
    )

    db.add(
        inspection
    )

    try:
        db.commit()
    except Exception:
        db.rollback()
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500,
            detail="Could not save inspection record."
        )

    db.refresh(
        inspection
    )


    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "message":
            "Image uploaded and inspected successfully",

        "inspection_id":
            inspection.id,

        "product_category":
            ai_result.get("product_category", product_category),

        "status":
            status,

        "defect_detected":
            defect_detected,

        "defect_type":
            defect_type,

        "classification_confidence":
            round(
                classification_confidence,
                4
            ),

        "reconstruction_error":
            round(
                reconstruction_error,
                6
            ),

        "local_anomaly_score":
            round(
                local_anomaly_score,
                2
            ),

        "detection_threshold":
            ai_result.get(
                "threshold",
                90.0
            ),

        "defect_region":
            defect_region,

        "severity_score":
            severity_score,

        "severity_level":
            severity_level,

        "quality_status":
            quality_assessment[
                "quality_status"
            ],

        "quality_decision":
            quality_assessment[
                "quality_decision"
            ],

        "risk_level":
            quality_assessment[
                "risk_level"
            ],

        "recommendation":
            quality_assessment[
                "recommendation"
            ],

        "inspector":
            current_user.username,

        "user_role":
            current_user.role,

        "image_path":
            f"/inspections/image/{unique_filename}"
    }


# ============================================================
# PRODUCT CATEGORIES (Phase 2: dynamic model discovery)
# ============================================================

@router.get(
    "/categories"
)
def list_product_categories(
    current_user: User = Depends(get_current_user),
):
    try:
        categories = get_supported_categories()
    except Exception as exc:
        logger.exception("Category listing failed")
        raise HTTPException(
            status_code=500,
            detail=f"Could not list categories: {exc}",
        )
    return {
        "categories": categories,
        "default": "bottle",
    }


# ============================================================
# ANOMALY VISUALIZATION (Phase 3: heatmap / mask overlays)
# POST a file + category, receive base64 heatmap payload.
# ============================================================

@router.post(
    "/visualize"
)
def visualize_anomaly(
    file: UploadFile = File(...),
    category: Optional[str] = Form(default=None),
    category_query: Optional[str] = Query(default=None, alias="category"),
    current_user: User = Depends(get_current_user),
):
    _raw = category if category else category_query
    try:
        product_category = normalize_category(_raw or "bottle")
    except Exception:
        product_category = "bottle"

    extension = os.path.splitext(file.filename or "")[1].lower()
    if extension not in {".jpg", ".jpeg", ".png", ".bmp"}:
        raise HTTPException(status_code=400, detail="Unsupported image format.")

    tmp_name = f"viz_{uuid.uuid4()}{extension or '.png'}"
    tmp_path = os.path.join(UPLOAD_DIR, tmp_name)
    try:
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        payload = get_anomaly_visualization(tmp_path, category=product_category)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("Visualization failed")
        raise HTTPException(status_code=500, detail="Visualization failed.")
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
    return payload


# ============================================================
# AUTHENTICATED IMAGE SERVING
# (uploads/ NOT exposed via StaticFiles)
# ============================================================

_ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@router.get(
    "/image/{filename}"
)
def serve_inspection_image(
    filename: str,
    request: Request,
    db: Session = Depends(get_db),
    token: Optional[str] = Query(default=None),
):
    # Manual auth: Authorization: Bearer <jwt> OR ?token=<jwt> (for <img> tags
    # which cannot send headers). Mirrors auth.get_current_user logic.
    auth_token = token
    try:
        _auth = request.headers.get("authorization", "")
        if _auth.lower().startswith("bearer "):
            auth_token = _auth.split(" ", 1)[1].strip()
    except Exception:
        pass
    if not auth_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        import jwt as _jwt
        try:
            from auth import SECRET_KEY as _SK, ALGORITHM as _ALG
        except ImportError:
            try:
                from app.auth import SECRET_KEY as _SK, ALGORITHM as _ALG
            except ImportError:
                from backend.app.auth import SECRET_KEY as _SK, ALGORITHM as _ALG
        _payload = _jwt.decode(auth_token, _SK, algorithms=[_ALG])
        _sub = _payload.get("sub")
        if _sub is None:
            raise HTTPException(status_code=401, detail="Could not validate credentials")
        _user = None
        try:
            if str(_sub).isdigit():
                _user = db.query(User).filter(User.id == int(_sub)).first()
            if _user is None:
                _user = db.query(User).filter(User.username == str(_sub)).first()
                if _user is None and _payload.get("username"):
                    _user = db.query(User).filter(User.username == _payload.get("username")).first()
        except Exception:
            _user = None
        if _user is None:
            raise HTTPException(status_code=401, detail="Could not validate credentials")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Could not validate credentials")

    # Reject path traversal / subdirectories
    if (
        not filename
        or ".." in filename
        or "/" in filename
        or "\\" in filename
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid filename."
        )

    safe_name = os.path.basename(filename)

    if safe_name != filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename."
        )

    extension = os.path.splitext(safe_name)[1].lower()

    if extension not in _ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format."
        )

    file_path = os.path.join(UPLOAD_DIR, safe_name)

    # Resolve and confirm containment inside UPLOAD_DIR
    abs_upload = os.path.abspath(UPLOAD_DIR)
    abs_target = os.path.abspath(file_path)

    if os.path.commonpath([abs_upload, abs_target]) != abs_upload:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename."
        )

    if not os.path.isfile(abs_target):
        raise HTTPException(
            status_code=404,
            detail="Image not found."
        )

    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
    }

    return FileResponse(
        abs_target,
        media_type=media_types.get(extension, "application/octet-stream"),
        filename=safe_name,
    )


# ============================================================
# INSPECTION HISTORY
# ============================================================

@router.get(
    "/history"
)
def inspection_history(

    current_user: User = Depends(
        get_current_user
    ),

    db: Session = Depends(
        get_db
    ),

    limit: int = Query(default=200, le=1000, ge=1),
    offset: int = Query(default=0, ge=0),
    user_id: Optional[int] = Query(default=None),
    category: Optional[str] = Query(default=None),
):

    query = db.query(Inspection)

    if current_user.role == "inspector":
        # Inspectors only see their own inspections
        query = query.filter(
            Inspection.inspected_by == current_user.id
        )
    elif user_id is not None:
        # Supervisors/admins may optionally filter by user
        if current_user.role not in ("supervisor", "admin"):
            raise HTTPException(
                status_code=403,
                detail="Not authorized to filter by user."
            )
        query = query.filter(
            Inspection.inspected_by == user_id
        )

    if category:
        try:
            _cat = normalize_category(category)
            query = query.filter(Inspection.product_category == _cat)
        except Exception:
            pass

    inspections = (
        query
        .order_by(
            Inspection.created_at.desc()
        )
        .offset(offset)
        .limit(limit)
        .all()
    )

    results = []

    for inspection in inspections:

        _stored = inspection.image_path or ""
        _base = os.path.basename(_stored.replace("\\", "/"))
        _url = f"/inspections/image/{_base}" if _base else _stored

        results.append({

            "id":
                inspection.id,

            "image_path":
                inspection.image_path,

            "image_url":
                _url,

            "product_category":
                getattr(inspection, "product_category", None) or "bottle",

            "defect_type":
                inspection.defect_type,

            "confidence":
                inspection.confidence,

            "status":
                inspection.status,

            "severity_score":
                inspection.severity_score,

            "severity_level":
                inspection.severity_level,

            "inspected_by":
                inspection.inspected_by,

            "is_overridden":
                inspection.is_overridden or False,

            "override_reason":
                inspection.override_reason,

            "original_status":
                inspection.original_status,

            "created_at":
                inspection.created_at
        })

    return results



# ============================================================
# INSPECTION ANALYTICS
# ============================================================

@router.get(
    "/analytics"
)
def inspection_analytics(

    current_user: User = Depends(
        get_current_user
    ),

    db: Session = Depends(
        get_db
    )
):

    # ========================================================
    # GET ALL INSPECTIONS
    # ========================================================

    inspections = (
        db.query(
            Inspection
        )
        .order_by(
            Inspection.id
        )
        .all()
    )


    # ========================================================
    # TOTAL INSPECTIONS
    # ========================================================

    total_inspections = len(
        inspections
    )


    # ========================================================
    # PASS / FAIL / PENDING COUNTS
    # ========================================================

    passed = 0

    failed = 0

    pending = 0


    for inspection in inspections:

        inspection_status = (
            inspection.status
            or ""
        ).upper()


        if inspection_status == "PASS":

            passed += 1

        elif inspection_status == "FAIL":

            failed += 1

        elif inspection_status == "PENDING":

            pending += 1


    # ========================================================
    # DEFECT COUNT
    # ========================================================
    # A defect is counted only when the inspection failed
    # and the defect type is not "good".

    defect_count = 0


    for inspection in inspections:

        inspection_status = (
            inspection.status
            or ""
        ).upper()

        defect_type = (
            inspection.defect_type
            or ""
        ).lower()


        if (
            inspection_status == "FAIL"
            and defect_type != "good"
        ):

            defect_count += 1


    # ========================================================
    # DEFECT RATE
    # ========================================================

    if total_inspections > 0:

        defect_rate = (
            defect_count
            /
            total_inspections
        ) * 100.0

    else:

        defect_rate = 0.0


    defect_rate = round(
        defect_rate,
        2
    )


    # ========================================================
    # DEFECT DISTRIBUTION
    # ========================================================

    defect_distribution = {

        "broken_large": 0,

        "broken_small": 0,

        "contamination": 0
    }


    for inspection in inspections:

        defect_type = (
            inspection.defect_type
            or ""
        ).lower()

        inspection_status = (
            inspection.status
            or ""
        ).upper()


        if (
            defect_type
            in defect_distribution
            and inspection_status == "FAIL"
        ):

            defect_distribution[
                defect_type
            ] += 1


    # ========================================================
    # SEVERITY DISTRIBUTION
    # ========================================================

    severity_distribution = {

        "Critical": 0,

        "High": 0,

        "Medium": 0,

        "Low": 0
    }


    for inspection in inspections:

        severity_level = (
            inspection.severity_level
        )


        if (
            severity_level
            in severity_distribution
        ):

            severity_distribution[
                severity_level
            ] += 1


    # ========================================================
    # RETURN ANALYTICS
    # ========================================================

    return {

        "total_inspections":
            total_inspections,

        "passed":
            passed,

        "failed":
            failed,

        "pending":
            pending,

        "defect_count":
            defect_count,

        "defect_rate":
            defect_rate,

        "defect_distribution":
            defect_distribution,

        "severity_distribution":
            severity_distribution
    }


# ============================================================
# INSPECTION TRENDS
# ============================================================

@router.get(
    "/trends"
)
def inspection_trends(

    current_user: User = Depends(
        get_current_user
    ),

    db: Session = Depends(
        get_db
    )
):

    # ========================================================
    # GET ALL INSPECTIONS
    # ========================================================

    inspections = (
        db.query(
            Inspection
        )
        .order_by(
            Inspection.created_at
        )
        .all()
    )


    # ========================================================
    # GROUP INSPECTIONS BY DATE
    # ========================================================

    daily_data = {}


    for inspection in inspections:

        if inspection.created_at is None:

            continue


        inspection_date = (
            inspection.created_at
            .date()
            .isoformat()
        )


        if inspection_date not in daily_data:

            daily_data[
                inspection_date
            ] = {

                "date":
                    inspection_date,

                "total":
                    0,

                "passed":
                    0,

                "failed":
                    0,

                "defects":
                    0
            }


        daily_data[
            inspection_date
        ][
            "total"
        ] += 1


        inspection_status = (
            inspection.status
            or ""
        ).upper()


        if inspection_status == "PASS":

            daily_data[
                inspection_date
            ][
                "passed"
            ] += 1


        elif inspection_status == "FAIL":

            daily_data[
                inspection_date
            ][
                "failed"
            ] += 1


        defect_type = (
            inspection.defect_type
            or ""
        ).lower()


        if (
            inspection_status == "FAIL"
            and defect_type != "good"
        ):

            daily_data[
                inspection_date
            ][
                "defects"
            ] += 1


    # ========================================================
    # CALCULATE DAILY DEFECT RATE
    # ========================================================

    trends = []


    for date, data in daily_data.items():

        if data["total"] > 0:

            defect_rate = (
                data["defects"]
                /
                data["total"]
            ) * 100.0

        else:

            defect_rate = 0.0


        trends.append({

            "date":
                data["date"],

            "total":
                data["total"],

            "passed":
                data["passed"],

            "failed":
                data["failed"],

            "defects":
                data["defects"],

            "defect_rate":
                round(
                    defect_rate,
                    2
                )
        })


    # ========================================================
    # RETURN TREND DATA
    # ========================================================

    return {

        "trends":
            trends
    }


@router.get("/quality-report")
def production_quality_report(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    inspections = (
        db.query(Inspection)
        .order_by(Inspection.created_at)
        .all()
    )

    total = len(inspections)

    passed = 0
    failed = 0
    pending = 0

    defect_counts = {
        "broken_large": 0,
        "broken_small": 0,
        "contamination": 0
    }

    severity_counts = {
        "Critical": 0,
        "High": 0,
        "Medium": 0,
        "Low": 0
    }

    for inspection in inspections:
        status = (inspection.status or "").upper()

        if status == "PASS":
            passed += 1
        elif status == "FAIL":
            failed += 1
        else:
            pending += 1

        defect_type = (inspection.defect_type or "").lower()

        if defect_type in defect_counts and status == "FAIL":
            defect_counts[defect_type] += 1

        severity_level = inspection.severity_level

        if severity_level in severity_counts:
            severity_counts[severity_level] += 1

    if total > 0:
        defect_rate = (failed / total) * 100
    else:
        defect_rate = 0.0

    if total > 0:
        pass_rate = (passed / total) * 100
    else:
        pass_rate = 0.0

    return {
        "report": {
            "total_inspections": total,
            "passed_inspections": passed,
            "failed_inspections": failed,
            "pending_inspections": pending,
            "pass_rate": round(pass_rate, 2),
            "defect_rate": round(defect_rate, 2),
            "defect_distribution": defect_counts,
            "severity_distribution": severity_counts
        }
    }


# ============================================================
# SUPERVISOR DECISION OVERRIDE
# ============================================================

class OverrideRequest(BaseModel):
    new_status: str = Field(min_length=4, max_length=4)
    reason: str = Field(min_length=5, max_length=500)


@router.put("/{inspection_id}/override")
def override_inspection_decision(
    inspection_id: int,
    request: OverrideRequest,
    current_user: User = Depends(require_supervisor),
    db: Session = Depends(get_db)
):
    inspection = db.query(Inspection).filter(Inspection.id == inspection_id).first()
    if not inspection:
        raise HTTPException(status_code=404, detail="Inspection not found")

    new_status = request.new_status.upper()
    if new_status not in ["PASS", "FAIL"]:
        raise HTTPException(status_code=400, detail="Status must be 'PASS' or 'FAIL'")

    if not inspection.is_overridden:
        inspection.original_status = inspection.status

    inspection.status = new_status
    inspection.is_overridden = True
    inspection.overridden_by = current_user.id
    inspection.override_reason = request.reason

    db.commit()
    db.refresh(inspection)

    return {
        "message": f"Inspection #{inspection_id} status overridden to {new_status}",
        "inspection_id": inspection.id,
        "new_status": inspection.status,
        "is_overridden": True,
        "override_reason": inspection.override_reason,
        "overridden_by": current_user.username
    }


# ============================================================
# QUALITY THRESHOLD SETTINGS
# ============================================================

class SettingsUpdateSchema(BaseModel):
    blur_threshold: Optional[float] = 20.0
    anomaly_threshold: Optional[float] = 90.0
    classification_confidence_threshold: Optional[float] = 0.50
    critical_severity_cutoff: Optional[float] = 80.0
    high_severity_cutoff: Optional[float] = 60.0
    medium_severity_cutoff: Optional[float] = 40.0


DEFAULT_SETTINGS = {
    "blur_threshold": "20.0",
    "anomaly_threshold": "90.0",
    "classification_confidence_threshold": "0.50",
    "critical_severity_cutoff": "80.0",
    "high_severity_cutoff": "60.0",
    "medium_severity_cutoff": "40.0",
}


@router.get("/config/settings")
def get_system_settings(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    stored_settings = db.query(SystemSettings).all()
    settings_dict = dict(DEFAULT_SETTINGS)

    for s in stored_settings:
        settings_dict[s.setting_key] = s.setting_value

    parsed = {}
    for k, v in settings_dict.items():
        try:
            parsed[k] = float(v)
        except (TypeError, ValueError):
            parsed[k] = v

    return {
        "settings": parsed
    }


@router.post("/config/settings")
def update_system_settings(
    payload: SettingsUpdateSchema,
    current_user: User = Depends(require_supervisor),
    db: Session = Depends(get_db)
):
    updates = payload.model_dump(exclude_unset=True, exclude_none=True)

    for key, val in updates.items():
        val_str = str(val)
        existing = db.query(SystemSettings).filter(SystemSettings.setting_key == key).first()
        if existing:
            existing.setting_value = val_str
        else:
            db.add(SystemSettings(setting_key=key, setting_value=val_str))

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="Could not save settings.")
    return {
        "message": "System threshold settings updated successfully",
        "updated_keys": list(updates.keys())
    }