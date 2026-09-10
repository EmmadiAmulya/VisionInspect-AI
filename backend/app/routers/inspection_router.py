import os
import shutil
import uuid

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile
)

from sqlalchemy.orm import Session

from database import get_db
from models import Inspection, User

from auth import (
    get_current_user,
    require_inspector
)

from ml_inference import inspect_image
from image_quality import analyze_image_quality


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
# SEVERITY CALCULATION
# ============================================================

def calculate_severity(
    defect_detected,
    defect_type,
    confidence,
    defect_region
):

    # --------------------------------------------------------
    # No defect
    # --------------------------------------------------------

    if not defect_detected:

        return {
            "severity_score": 0.0,
            "severity_level": "Low"
        }

    # --------------------------------------------------------
    # SIZE SCORE
    # --------------------------------------------------------

    size_score = 50.0

    if defect_region:

        width = defect_region.get(
            "width",
            0
        )

        height = defect_region.get(
            "height",
            0
        )

        area = width * height

        if area >= 1000:

            size_score = 100.0

        elif area >= 500:

            size_score = 80.0

        elif area >= 250:

            size_score = 65.0

        elif area >= 100:

            size_score = 50.0

        else:

            size_score = 30.0

    # --------------------------------------------------------
    # LOCATION SCORE
    # --------------------------------------------------------

    location_score = 60.0

    # --------------------------------------------------------
    # DEFECT TYPE SCORE
    # --------------------------------------------------------

    defect_type_scores = {

        "broken_large": 90.0,

        "broken_small": 65.0,

        "contamination": 75.0,

        "good": 0.0
    }

    defect_type_score = defect_type_scores.get(
        defect_type,
        60.0
    )

    # --------------------------------------------------------
    # CONFIDENCE SCORE
    # --------------------------------------------------------

    confidence_score = (
        float(confidence)
        * 100.0
    )

    # --------------------------------------------------------
    # FINAL SEVERITY SCORE
    #
    # Size       = 30%
    # Location   = 25%
    # Defect     = 25%
    # Confidence = 20%
    # --------------------------------------------------------

    severity_score = (

        size_score * 0.30

        +

        location_score * 0.25

        +

        defect_type_score * 0.25

        +

        confidence_score * 0.20
    )

    severity_score = round(
        severity_score,
        2
    )

    # --------------------------------------------------------
    # SEVERITY LEVEL
    # --------------------------------------------------------

    if severity_score >= 80:

        severity_level = "Critical"

    elif severity_score >= 60:

        severity_level = "High"

    elif severity_score >= 40:

        severity_level = "Medium"

    else:

        severity_level = "Low"

    return {

        "severity_score":
            severity_score,

        "severity_level":
            severity_level
    }


# ============================================================
# UPLOAD + INSPECT
# ============================================================

@router.post(
    "/upload"
)
async def upload_and_inspect(

    file: UploadFile = File(...),

    current_user: User = Depends(
        require_inspector
    ),

    db: Session = Depends(
        get_db
    )
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

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not save image: {str(e)}"
            )
        )

    # ========================================================
    # IMAGE QUALITY + AI INSPECTION
    # ========================================================

    try:

        # ----------------------------------------------------
        # IMAGE QUALITY ANALYSIS
        # ----------------------------------------------------

        quality_result = analyze_image_quality(
            file_path
        )

        # ----------------------------------------------------
        # REJECT POOR QUALITY IMAGE
        # ----------------------------------------------------

        if not quality_result["quality_ok"]:

            # Remove poor-quality image
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
            file_path
        )

    except Exception as e:

        # ----------------------------------------------------
        # REMOVE IMAGE IF PROCESSING FAILS
        # ----------------------------------------------------

        if os.path.exists(file_path):

            os.remove(file_path)

        raise HTTPException(
            status_code=500,
            detail=(
                f"AI inspection failed: {str(e)}"
            )
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

    # ========================================================
    # SEVERITY
    # ========================================================

    severity_result = calculate_severity(

        defect_detected,

        defect_type,

        classification_confidence,

        defect_region
    )

    severity_score = severity_result[
        "severity_score"
    ]

    severity_level = severity_result[
        "severity_level"
    ]

    # ========================================================
    # DATABASE PATH
    # ========================================================

    relative_image_path = os.path.join(
        "uploads",
        unique_filename
    )

    # ========================================================
    # CREATE INSPECTION RECORD
    # ========================================================

    inspection = Inspection(

        image_path=relative_image_path,

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

    db.commit()

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

        "inspector":
            current_user.username,

        "user_role":
            current_user.role,

        "image_path":
            relative_image_path
    }


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
    )
):

    inspections = (
        db.query(
            Inspection
        )
        .order_by(
            Inspection.created_at.desc()
        )
        .all()
    )

    results = []

    for inspection in inspections:

        results.append({

            "id":
                inspection.id,

            "image_path":
                inspection.image_path,

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

            "created_at":
                inspection.created_at
        })

    return results