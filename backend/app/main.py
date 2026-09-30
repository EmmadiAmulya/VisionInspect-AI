import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from routers.inspection_router import router as inspection_router
    from routers.auth_router import router as auth_router
    from database import Base, engine
    import models
except ImportError:
    try:  # allow `uvicorn app.main:app` from backend/
        from app.routers.inspection_router import router as inspection_router
        from app.routers.auth_router import router as auth_router
        from app.database import Base, engine
        import app.models  # noqa: F401
    except ImportError:  # allow `python -c "import backend.app.main"` from root
        from backend.app.routers.inspection_router import router as inspection_router
        from backend.app.routers.auth_router import router as auth_router
        from backend.app.database import Base, engine
        import backend.app.models  # noqa: F401


# ==========================================
# CREATE DATABASE TABLES
# ==========================================

try:
    Base.metadata.create_all(bind=engine)
except Exception as exc:  # DB down should not crash import; log and continue
    print(f"Warning: could not create tables: {exc}")


# ==========================================
# BACKWARD-COMPAT MIGRATION (Phase 2: product_category)
# create_all() never alters existing tables, so add the column
# best-effort for Postgres/SQLite without a full Alembic setup.
# ==========================================

try:
    from sqlalchemy import text as _migrate_text

    with engine.connect() as _conn:
        try:
            _conn.execute(
                _migrate_text(
                    "ALTER TABLE inspections "
                    "ADD COLUMN IF NOT EXISTS "
                    'product_category VARCHAR(100) DEFAULT \'bottle\''
                )
            )
            _conn.commit()
        except Exception:
            # SQLite (<3.35) lacks IF NOT EXISTS for ADD COLUMN.
            try:
                _conn.rollback()
                _cols = _conn.execute(
                    _migrate_text("PRAGMA table_info(inspections)")
                ).fetchall()
                _names = {c[1] for c in _cols}
                if "product_category" not in _names:
                    _conn.execute(
                        _migrate_text(
                            "ALTER TABLE inspections ADD COLUMN "
                            "product_category VARCHAR(100) DEFAULT 'bottle'"
                        )
                    )
                    _conn.commit()
            except Exception as _exc2:
                print(f"Warning: product_category migration skipped: {_exc2}")
except Exception as exc:  # DB down should not crash import; log and continue
    print(f"Warning: migration check skipped: {exc}")


# ==========================================
# CREATE FASTAPI APPLICATION
# ==========================================
from contextlib import asynccontextmanager


def _seed_default_accounts():
    try:
        from database import SessionLocal
        import models as _models
        from auth import hash_password
    except ImportError:
        try:
            from app.database import SessionLocal
            import app.models as _models
            from app.auth import hash_password
        except ImportError:
            from backend.app.database import SessionLocal
            import backend.app.models as _models
            from backend.app.auth import hash_password
    admin_user = os.getenv("ADMIN_USERNAME")
    admin_pass = os.getenv("ADMIN_PASSWORD")
    admin_email = os.getenv("ADMIN_EMAIL", "admin@visioninspect.local")
    if not admin_user or not admin_pass:
        return
    db = SessionLocal()
    try:
        existing = db.query(_models.User).filter(
            _models.User.username == admin_user
        ).first()
        if not existing:
            db.add(_models.User(
                username=admin_user,
                email=admin_email,
                password_hash=hash_password(admin_pass),
                role="supervisor",
                is_active=True,
            ))
            db.commit()
            print(f"Seeded supervisor account: {admin_user}")
    except Exception as exc:
        print(f"Warning: admin seed failed: {exc}")
        db.rollback()
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _seed_default_accounts()
    yield


app = FastAPI(
    title="VisionInspect AI",
    description="Manufacturing Defect Detection & Quality Inspection System",
    version="1.0.0",
    lifespan=lifespan,
)


# ==========================================
# CORS
# ==========================================

cors_origins = os.getenv(
    "CORS_ORIGINS",
    "http://127.0.0.1:3000,http://localhost:3000,"
    "http://127.0.0.1:5173,http://localhost:5173"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in cors_origins if o.strip()],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+)(:\d+)?",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ==========================================
# UPLOAD DIRECTORY (served via authenticated endpoint, not StaticFiles)
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "uploads"
)

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


# ==========================================
# FAIL CLOSED IF DEFAULT SECRET IN PRODUCTION
# ==========================================

_DEFAULT_SECRET_KEY = "visioninspect-secret-key-change-later"
_APP_ENV = os.getenv("ENV", os.getenv("ENVIRONMENT", "development"))

if (
    _APP_ENV.lower() == "production"
    and os.getenv("SECRET_KEY", _DEFAULT_SECRET_KEY) == _DEFAULT_SECRET_KEY
):
    raise RuntimeError(
        "SECRET_KEY must be set to a secure value when ENV=production."
    )


# ==========================================
# ROUTERS
# ==========================================

app.include_router(
    inspection_router
)

app.include_router(
    auth_router
)


# ==========================================
# ROOT + HEALTH (Phase 1: orchestrator probes)
# ==========================================

@app.get("/")
def root():

    return {
        "message": "VisionInspect AI Backend is running"
    }


@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "visioninspect-backend",
    }


@app.get("/ready")
def ready():
    checks = {"database": False, "models": False}
    details = {}
    try:
        try:
            from database import engine as _engine
        except ImportError:
            try:
                from app.database import engine as _engine
            except ImportError:
                from backend.app.database import engine as _engine
        from sqlalchemy import text as _text
        with _engine.connect() as conn:
            conn.execute(_text("SELECT 1"))
        checks["database"] = True
    except Exception as exc:
        details["database_error"] = str(exc)
    try:
        try:
            from ml_inference import get_supported_categories as _cats
        except ImportError:
            try:
                from app.ml_inference import (
                    get_supported_categories as _cats,
                )
            except ImportError:
                from backend.app.ml_inference import (
                    get_supported_categories as _cats,
                )
        cats = _cats()
        checks["models"] = any(
            c.get("autoencoder_available") and c.get("classifier_available")
            for c in cats
        )
        details["categories"] = len(cats)
    except Exception as exc:
        details["models_error"] = str(exc)
    ready_ok = all(checks.values())
    return {
        "ready": ready_ok,
        "checks": checks,
        "details": details,
    }


# ==========================================
# SEED DEFAULT SUPERVISOR (env-driven, idempotent)
# Handled in lifespan() above.
# ==========================================