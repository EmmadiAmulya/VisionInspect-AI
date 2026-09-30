from datetime import datetime, timedelta, timezone
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jwt
from pwdlib import PasswordHash
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

try:
    from database import get_db
    from models import User
except ImportError:
    try:
        from app.database import get_db
        from app.models import User
    except ImportError:
        from backend.app.database import get_db
        from backend.app.models import User


# ==========================================
# Password Hashing
# ==========================================

password_hash = PasswordHash.recommended()


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(
    password: str,
    hashed_password: str
) -> bool:
    return password_hash.verify(
        password,
        hashed_password
    )


# ==========================================
# JWT Configuration
# ==========================================

DEFAULT_SECRET_KEY = "visioninspect-secret-key-change-later"

SECRET_KEY = os.getenv("SECRET_KEY", DEFAULT_SECRET_KEY)

_APP_ENV = os.getenv("ENV", os.getenv("ENVIRONMENT", "development"))

if SECRET_KEY == DEFAULT_SECRET_KEY:
    if _APP_ENV.lower() == "production":
        raise RuntimeError(
            "SECRET_KEY must be set to a secure value when ENV=production."
        )
    import warnings
    warnings.warn(
        "Using default SECRET_KEY. Set SECRET_KEY env var in production.",
        UserWarning,
    )

ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))


oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/login"
)


# ==========================================
# Create JWT Access Token
# ==========================================

def create_access_token(user_id, username=None) -> str:
    """Create a JWT with ``sub`` = user id.

    New style: ``create_access_token(user.id, user.username)``.
    Legacy single-arg ``create_access_token(username)`` is still accepted
    so old callers do not crash, but new code must pass id + username.
    """

    # Backward-compat: called as create_access_token(username)
    if username is None:
        legacy_username = str(user_id)
        now = datetime.now(timezone.utc)
        expire = now + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )
        payload = {
            "sub": legacy_username,
            "iat": now,
            "exp": expire,
            "jti": str(uuid.uuid4()),
        }
        token = jwt.encode(
            payload,
            SECRET_KEY,
            algorithm=ALGORITHM
        )
        return token

    now = datetime.now(timezone.utc)
    expire = now + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    payload = {
        "sub": str(user_id),
        "username": username,
        "iat": now,
        "exp": expire,
        "jti": str(uuid.uuid4()),
    }

    token = jwt.encode(
        payload,
        SECRET_KEY,
        algorithm=ALGORITHM
    )

    return token


# ==========================================
# Get Current User
# ==========================================

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={
            "WWW-Authenticate": "Bearer"
        }
    )

    try:

        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        sub = payload.get("sub")

        if sub is None:
            raise credentials_exception

    except jwt.InvalidTokenError:

        raise credentials_exception


    user = None
    sub_str = str(sub)

    # New-style token: sub is numeric user id
    if sub_str.isdigit():
        user = db.query(User).filter(
            User.id == int(sub_str)
        ).first()
        # Fallback: verify username claim if id lookup misses
        # (e.g. user recreated) — try username claim, then legacy path.
        if user is None:
            username_claim = payload.get("username")
            if username_claim:
                user = db.query(User).filter(
                    User.username == username_claim
                ).first()
    else:
        # Legacy token: sub holds the username
        user = db.query(User).filter(
            User.username == sub_str
        ).first()

    if user is None:
        raise credentials_exception


    return user


# ==========================================
# Inspector Role
# ==========================================

def require_inspector(
    current_user: User = Depends(get_current_user)
):

    if current_user.role not in ["inspector", "supervisor", "admin"]:

        raise HTTPException(
            status_code=403,
            detail="Inspection access required"
        )

    return current_user



# ==========================================
# Supervisor Role
# ==========================================

def require_supervisor(
    current_user: User = Depends(get_current_user)
):

    if current_user.role not in ["supervisor", "admin"]:

        raise HTTPException(
            status_code=403,
            detail="Supervisor or Admin access required"
        )

    return current_user