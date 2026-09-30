from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from database import get_db
    from models import User
    from schemas import UserRegister
    from auth import (
        hash_password,
        verify_password,
        create_access_token,
        get_current_user
    )
except ImportError:
    try:
        from app.database import get_db
        from app.models import User
        from app.schemas import UserRegister
        from app.auth import (
            hash_password,
            verify_password,
            create_access_token,
            get_current_user
        )
    except ImportError:
        from backend.app.database import get_db
        from backend.app.models import User
        from backend.app.schemas import UserRegister
        from backend.app.auth import (
            hash_password,
            verify_password,
            create_access_token,
            get_current_user
        )


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)


# ==========================================
# LOGIN RATE LIMIT (in-memory, no new deps)
# ==========================================

_login_attempts: dict[str, list[float]] = {}
LOGIN_RATE_LIMIT_MAX = 10
LOGIN_RATE_LIMIT_WINDOW = 60.0  # seconds


def _check_login_rate_limit(key: str, now: float) -> bool:
    """Return True if key exceeded max attempts in window."""
    attempts = _login_attempts.get(key, [])
    cutoff = now - LOGIN_RATE_LIMIT_WINDOW
    attempts = [t for t in attempts if t > cutoff]
    _login_attempts[key] = attempts
    return len(attempts) >= LOGIN_RATE_LIMIT_MAX


def _record_login_attempt(key: str, now: float) -> None:
    attempts = _login_attempts.get(key, [])
    cutoff = now - LOGIN_RATE_LIMIT_WINDOW
    attempts = [t for t in attempts if t > cutoff]
    attempts.append(now)
    _login_attempts[key] = attempts


# ==========================================
# REGISTER
# ==========================================

@router.post("/register")
def register(
    user_data: UserRegister,
    db: Session = Depends(get_db)
):

    if len(user_data.password) < 6:
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 6 characters"
        )

    # Check username
    existing_username = db.query(User).filter(
        User.username == user_data.username
    ).first()

    if existing_username:
        raise HTTPException(
            status_code=400,
            detail="Username already exists"
        )

    # Check email
    existing_email = db.query(User).filter(
        User.email == user_data.email
    ).first()

    if existing_email:
        raise HTTPException(
            status_code=400,
            detail="Email already exists"
        )

    # Hash password
    hashed_password = hash_password(
        user_data.password
    )

    # Create user
    new_user = User(
        username=user_data.username,
        email=user_data.email,
        password_hash=hashed_password,
        role="inspector"
    )

    db.add(new_user)

    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=400,
            detail="Username or email already exists"
        )

    db.refresh(new_user)

    return {
        "message": "User registered successfully",
        "user_id": new_user.id,
        "username": new_user.username,
        "email": new_user.email,
        "role": new_user.role
    }


# ==========================================
# LOGIN
# ==========================================

@router.post("/login")
def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):

    # Rate-limit per IP + username: max 10 attempts per 60s
    client_ip = request.client.host if request.client else "unknown"
    rate_key = f"{client_ip}:{form_data.username}"
    now = time.time()
    if _check_login_rate_limit(rate_key, now):
        raise HTTPException(
            status_code=429,
            detail="Too many login attempts. Try again later."
        )
    _record_login_attempt(rate_key, now)

    # Find user
    user = db.query(User).filter(
        User.username == form_data.username
    ).first()

    # Check username/password
    if not user or not verify_password(
        form_data.password,
        user.password_hash
    ):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password"
        )

    # Check active account
    if not user.is_active:
        raise HTTPException(
            status_code=403,
            detail="User account is inactive"
        )

    # Create JWT (sub=user.id, with username claim)
    access_token = create_access_token(
        user.id,
        user.username
    )

    return {
        "access_token": access_token,
        "token_type": "bearer"
    }


# ==========================================
# CURRENT USER
# ==========================================

@router.get("/me")
def get_me(
    current_user: User = Depends(get_current_user)
):

    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "role": current_user.role,
        "is_active": current_user.is_active
    }