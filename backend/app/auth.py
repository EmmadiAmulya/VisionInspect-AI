from datetime import datetime, timedelta, timezone

import jwt
from pwdlib import PasswordHash
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from database import get_db
from models import User


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

SECRET_KEY = "visioninspect-secret-key-change-later"

ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES = 60


oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/login"
)


# ==========================================
# Create JWT Access Token
# ==========================================

def create_access_token(username: str) -> str:

    expire = datetime.now(timezone.utc) + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    payload = {
        "sub": username,
        "exp": expire
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
    db=Depends(get_db)
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

        username = payload.get("sub")

        if username is None:
            raise credentials_exception

    except jwt.InvalidTokenError:

        raise credentials_exception


    user = db.query(User).filter(
        User.username == username
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

    if current_user.role != "inspector":

        raise HTTPException(
            status_code=403,
            detail="Inspector access required"
        )

    return current_user


# ==========================================
# Supervisor Role
# ==========================================

def require_supervisor(
    current_user: User = Depends(get_current_user)
):

    if current_user.role != "supervisor":

        raise HTTPException(
            status_code=403,
            detail="Supervisor access required"
        )

    return current_user