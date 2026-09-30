import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, ForeignKey, CheckConstraint
from sqlalchemy.sql import func
try:
    from database import Base
except ImportError:
    try:
        from app.database import Base
    except ImportError:
        from backend.app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(150), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), default="inspector", nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "role IN ('inspector', 'supervisor', 'admin')",
            name="ck_users_role",
        ),
    )


class Inspection(Base):
    __tablename__ = "inspections"

    id = Column(Integer, primary_key=True, index=True)

    image_path = Column(String(500), nullable=False)

    # Phase 2: multi-category inspection (default bottle, backward compat).
    product_category = Column(String(100), nullable=True, default="bottle")

    defect_type = Column(String(100), nullable=True)

    confidence = Column(Float, nullable=True)

    status = Column(String(50), default="pending")

    inspected_by = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=True
    )

    # Severity information
    severity_score = Column(Float, nullable=True)
    severity_level = Column(String(50), nullable=True)

    # Supervisor override fields
    is_overridden = Column(Boolean, default=False, nullable=False)
    overridden_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    override_reason = Column(String(500), nullable=True)
    original_status = Column(String(50), nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )


class SystemSettings(Base):
    __tablename__ = "system_settings"

    id = Column(Integer, primary_key=True, index=True)
    setting_key = Column(String(100), unique=True, nullable=False)
    setting_value = Column(String(255), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())