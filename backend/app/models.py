from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, ForeignKey
from sqlalchemy.sql import func
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(150), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), default="inspector", nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Inspection(Base):
    __tablename__ = "inspections"

    id = Column(Integer, primary_key=True, index=True)

    image_path = Column(String(500), nullable=False)

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

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )