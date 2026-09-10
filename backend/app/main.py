import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from routers.inspection_router import router as inspection_router
from routers.auth_router import router as auth_router

from database import Base, engine
import models


# ==========================================
# CREATE DATABASE TABLES
# ==========================================

Base.metadata.create_all(bind=engine)


# ==========================================
# CREATE FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="VisionInspect AI",
    description="Manufacturing Defect Detection & Quality Inspection System",
    version="1.0.0"
)


# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# UPLOAD DIRECTORY
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
# SERVE UPLOADED IMAGES
# ==========================================

app.mount(
    "/uploads",
    StaticFiles(directory=UPLOAD_DIR),
    name="uploads"
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
# ROOT
# ==========================================

@app.get("/")
def root():

    return {
        "message": "VisionInspect AI Backend is running"
    }