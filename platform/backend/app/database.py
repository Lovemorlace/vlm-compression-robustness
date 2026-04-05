"""
=============================================================================
database.py — Connexion PostgreSQL et modèles SQLAlchemy
=============================================================================
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://vlm_user:changeme@localhost:5432/vlm_compression"
)

engine = create_engine(DATABASE_URL, echo=False, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Générateur de session pour FastAPI Depends()."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
