"""
=============================================================================
models.py — Modèles SQLAlchemy (miroir du schéma init_schema.sql)
=============================================================================
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, Float, String, Text, DateTime,
    ForeignKey, UniqueConstraint, Index,
)
from sqlalchemy.orm import relationship
from app.database import Base


class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, unique=True, nullable=False, index=True)
    filename = Column(String(512), nullable=False)
    category = Column(String(128), default="Unknown")
    split = Column(String(32), default="unknown")
    width = Column(Integer)
    height = Column(Integer)
    original_size_kb = Column(Float)
    original_path = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relations
    ground_truth = relationship("GroundTruth", uselist=False, back_populates="image")
    compressions = relationship("Compression", back_populates="image")
    predictions = relationship("Prediction", back_populates="image")
    metrics = relationship("Metric", back_populates="image")


class GroundTruth(Base):
    __tablename__ = "ground_truth"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey("images.image_id"), unique=True, nullable=False, index=True)
    gt_text = Column(Text)
    gt_text_normalized = Column(Text)
    num_annotations = Column(Integer, default=0)
    num_text_annotations = Column(Integer, default=0)
    num_characters = Column(Integer, default=0)
    num_words = Column(Integer, default=0)
    layout_types = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    image = relationship("Image", back_populates="ground_truth")


class Compression(Base):
    __tablename__ = "compressions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey("images.image_id"), nullable=False, index=True)
    compression_type = Column(String(32), nullable=False)
    compression_level = Column(Integer)
    quality_label = Column(String(32))
    bitrate_bpp = Column(Float)
    file_size_kb = Column(Float)
    compression_ratio = Column(Float)
    ssim = Column(Float)
    compressed_path = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("image_id", "compression_type", "compression_level",
                         name="uq_image_compression"),
    )

    image = relationship("Image", back_populates="compressions")
    predictions = relationship("Prediction", back_populates="compression")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey("images.image_id"), nullable=False, index=True)
    compression_id = Column(Integer, ForeignKey("compressions.id"), nullable=True)
    vlm_name = Column(String(64), nullable=False, index=True)
    compression_type = Column(String(32), nullable=False)
    compression_level = Column(Integer, nullable=True)
    prompt_used = Column(Text)
    predicted_text = Column(Text)
    inference_time_s = Column(Float)
    num_tokens_generated = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("image_id", "vlm_name", "compression_type", "compression_level",
                         name="uq_prediction"),
    )

    image = relationship("Image", back_populates="predictions")
    compression = relationship("Compression", back_populates="predictions")
    metric = relationship("Metric", uselist=False, back_populates="prediction")


class Metric(Base):
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), unique=True, nullable=False, index=True)
    image_id = Column(Integer, ForeignKey("images.image_id"), nullable=False, index=True)
    vlm_name = Column(String(64), nullable=False, index=True)
    compression_type = Column(String(32), nullable=False)
    compression_level = Column(Integer, nullable=True)
    category = Column(String(128))
    cer = Column(Float)
    wer = Column(Float)
    bleu = Column(Float)
    gt_length = Column(Integer)
    pred_length = Column(Integer)
    length_ratio = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    prediction = relationship("Prediction", back_populates="metric")
    image = relationship("Image", back_populates="metrics")
