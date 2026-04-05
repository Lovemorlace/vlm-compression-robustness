"""
=============================================================================
routers/meta.py — Endpoints métadonnées et filtres
=============================================================================
GET /api/stats          — Statistiques globales du projet
GET /api/filters        — Options de filtres disponibles (Écran 1)
GET /api/health         — Health check
=============================================================================
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, distinct

from app.database import get_db
from app.models import Image, GroundTruth, Compression, Prediction, Metric
from app.schemas import ProjectStats, FilterOptions

router = APIRouter(prefix="/api", tags=["meta"])


@router.get("/health")
def health_check():
    return {"status": "ok", "service": "vlm-compression-platform"}


@router.get("/stats", response_model=ProjectStats)
def get_project_stats(db: Session = Depends(get_db)):
    """Statistiques globales du projet — utile pour le dashboard."""

    total_images = db.query(func.count(Image.image_id)).scalar() or 0
    total_gt = db.query(func.count(GroundTruth.id)).scalar() or 0
    total_comp = db.query(func.count(Compression.id)).scalar() or 0
    total_pred = db.query(func.count(Prediction.id)).scalar() or 0
    total_met = db.query(func.count(Metric.id)).scalar() or 0

    vlm_names = sorted([
        r[0] for r in db.query(distinct(Prediction.vlm_name)).all() if r[0]
    ])
    categories = sorted([
        r[0] for r in db.query(distinct(Image.category)).all() if r[0]
    ])
    comp_types = sorted([
        r[0] for r in db.query(distinct(Compression.compression_type)).all() if r[0]
    ])

    jpeg_levels = sorted([
        r[0] for r in db.query(distinct(Compression.compression_level))
        .filter(Compression.compression_type == "jpeg").all() if r[0] is not None
    ])
    neural_levels = sorted([
        r[0] for r in db.query(distinct(Compression.compression_level))
        .filter(Compression.compression_type == "neural").all() if r[0] is not None
    ])

    return ProjectStats(
        total_images=total_images,
        total_ground_truths=total_gt,
        total_compressions=total_comp,
        total_predictions=total_pred,
        total_metrics=total_met,
        vlm_names=vlm_names,
        categories=categories,
        compression_types=comp_types,
        jpeg_levels=jpeg_levels,
        neural_levels=neural_levels,
    )


@router.get("/filters", response_model=FilterOptions)
def get_filter_options(db: Session = Depends(get_db)):
    """
    Options de filtres disponibles — alimente l'Écran 1 (sélection).
    Retourne les catégories, VLMs, et conditions de compression pour
    lesquels des résultats existent en base.
    """

    # Catégories avec résultats
    categories = sorted([
        r[0] for r in db.query(distinct(Metric.category)).all()
        if r[0] and r[0] != "Unknown"
    ])

    # VLMs avec résultats
    vlm_names = sorted([
        r[0] for r in db.query(distinct(Metric.vlm_name)).all() if r[0]
    ])

    # Conditions de compression disponibles
    conditions = []

    # Baseline
    n_baseline = (
        db.query(func.count(Metric.id))
        .filter(Metric.compression_type == "baseline")
        .scalar() or 0
    )
    if n_baseline > 0:
        conditions.append({
            "type": "baseline",
            "level": None,
            "label": "baseline",
            "count": n_baseline,
        })

    # JPEG
    jpeg_rows = (
        db.query(
            Metric.compression_level,
            func.count(Metric.id),
        )
        .filter(Metric.compression_type == "jpeg")
        .group_by(Metric.compression_level)
        .order_by(Metric.compression_level.desc())
        .all()
    )
    for level, count in jpeg_rows:
        conditions.append({
            "type": "jpeg",
            "level": level,
            "label": f"jpeg_QF{level}",
            "count": count,
        })

    # Neural
    neural_rows = (
        db.query(
            Metric.compression_level,
            func.count(Metric.id),
        )
        .filter(Metric.compression_type == "neural")
        .group_by(Metric.compression_level)
        .order_by(Metric.compression_level)
        .all()
    )
    for level, count in neural_rows:
        conditions.append({
            "type": "neural",
            "level": level,
            "label": f"neural_q{level}",
            "count": count,
        })

    return FilterOptions(
        categories=categories,
        vlm_names=vlm_names,
        compression_conditions=conditions,
    )
