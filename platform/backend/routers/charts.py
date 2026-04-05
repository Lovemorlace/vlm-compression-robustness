"""
=============================================================================
routers/charts.py — Endpoints données agrégées pour les graphiques
=============================================================================
GET /api/charts/degradation     — Courbes de dégradation WER/CER/BLEU vs compression
GET /api/charts/heatmap         — Heatmap catégorie × condition
GET /api/charts/iso-bitrate     — Comparaison JPEG vs Neural à bitrate comparable
GET /api/charts/distribution    — Distribution des scores (histogramme)
GET /api/charts/vlm-comparison  — Comparaison inter-VLMs par condition
=============================================================================
"""

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, case, distinct

from app.database import get_db
from app.models import Image, Compression, Prediction, Metric

router = APIRouter(prefix="/api/charts", tags=["charts"])


def _condition_label_expr():
    """Expression SQL pour construire le label de condition."""
    return case(
        (Metric.compression_type == "baseline", "baseline"),
        (Metric.compression_type == "jpeg",
         func.concat("jpeg_QF", Metric.compression_level)),
        (Metric.compression_type == "neural",
         func.concat("neural_q", Metric.compression_level)),
        else_=func.concat(Metric.compression_type, "_", Metric.compression_level),
    )


# ============================================================================
# GET /api/charts/degradation
# ============================================================================

@router.get("/degradation")
def get_degradation_curves(
    metric_name: str = Query("wer", description="Métrique : cer, wer, bleu"),
    vlm_name: Optional[str] = Query(None, description="Filtrer par VLM (None = tous)"),
    category: Optional[str] = Query(None, description="Filtrer par catégorie"),
    compression_type: Optional[str] = Query(None, description="jpeg, neural, ou None (les deux)"),
    db: Session = Depends(get_db),
):
    """
    Courbes de dégradation — score moyen en fonction du niveau de compression.
    Utilisé par le Graphique 1 de l'Écran 3.

    Retourne une série par VLM, ordonnée par niveau de compression croissant.
    Le point baseline est toujours inclus comme référence.
    """

    # Choisir la colonne métrique
    metric_col_map = {
        "cer": Metric.cer,
        "wer": Metric.wer,
        "bleu": Metric.bleu,
    }
    if metric_name not in metric_col_map:
        metric_name = "wer"
    metric_col = metric_col_map[metric_name]

    # Requête de base
    query = db.query(
        Metric.vlm_name,
        Metric.compression_type,
        Metric.compression_level,
        _condition_label_expr().label("condition_label"),
        func.count(Metric.id).label("n_images"),
        func.avg(metric_col).label("mean"),
        func.stddev(metric_col).label("std"),
    )

    if vlm_name:
        query = query.filter(Metric.vlm_name == vlm_name)
    if category:
        query = query.filter(Metric.category == category)
    if compression_type:
        # Inclure baseline + le type demandé
        query = query.filter(
            Metric.compression_type.in_(["baseline", compression_type])
        )

    rows = (
        query.group_by(
            Metric.vlm_name,
            Metric.compression_type,
            Metric.compression_level,
        )
        .order_by(Metric.vlm_name, Metric.compression_type, Metric.compression_level)
        .all()
    )

    # Structurer par VLM
    series = {}
    for row in rows:
        vlm = row.vlm_name
        if vlm not in series:
            series[vlm] = []

        series[vlm].append({
            "compression_type": row.compression_type,
            "compression_level": row.compression_level,
            "condition_label": row.condition_label,
            "n_images": row.n_images,
            "mean": float(row.mean) if row.mean else 0,
            "std": float(row.std) if row.std else 0,
        })

    return {
        "metric": metric_name,
        "filters": {
            "vlm_name": vlm_name,
            "category": category,
            "compression_type": compression_type,
        },
        "series": series,
    }


# ============================================================================
# GET /api/charts/heatmap
# ============================================================================

@router.get("/heatmap")
def get_heatmap_data(
    metric_name: str = Query("bleu", description="Métrique : cer, wer, bleu"),
    vlm_name: Optional[str] = Query(None, description="Filtrer par VLM"),
    db: Session = Depends(get_db),
):
    """
    Heatmap catégorie × condition de compression.
    Utilisé par le Graphique 3 de l'Écran 3.

    Retourne une matrice : chaque cellule = score moyen pour
    (catégorie, condition, VLM).
    """

    metric_col_map = {"cer": Metric.cer, "wer": Metric.wer, "bleu": Metric.bleu}
    metric_col = metric_col_map.get(metric_name, Metric.bleu)

    query = db.query(
        Metric.vlm_name,
        Metric.compression_type,
        Metric.compression_level,
        _condition_label_expr().label("condition_label"),
        Metric.category,
        func.count(Metric.id).label("n_images"),
        func.avg(metric_col).label("value"),
    )

    if vlm_name:
        query = query.filter(Metric.vlm_name == vlm_name)

    # Exclure les catégories Unknown

    rows = (
        query.group_by(
            Metric.vlm_name,
            Metric.compression_type,
            Metric.compression_level,
            Metric.category,
        )
        .order_by(Metric.vlm_name, Metric.category)
        .all()
    )

    # Structurer en format heatmap
    # { vlm_name: { category: { condition_label: value } } }
    heatmaps = {}
    categories = set()
    conditions_set = set()

    for row in rows:
        vlm = row.vlm_name
        cat = row.category
        cond = row.condition_label

        categories.add(cat)
        conditions_set.add(cond)

        if vlm not in heatmaps:
            heatmaps[vlm] = {}
        if cat not in heatmaps[vlm]:
            heatmaps[vlm][cat] = {}

        heatmaps[vlm][cat][cond] = {
            "value": float(row.value) if row.value else 0,
            "n_images": row.n_images,
        }

    # Ordonner les conditions
    condition_order = ["baseline"]
    for qf in [90, 70, 50, 30, 10]:
        label = f"jpeg_QF{qf}"
        if label in conditions_set:
            condition_order.append(label)
    for ql in [1, 3, 6]:
        label = f"neural_q{ql}"
        if label in conditions_set:
            condition_order.append(label)

    return {
        "metric": metric_name,
        "categories": sorted(categories),
        "conditions": condition_order,
        "heatmaps": heatmaps,
    }


# ============================================================================
# GET /api/charts/iso-bitrate
# ============================================================================

@router.get("/iso-bitrate")
def get_iso_bitrate_comparison(
    metric_name: str = Query("bleu", description="Métrique : cer, wer, bleu"),
    vlm_name: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """
    Comparaison JPEG vs Neural à bitrate comparable.
    Utilisé par le Graphique 2 de l'Écran 3.

    Joint les métriques avec la table compressions pour récupérer
    le bitrate réel (bpp) de chaque image compressée.
    """

    metric_col_map = {"cer": Metric.cer, "wer": Metric.wer, "bleu": Metric.bleu}
    metric_col = metric_col_map.get(metric_name, Metric.bleu)

    # Joindre metrics → predictions → compressions pour le bitrate
    query = db.query(
        Metric.vlm_name,
        Metric.compression_type,
        Metric.compression_level,
        _condition_label_expr().label("condition_label"),
        func.count(Metric.id).label("n_images"),
        func.avg(Compression.bitrate_bpp).label("bpp_mean"),
        func.stddev(Compression.bitrate_bpp).label("bpp_std"),
        func.avg(metric_col).label("metric_mean"),
        func.stddev(metric_col).label("metric_std"),
        func.avg(Compression.ssim).label("ssim_mean"),
    ).join(
        Prediction, Metric.prediction_id == Prediction.id
    ).join(
        Compression, Prediction.compression_id == Compression.id
    ).filter(
        Metric.compression_type.in_(["jpeg", "neural"])
    )

    if vlm_name:
        query = query.filter(Metric.vlm_name == vlm_name)
    if category:
        query = query.filter(Metric.category == category)

    rows = (
        query.group_by(
            Metric.vlm_name,
            Metric.compression_type,
            Metric.compression_level,
        )
        .order_by(Metric.vlm_name, Metric.compression_type, Metric.compression_level)
        .all()
    )

    # Structurer par VLM
    series = {}
    for row in rows:
        vlm = row.vlm_name
        if vlm not in series:
            series[vlm] = {"jpeg": [], "neural": []}

        entry = {
            "compression_level": row.compression_level,
            "condition_label": row.condition_label,
            "n_images": row.n_images,
            "bpp_mean": float(row.bpp_mean) if row.bpp_mean else 0,
            "bpp_std": float(row.bpp_std) if row.bpp_std else 0,
            "metric_mean": float(row.metric_mean) if row.metric_mean else 0,
            "metric_std": float(row.metric_std) if row.metric_std else 0,
            "ssim_mean": float(row.ssim_mean) if row.ssim_mean else 0,
        }
        series[vlm][row.compression_type].append(entry)

    return {
        "metric": metric_name,
        "filters": {"vlm_name": vlm_name, "category": category},
        "series": series,
    }


# ============================================================================
# GET /api/charts/distribution
# ============================================================================

@router.get("/distribution")
def get_score_distribution(
    metric_name: str = Query("wer", description="Métrique : cer, wer, bleu"),
    vlm_name: str = Query(..., description="VLM"),
    compression_type: str = Query(..., description="baseline, jpeg, neural"),
    compression_level: Optional[int] = Query(None),
    category: Optional[str] = Query(None),
    n_bins: int = Query(20, ge=5, le=100, description="Nombre de bins"),
    db: Session = Depends(get_db),
):
    """
    Distribution des scores (pour histogramme).
    Retourne les valeurs brutes pour que le frontend construise l'histogramme.
    """

    metric_col_map = {"cer": Metric.cer, "wer": Metric.wer, "bleu": Metric.bleu}
    metric_col = metric_col_map.get(metric_name, Metric.wer)

    query = db.query(metric_col).filter(
        Metric.vlm_name == vlm_name,
        Metric.compression_type == compression_type,
    )

    if compression_level is not None:
        query = query.filter(Metric.compression_level == compression_level)
    else:
        query = query.filter(Metric.compression_level.is_(None))

    if category:
        query = query.filter(Metric.category == category)

    values = [r[0] for r in query.all() if r[0] is not None]

    if not values:
        return {
            "metric": metric_name,
            "n_values": 0,
            "values": [],
            "stats": {},
        }

    import numpy as np
    arr = np.array(values)

    return {
        "metric": metric_name,
        "n_values": len(values),
        "values": values,
        "stats": {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
        },
    }


# ============================================================================
# GET /api/charts/vlm-comparison
# ============================================================================

@router.get("/vlm-comparison")
def get_vlm_comparison(
    metric_name: str = Query("wer", description="Métrique : cer, wer, bleu"),
    category: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """
    Comparaison inter-VLMs : score moyen par condition pour chaque VLM.
    Utilisé pour l'Axe 5 — Sensibilité architecturale.
    """

    metric_col_map = {"cer": Metric.cer, "wer": Metric.wer, "bleu": Metric.bleu}
    metric_col = metric_col_map.get(metric_name, Metric.wer)

    query = db.query(
        Metric.vlm_name,
        Metric.compression_type,
        Metric.compression_level,
        _condition_label_expr().label("condition_label"),
        func.count(Metric.id).label("n_images"),
        func.avg(metric_col).label("mean"),
        func.stddev(metric_col).label("std"),
    )

    if category:
        query = query.filter(Metric.category == category)

    rows = (
        query.group_by(
            Metric.vlm_name,
            Metric.compression_type,
            Metric.compression_level,
        )
        .order_by(Metric.compression_type, Metric.compression_level)
        .all()
    )

    # Pivoter : condition → { vlm1: score, vlm2: score }
    comparison = {}
    vlm_names = set()

    for row in rows:
        cond = row.condition_label
        vlm = row.vlm_name
        vlm_names.add(vlm)

        if cond not in comparison:
            comparison[cond] = {}

        comparison[cond][vlm] = {
            "mean": float(row.mean) if row.mean else 0,
            "std": float(row.std) if row.std else 0,
            "n_images": row.n_images,
        }

    # Ordonner les conditions
    condition_order = ["baseline"]
    for qf in [90, 70, 50, 30, 10]:
        label = f"jpeg_QF{qf}"
        if label in comparison:
            condition_order.append(label)
    for ql in [1, 3, 6]:
        label = f"neural_q{ql}"
        if label in comparison:
            condition_order.append(label)

    ordered_data = []
    for cond in condition_order:
        if cond in comparison:
            entry = {"condition": cond}
            entry.update(comparison[cond])
            ordered_data.append(entry)

    return {
        "metric": metric_name,
        "vlm_names": sorted(vlm_names),
        "data": ordered_data,
    }
