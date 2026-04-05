"""
=============================================================================
routers/results.py — Endpoints résultats filtrés et vue par image
=============================================================================
GET  /api/results                — Liste filtrée + paginée (Écran 2)
GET  /api/results/{image_id}    — Détail d'une image (tous VLMs, toutes conditions)
GET  /api/results/{image_id}/detail — Détail pour un VLM + condition précis
GET  /api/images/serve           — Servir un fichier image (original ou compressé)
GET  /api/images/list            — Liste des images avec résumé des métriques
=============================================================================
"""

import os
DATA_BASE_DIR = os.getenv("DATA_BASE_DIR", "/home/mor/projet")
import math
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from app.database import get_db
from app.models import Image, GroundTruth, Compression, Prediction, Metric
from app.schemas import ImageResultDetail, PaginatedResponse

router = APIRouter(prefix="/api", tags=["results"])


def _make_condition_label(comp_type: str, comp_level: Optional[int]) -> str:
    if comp_type == "baseline":
        return "baseline"
    elif comp_type == "jpeg":
        return f"jpeg_QF{comp_level}"
    elif comp_type == "neural":
        return f"neural_q{comp_level}"
    return f"{comp_type}_{comp_level}"


# ============================================================================
# GET /api/results — Liste filtrée et paginée
# ============================================================================

@router.get("/results")
def get_results(
    vlm_name: Optional[str] = Query(None, description="Filtrer par VLM : qwen2-vl, internvl2"),
    compression_type: Optional[str] = Query(None, description="baseline, jpeg, neural"),
    compression_level: Optional[int] = Query(None, description="Niveau : QF ou quality"),
    category: Optional[str] = Query(None, description="Catégorie documentaire"),
    sort_by: str = Query("image_id", description="Tri : image_id, cer, wer, bleu"),
    sort_order: str = Query("asc", description="asc ou desc"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """
    Liste paginée des résultats filtrés.
    Retourne les métriques avec les métadonnées image pour chaque combinaison
    (image, VLM, condition).
    """

    # Construire la requête
    query = (
        db.query(Metric, Image)
        .join(Image, Metric.image_id == Image.image_id)
    )

    # Filtres
    if vlm_name:
        query = query.filter(Metric.vlm_name == vlm_name)
    if compression_type:
        query = query.filter(Metric.compression_type == compression_type)
    if compression_level is not None:
        query = query.filter(Metric.compression_level == compression_level)
    if category:
        query = query.filter(Metric.category == category)

    # Comptage total
    total = query.count()

    # Tri
    sort_col_map = {
        "image_id": Metric.image_id,
        "cer": Metric.cer,
        "wer": Metric.wer,
        "bleu": Metric.bleu,
        "category": Metric.category,
    }
    sort_col = sort_col_map.get(sort_by, Metric.image_id)
    if sort_order == "desc":
        query = query.order_by(sort_col.desc())
    else:
        query = query.order_by(sort_col.asc())

    # Pagination
    offset = (page - 1) * page_size
    rows = query.offset(offset).limit(page_size).all()

    # Formater
    data = []
    for metric, img in rows:
        data.append({
            "image_id": img.image_id,
            "filename": img.filename,
            "category": img.category,
            "vlm_name": metric.vlm_name,
            "compression_type": metric.compression_type,
            "compression_level": metric.compression_level,
            "condition_label": _make_condition_label(
                metric.compression_type, metric.compression_level
            ),
            "cer": metric.cer,
            "wer": metric.wer,
            "bleu": metric.bleu,
            "gt_length": metric.gt_length,
            "pred_length": metric.pred_length,
        })

    total_pages = math.ceil(total / page_size) if total > 0 else 0

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "data": data,
    }


# ============================================================================
# GET /api/results/{image_id} — Tous les résultats pour une image
# ============================================================================

@router.get("/results/{image_id}")
def get_image_results(
    image_id: int,
    vlm_name: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """
    Retourne tous les résultats (tous VLMs, toutes conditions) pour une image.
    Utilisé par l'Écran 2 — Vue par image.
    """

    # Vérifier que l'image existe
    image = db.query(Image).filter(Image.image_id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail=f"Image {image_id} non trouvée")

    # Ground-truth
    gt = db.query(GroundTruth).filter(GroundTruth.image_id == image_id).first()

    # Toutes les prédictions + métriques pour cette image
    query = (
        db.query(Prediction, Metric, Compression)
        .outerjoin(Metric, Prediction.id == Metric.prediction_id)
        .outerjoin(Compression, Prediction.compression_id == Compression.id)
        .filter(Prediction.image_id == image_id)
    )

    if vlm_name:
        query = query.filter(Prediction.vlm_name == vlm_name)

    query = query.order_by(
        Prediction.vlm_name,
        Prediction.compression_type,
        Prediction.compression_level,
    )

    rows = query.all()

    # Construire la réponse
    results = []
    for pred, metric, comp in rows:
        results.append(ImageResultDetail(
            image_id=image.image_id,
            filename=image.filename,
            category=image.category,
            width=image.width,
            height=image.height,
            original_size_kb=image.original_size_kb,
            original_path=image.original_path,
            gt_text=gt.gt_text if gt else None,
            gt_num_chars=gt.num_characters if gt else None,
            layout_types=gt.layout_types if gt else None,
            compression_type=pred.compression_type,
            compression_level=pred.compression_level,
            condition_label=_make_condition_label(
                pred.compression_type, pred.compression_level
            ),
            quality_label=comp.quality_label if comp else None,
            bitrate_bpp=comp.bitrate_bpp if comp else None,
            compressed_size_kb=comp.file_size_kb if comp else None,
            compression_ratio=comp.compression_ratio if comp else None,
            ssim=comp.ssim if comp else None,
            compressed_path=comp.compressed_path if comp else None,
            vlm_name=pred.vlm_name,
            predicted_text=pred.predicted_text,
            inference_time_s=pred.inference_time_s,
            num_tokens_generated=pred.num_tokens_generated,
            cer=metric.cer if metric else None,
            wer=metric.wer if metric else None,
            bleu=metric.bleu if metric else None,
        ))

    return {
        "image": {
            "image_id": image.image_id,
            "filename": image.filename,
            "category": image.category,
            "width": image.width,
            "height": image.height,
            "original_size_kb": image.original_size_kb,
        },
        "ground_truth": {
            "gt_text": gt.gt_text if gt else None,
            "num_characters": gt.num_characters if gt else None,
            "num_words": gt.num_words if gt else None,
            "layout_types": gt.layout_types if gt else None,
        } if gt else None,
        "results": [r.model_dump() for r in results],
    }


# ============================================================================
# GET /api/results/{image_id}/detail — Résultat pour un VLM + condition
# ============================================================================

@router.get("/results/{image_id}/detail")
def get_image_result_detail(
    image_id: int,
    vlm_name: str = Query(..., description="VLM : qwen2-vl ou internvl2"),
    compression_type: str = Query(..., description="baseline, jpeg, neural"),
    compression_level: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    """
    Résultat détaillé pour une image + un VLM + une condition précise.
    Endpoint ciblé utilisé par la vue comparative de l'Écran 2.
    """

    # Trouver la prédiction
    pred_query = (
        db.query(Prediction)
        .filter(
            Prediction.image_id == image_id,
            Prediction.vlm_name == vlm_name,
            Prediction.compression_type == compression_type,
        )
    )

    if compression_level is not None:
        pred_query = pred_query.filter(Prediction.compression_level == compression_level)
    else:
        pred_query = pred_query.filter(Prediction.compression_level.is_(None))

    pred = pred_query.first()
    if not pred:
        raise HTTPException(
            status_code=404,
            detail=f"Pas de prédiction pour image={image_id}, vlm={vlm_name}, "
                   f"type={compression_type}, level={compression_level}"
        )

    # Image + GT + Compression + Metric
    image = db.query(Image).filter(Image.image_id == image_id).first()
    gt = db.query(GroundTruth).filter(GroundTruth.image_id == image_id).first()
    comp = db.query(Compression).filter(Compression.id == pred.compression_id).first() if pred.compression_id else None
    metric = db.query(Metric).filter(Metric.prediction_id == pred.id).first()

    return ImageResultDetail(
        image_id=image.image_id,
        filename=image.filename,
        category=image.category,
        width=image.width,
        height=image.height,
        original_size_kb=image.original_size_kb,
        original_path=image.original_path,
        gt_text=gt.gt_text if gt else None,
        gt_num_chars=gt.num_characters if gt else None,
        layout_types=gt.layout_types if gt else None,
        compression_type=pred.compression_type,
        compression_level=pred.compression_level,
        condition_label=_make_condition_label(pred.compression_type, pred.compression_level),
        quality_label=comp.quality_label if comp else None,
        bitrate_bpp=comp.bitrate_bpp if comp else None,
        compressed_size_kb=comp.file_size_kb if comp else None,
        compression_ratio=comp.compression_ratio if comp else None,
        ssim=comp.ssim if comp else None,
        compressed_path=comp.compressed_path if comp else None,
        vlm_name=pred.vlm_name,
        predicted_text=pred.predicted_text,
        inference_time_s=pred.inference_time_s,
        num_tokens_generated=pred.num_tokens_generated,
        cer=metric.cer if metric else None,
        wer=metric.wer if metric else None,
        bleu=metric.bleu if metric else None,
    ).model_dump()


# ============================================================================
# GET /api/images/serve — Servir un fichier image
# ============================================================================

@router.get("/images/serve")

def serve_image(
    path: str = Query(..., description="Chemin absolu du fichier image"),
):
    """
    Sert un fichier image (original ou compressé) pour affichage frontend.
    Le chemin est vérifié pour éviter la traversée de répertoire.
    """

    # Sécurité : normaliser le chemin
    if not os.path.isabs(path):
        path = os.path.join(DATA_BASE_DIR, path)
    real_path = os.path.realpath(path)

    if not os.path.exists(real_path):
        raise HTTPException(status_code=404, detail="Fichier non trouvé")

    if not os.path.isfile(real_path):
        raise HTTPException(status_code=400, detail="Le chemin n'est pas un fichier")

    # Vérifier l'extension
    ext = os.path.splitext(real_path)[1].lower()
    allowed_ext = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if ext not in allowed_ext:
        raise HTTPException(status_code=400, detail=f"Extension non autorisée : {ext}")

    # Types MIME
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
    }

    return FileResponse(
        real_path,
        media_type=mime_map.get(ext, "application/octet-stream"),
    )


# ============================================================================
# GET /api/images/list — Liste des images avec résumé
# ============================================================================

@router.get("/images/list")
def list_images(
    category: Optional[str] = Query(None),
    split: Optional[str] = Query(None),
    has_results: bool = Query(True, description="Uniquement les images avec des résultats"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Liste des images avec le nombre de résultats associés."""

    query = db.query(Image)

    if category:
        query = query.filter(Image.category == category)
    if split:
        query = query.filter(Image.split == split)

    if has_results:
        image_ids_with_results = (
            db.query(Metric.image_id).distinct().subquery()
        )
        query = query.filter(Image.image_id.in_(
            db.query(image_ids_with_results)
        ))

    total = query.count()
    offset = (page - 1) * page_size
    images = query.order_by(Image.image_id).offset(offset).limit(page_size).all()

    data = []
    for img in images:
        n_results = (
            db.query(func.count(Metric.id))
            .filter(Metric.image_id == img.image_id)
            .scalar() or 0
        )
        data.append({
            "image_id": img.image_id,
            "filename": img.filename,
            "category": img.category,
            "split": img.split,
            "width": img.width,
            "height": img.height,
            "original_size_kb": img.original_size_kb,
            "n_results": n_results,
        })

    total_pages = math.ceil(total / page_size) if total > 0 else 0

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "data": data,
    }
