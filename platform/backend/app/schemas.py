"""
=============================================================================
schemas.py — Schémas Pydantic (réponses API)
=============================================================================
"""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


# ============================================================================
# Réponses unitaires
# ============================================================================

class ImageSchema(BaseModel):
    image_id: int
    filename: str
    category: str
    split: str
    width: Optional[int] = None
    height: Optional[int] = None
    original_size_kb: Optional[float] = None

    class Config:
        from_attributes = True


class CompressionSchema(BaseModel):
    id: int
    image_id: int
    compression_type: str
    compression_level: Optional[int] = None
    quality_label: Optional[str] = None
    bitrate_bpp: Optional[float] = None
    file_size_kb: Optional[float] = None
    compression_ratio: Optional[float] = None
    ssim: Optional[float] = None

    class Config:
        from_attributes = True


class PredictionSchema(BaseModel):
    id: int
    image_id: int
    vlm_name: str
    compression_type: str
    compression_level: Optional[int] = None
    predicted_text: Optional[str] = None
    inference_time_s: Optional[float] = None
    num_tokens_generated: Optional[int] = None

    class Config:
        from_attributes = True


class MetricSchema(BaseModel):
    prediction_id: int
    image_id: int
    vlm_name: str
    compression_type: str
    compression_level: Optional[int] = None
    category: Optional[str] = None
    cer: Optional[float] = None
    wer: Optional[float] = None
    bleu: Optional[float] = None
    gt_length: Optional[int] = None
    pred_length: Optional[int] = None
    length_ratio: Optional[float] = None

    class Config:
        from_attributes = True


class GroundTruthSchema(BaseModel):
    image_id: int
    gt_text: Optional[str] = None
    num_characters: Optional[int] = None
    num_words: Optional[int] = None
    layout_types: Optional[str] = None

    class Config:
        from_attributes = True


# ============================================================================
# Vue par image — réponse enrichie (Écran 2 de la plateforme)
# ============================================================================

class ImageResultDetail(BaseModel):
    """Résultat complet pour une image + condition + VLM."""
    image_id: int
    filename: str
    category: str
    width: Optional[int] = None
    height: Optional[int] = None
    original_size_kb: Optional[float] = None
    original_path: Optional[str] = None

    # Ground-truth
    gt_text: Optional[str] = None
    gt_num_chars: Optional[int] = None
    layout_types: Optional[str] = None

    # Compression
    compression_type: str
    compression_level: Optional[int] = None
    condition_label: str
    quality_label: Optional[str] = None
    bitrate_bpp: Optional[float] = None
    compressed_size_kb: Optional[float] = None
    compression_ratio: Optional[float] = None
    ssim: Optional[float] = None
    compressed_path: Optional[str] = None

    # Prédiction
    vlm_name: str
    predicted_text: Optional[str] = None
    inference_time_s: Optional[float] = None
    num_tokens_generated: Optional[int] = None

    # Métriques
    cer: Optional[float] = None
    wer: Optional[float] = None
    bleu: Optional[float] = None


# ============================================================================
# Données agrégées — réponses graphiques (Écran 3)
# ============================================================================

class AggregatedCondition(BaseModel):
    """Une ligne du graphique courbe de dégradation."""
    vlm_name: str
    compression_type: str
    compression_level: Optional[int] = None
    condition_label: str
    n_images: int
    cer_mean: float
    cer_std: Optional[float] = None
    wer_mean: float
    wer_std: Optional[float] = None
    bleu_mean: float
    bleu_std: Optional[float] = None


class AggregatedCategory(BaseModel):
    """Une cellule de la heatmap catégorie × condition."""
    vlm_name: str
    compression_type: str
    compression_level: Optional[int] = None
    condition_label: str
    category: str
    n_images: int
    cer_mean: float
    wer_mean: float
    bleu_mean: float


class IsoBitrateRow(BaseModel):
    """Une ligne de la comparaison JPEG vs Neural à iso-bitrate."""
    vlm_name: str
    compression_type: str
    compression_level: Optional[int] = None
    condition_label: str
    n_images: int
    bpp_mean: float
    bpp_std: Optional[float] = None
    cer_mean: float
    wer_mean: float
    bleu_mean: float


# ============================================================================
# Métadonnées et filtres
# ============================================================================

class ProjectStats(BaseModel):
    total_images: int
    total_ground_truths: int
    total_compressions: int
    total_predictions: int
    total_metrics: int
    vlm_names: list[str]
    categories: list[str]
    compression_types: list[str]
    jpeg_levels: list[int]
    neural_levels: list[int]


class FilterOptions(BaseModel):
    """Options disponibles pour les filtres de la plateforme (Écran 1)."""
    categories: list[str]
    vlm_names: list[str]
    compression_conditions: list[dict]


# ============================================================================
# Pagination
# ============================================================================

class PaginatedResponse(BaseModel):
    total: int
    page: int
    page_size: int
    total_pages: int
    data: list
