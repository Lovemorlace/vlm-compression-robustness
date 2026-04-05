"""
=============================================================================
routers/export.py — Prompt 6.2
=============================================================================
Endpoints d'export :
  GET /api/export/csv              — Export CSV des résultats filtrés
  GET /api/export/csv/aggregated   — Export CSV des données agrégées
  GET /api/export/csv/heatmap      — Export CSV format heatmap (pivot)
  GET /api/export/csv/iso-bitrate  — Export CSV comparaison iso-bitrate
  GET /api/export/report           — Rapport de synthèse PDF complet
=============================================================================
"""

import io
import os
import math
import tempfile
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, case

from app.database import get_db
from app.models import Image, GroundTruth, Compression, Prediction, Metric

router = APIRouter(prefix="/api/export", tags=["export"])


# ============================================================================
# Helpers
# ============================================================================

def _condition_label(comp_type, comp_level):
    if comp_type == "baseline":
        return "baseline"
    elif comp_type == "jpeg":
        return f"jpeg_QF{comp_level}"
    elif comp_type == "neural":
        return f"neural_q{comp_level}"
    return f"{comp_type}_{comp_level}"


def _build_filtered_df(db: Session, vlm_name=None, compression_type=None,
                       compression_level=None, category=None) -> pd.DataFrame:
    """Construit un DataFrame filtré depuis les tables metrics + images + compressions."""

    query = (
        db.query(
            Metric.image_id,
            Image.filename,
            Image.category,
            Image.width,
            Image.height,
            Image.original_size_kb,
            Metric.vlm_name,
            Metric.compression_type,
            Metric.compression_level,
            Metric.cer,
            Metric.wer,
            Metric.bleu,
            Metric.gt_length,
            Metric.pred_length,
            Metric.length_ratio,
            Compression.bitrate_bpp,
            Compression.file_size_kb.label("compressed_size_kb"),
            Compression.compression_ratio,
            Compression.ssim,
        )
        .join(Image, Metric.image_id == Image.image_id)
        .outerjoin(Prediction, Metric.prediction_id == Prediction.id)
        .outerjoin(Compression, Prediction.compression_id == Compression.id)
    )

    if vlm_name:
        query = query.filter(Metric.vlm_name == vlm_name)
    if compression_type:
        query = query.filter(Metric.compression_type == compression_type)
    if compression_level is not None:
        query = query.filter(Metric.compression_level == compression_level)
    if category:
        query = query.filter(Metric.category == category)

    query = query.order_by(Metric.vlm_name, Metric.compression_type,
                           Metric.compression_level, Metric.image_id)

    rows = query.all()
    if not rows:
        return pd.DataFrame()

    columns = [
        "image_id", "filename", "category", "width", "height",
        "original_size_kb", "vlm_name", "compression_type",
        "compression_level", "cer", "wer", "bleu", "gt_length",
        "pred_length", "length_ratio", "bitrate_bpp",
        "compressed_size_kb", "compression_ratio", "ssim",
    ]
    df = pd.DataFrame(rows, columns=columns)

    df["condition_label"] = df.apply(
        lambda r: _condition_label(r["compression_type"], r["compression_level"]),
        axis=1,
    )

    return df


def _df_to_csv_response(df: pd.DataFrame, filename: str) -> StreamingResponse:
    """Convertit un DataFrame en réponse CSV streamée."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, float_format="%.6f")
    buffer.seek(0)

    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ============================================================================
# GET /api/export/csv — Résultats détaillés filtrés
# ============================================================================

@router.get("/csv")
def export_csv_results(
    vlm_name: Optional[str] = Query(None),
    compression_type: Optional[str] = Query(None),
    compression_level: Optional[int] = Query(None),
    category: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """
    Export CSV des résultats détaillés avec tous les filtres.
    Une ligne par (image, VLM, condition).
    """

    df = _build_filtered_df(db, vlm_name, compression_type,
                            compression_level, category)

    if df.empty:
        raise HTTPException(status_code=404, detail="Aucun résultat pour ces filtres")

    # Nom de fichier dynamique
    parts = ["results"]
    if vlm_name:
        parts.append(vlm_name)
    if compression_type:
        parts.append(compression_type)
        if compression_level is not None:
            parts.append(str(compression_level))
    if category:
        parts.append(category)
    parts.append(datetime.now().strftime("%Y%m%d"))
    filename = "_".join(parts) + ".csv"

    return _df_to_csv_response(df, filename)


# ============================================================================
# GET /api/export/csv/aggregated — Données agrégées par condition
# ============================================================================

@router.get("/csv/aggregated")
def export_csv_aggregated(
    vlm_name: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """
    Export CSV agrégé par VLM × compression_type × compression_level.
    Moyenne, écart-type, médiane de CER/WER/BLEU.
    """

    df = _build_filtered_df(db, vlm_name=vlm_name, category=category)
    if df.empty:
        raise HTTPException(status_code=404, detail="Aucun résultat")

    agg = (
        df.groupby(["vlm_name", "compression_type", "compression_level", "condition_label"])
        .agg(
            n_images=("cer", "size"),
            cer_mean=("cer", "mean"),
            cer_std=("cer", "std"),
            cer_median=("cer", "median"),
            wer_mean=("wer", "mean"),
            wer_std=("wer", "std"),
            wer_median=("wer", "median"),
            bleu_mean=("bleu", "mean"),
            bleu_std=("bleu", "std"),
            bleu_median=("bleu", "median"),
            bpp_mean=("bitrate_bpp", "mean"),
            ssim_mean=("ssim", "mean"),
        )
        .reset_index()
        .sort_values(["vlm_name", "compression_type", "compression_level"])
    )

    filename = f"aggregated_{vlm_name or 'all'}_{datetime.now():%Y%m%d}.csv"
    return _df_to_csv_response(agg, filename)


# ============================================================================
# GET /api/export/csv/heatmap — Format heatmap (pivot)
# ============================================================================

@router.get("/csv/heatmap")
def export_csv_heatmap(
    metric_name: str = Query("bleu", description="cer, wer, bleu"),
    vlm_name: str = Query(..., description="VLM obligatoire pour le pivot"),
    db: Session = Depends(get_db),
):
    """
    Export CSV au format pivot : catégorie (lignes) × condition (colonnes).
    Directement utilisable pour des figures de rapport.
    """

    df = _build_filtered_df(db, vlm_name=vlm_name)
    if df.empty:
        raise HTTPException(status_code=404, detail="Aucun résultat")

    if metric_name not in ("cer", "wer", "bleu"):
        metric_name = "bleu"

    # Exclure Unknown
    df = df[df["category"] != "Unknown"]

    pivot = df.pivot_table(
        index="category",
        columns="condition_label",
        values=metric_name,
        aggfunc="mean",
    )

    # Ordonner les colonnes
    col_order = ["baseline"]
    for qf in [90, 70, 50, 30, 10]:
        label = f"jpeg_QF{qf}"
        if label in pivot.columns:
            col_order.append(label)
    for ql in [1, 3, 6]:
        label = f"neural_q{ql}"
        if label in pivot.columns:
            col_order.append(label)
    col_order = [c for c in col_order if c in pivot.columns]
    pivot = pivot[col_order]

    # Convertir en CSV
    buffer = io.StringIO()
    pivot.to_csv(buffer, float_format="%.4f")
    buffer.seek(0)

    filename = f"heatmap_{vlm_name}_{metric_name}_{datetime.now():%Y%m%d}.csv"
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ============================================================================
# GET /api/export/csv/iso-bitrate — Comparaison JPEG vs Neural
# ============================================================================

@router.get("/csv/iso-bitrate")
def export_csv_iso_bitrate(
    vlm_name: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """
    Export CSV de la comparaison JPEG vs Neural à bitrate comparable.
    Inclut le bitrate moyen réel pour chaque condition.
    """

    df = _build_filtered_df(db, vlm_name=vlm_name)
    if df.empty:
        raise HTTPException(status_code=404, detail="Aucun résultat")

    # Filtrer uniquement jpeg et neural (pas baseline)
    compressed = df[df["compression_type"].isin(["jpeg", "neural"])].copy()
    if compressed.empty:
        raise HTTPException(status_code=404, detail="Aucune donnée compressée")

    agg = (
        compressed.groupby(["vlm_name", "compression_type", "compression_level", "condition_label"])
        .agg(
            n_images=("cer", "size"),
            bpp_mean=("bitrate_bpp", "mean"),
            bpp_std=("bitrate_bpp", "std"),
            cer_mean=("cer", "mean"),
            wer_mean=("wer", "mean"),
            bleu_mean=("bleu", "mean"),
            ssim_mean=("ssim", "mean"),
        )
        .reset_index()
        .sort_values(["vlm_name", "bpp_mean"])
    )

    filename = f"iso_bitrate_{vlm_name or 'all'}_{datetime.now():%Y%m%d}.csv"
    return _df_to_csv_response(agg, filename)


# ============================================================================
# GET /api/export/report — Rapport de synthèse PDF
# ============================================================================

@router.get("/report")
def export_synthesis_report(
    vlm_name: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """
    Génère un rapport de synthèse PDF complet.

    Contenu :
    - Page de garde
    - Statistiques globales du projet
    - Tableau baseline par VLM et catégorie
    - Tableau de dégradation JPEG
    - Tableau de dégradation Neural
    - Tableau comparatif JPEG vs Neural à iso-bitrate
    - Heatmap numérique (catégorie × condition)
    """

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, HRFlowable,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    # ------------------------------------------------------------------
    # 1. Charger les données
    # ------------------------------------------------------------------
    df = _build_filtered_df(db, vlm_name=vlm_name, category=category)
    if df.empty:
        raise HTTPException(status_code=404, detail="Aucun résultat pour générer le rapport")

    # Stats globales
    total_images = db.query(func.count(func.distinct(Metric.image_id))).scalar() or 0
    total_preds = db.query(func.count(Metric.id)).scalar() or 0
    vlm_names = sorted(df["vlm_name"].unique())
    categories = sorted([c for c in df["category"].unique() if c != "Unknown"])

    # ------------------------------------------------------------------
    # 2. Construire le PDF
    # ------------------------------------------------------------------
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp_path = tmp_file.name
    tmp_file.close()

    doc = SimpleDocTemplate(
        tmp_path,
        pagesize=A4,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    # Styles personnalisés
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=20,
        spaceAfter=12,
        alignment=TA_CENTER,
    )
    h1_style = ParagraphStyle(
        "CustomH1",
        parent=styles["Heading1"],
        fontSize=14,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor("#1a1a2e"),
    )
    h2_style = ParagraphStyle(
        "CustomH2",
        parent=styles["Heading2"],
        fontSize=12,
        spaceBefore=14,
        spaceAfter=8,
        textColor=colors.HexColor("#16213e"),
    )
    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["Normal"],
        fontSize=10,
        spaceAfter=6,
        leading=14,
    )
    small_style = ParagraphStyle(
        "Small",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.grey,
    )

    elements = []

    # ------------------------------------------------------------------
    # Page de garde
    # ------------------------------------------------------------------
    elements.append(Spacer(1, 4 * cm))
    elements.append(Paragraph(
        "Compression d'Images &amp; Robustesse des VLMs",
        title_style,
    ))
    elements.append(Spacer(1, 1 * cm))
    elements.append(Paragraph(
        "Rapport de Synthèse des Résultats",
        ParagraphStyle("Subtitle", parent=styles["Heading2"],
                       alignment=TA_CENTER, textColor=colors.grey),
    ))
    elements.append(Spacer(1, 2 * cm))

    filter_text = "Filtres : "
    if vlm_name:
        filter_text += f"VLM = {vlm_name} | "
    if category:
        filter_text += f"Catégorie = {category} | "
    filter_text += f"Généré le {datetime.now():%d/%m/%Y à %H:%M}"
    elements.append(Paragraph(filter_text, ParagraphStyle(
        "FilterInfo", parent=body_style, alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"),
    )))

    elements.append(PageBreak())

    # ------------------------------------------------------------------
    # Section 1 : Statistiques globales
    # ------------------------------------------------------------------
    elements.append(Paragraph("1. Statistiques Globales", h1_style))
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#e0e0e0")))

    stats_data = [
        ["Métrique", "Valeur"],
        ["Images évaluées", str(total_images)],
        ["Prédictions totales", str(total_preds)],
        ["VLMs", ", ".join(vlm_names)],
        ["Catégories", ", ".join(categories)],
        ["Conditions JPEG", "QF " + ", ".join(
            str(int(x)) for x in sorted(df[df["compression_type"] == "jpeg"]["compression_level"].dropna().unique(), reverse=True)
        ) if len(df[df["compression_type"] == "jpeg"]) > 0 else "—"],
        ["Conditions Neural", "q" + ", q".join(
            str(int(x)) for x in sorted(df[df["compression_type"] == "neural"]["compression_level"].dropna().unique())
        ) if len(df[df["compression_type"] == "neural"]) > 0 else "—"],
    ]
    elements.append(_make_table(stats_data))
    elements.append(Spacer(1, 0.5 * cm))

    # ------------------------------------------------------------------
    # Section 2 : Baseline
    # ------------------------------------------------------------------
    elements.append(Paragraph("2. Performance Baseline (Images Originales)", h1_style))
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#e0e0e0")))

    baseline_df = df[df["compression_type"] == "baseline"]
    if not baseline_df.empty:
        elements.append(Paragraph(
            "Performance de transcription sur les images PNG originales, sans compression.",
            body_style,
        ))

        # Par VLM
        baseline_agg = (
            baseline_df.groupby("vlm_name")
            .agg(n=("cer", "size"), cer=("cer", "mean"),
                 wer=("wer", "mean"), bleu=("bleu", "mean"))
            .reset_index()
        )
        data = [["VLM", "N images", "CER moy", "WER moy", "BLEU moy"]]
        for _, r in baseline_agg.iterrows():
            data.append([
                r["vlm_name"], str(r["n"]),
                f"{r['cer']:.4f}", f"{r['wer']:.4f}", f"{r['bleu']:.4f}",
            ])
        elements.append(_make_table(data))
        elements.append(Spacer(1, 0.3 * cm))

        # Par VLM × catégorie
        if categories:
            elements.append(Paragraph("2.1 Baseline par catégorie", h2_style))
            bl_cat = (
                baseline_df[baseline_df["category"] != "Unknown"]
                .groupby(["vlm_name", "category"])
                .agg(n=("cer", "size"), cer=("cer", "mean"),
                     wer=("wer", "mean"), bleu=("bleu", "mean"))
                .reset_index()
            )
            data = [["VLM", "Catégorie", "N", "CER", "WER", "BLEU"]]
            for _, r in bl_cat.iterrows():
                data.append([
                    r["vlm_name"], r["category"], str(r["n"]),
                    f"{r['cer']:.4f}", f"{r['wer']:.4f}", f"{r['bleu']:.4f}",
                ])
            elements.append(_make_table(data))
    else:
        elements.append(Paragraph("Aucune donnée baseline disponible.", body_style))

    elements.append(Spacer(1, 0.5 * cm))

    # ------------------------------------------------------------------
    # Section 3 : Impact JPEG
    # ------------------------------------------------------------------
    elements.append(Paragraph("3. Impact de la Compression JPEG", h1_style))
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#e0e0e0")))

    jpeg_df = df[df["compression_type"] == "jpeg"]
    if not jpeg_df.empty:
        jpeg_agg = (
            jpeg_df.groupby(["vlm_name", "compression_level", "condition_label"])
            .agg(n=("cer", "size"), cer=("cer", "mean"),
                 wer=("wer", "mean"), bleu=("bleu", "mean"),
                 bpp=("bitrate_bpp", "mean"))
            .reset_index()
            .sort_values(["vlm_name", "compression_level"], ascending=[True, False])
        )
        data = [["VLM", "Condition", "N", "bpp moy", "CER", "WER", "BLEU"]]
        for _, r in jpeg_agg.iterrows():
            data.append([
                r["vlm_name"], r["condition_label"], str(r["n"]),
                f"{r['bpp']:.3f}" if pd.notna(r["bpp"]) else "—",
                f"{r['cer']:.4f}", f"{r['wer']:.4f}", f"{r['bleu']:.4f}",
            ])
        elements.append(_make_table(data))
    else:
        elements.append(Paragraph("Aucune donnée JPEG disponible.", body_style))

    elements.append(Spacer(1, 0.5 * cm))

    # ------------------------------------------------------------------
    # Section 4 : Impact Neural
    # ------------------------------------------------------------------
    elements.append(Paragraph("4. Impact de la Compression Neuronale", h1_style))
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#e0e0e0")))

    neural_df = df[df["compression_type"] == "neural"]
    if not neural_df.empty:
        neural_agg = (
            neural_df.groupby(["vlm_name", "compression_level", "condition_label"])
            .agg(n=("cer", "size"), cer=("cer", "mean"),
                 wer=("wer", "mean"), bleu=("bleu", "mean"),
                 bpp=("bitrate_bpp", "mean"))
            .reset_index()
            .sort_values(["vlm_name", "compression_level"])
        )
        data = [["VLM", "Condition", "N", "bpp moy", "CER", "WER", "BLEU"]]
        for _, r in neural_agg.iterrows():
            data.append([
                r["vlm_name"], r["condition_label"], str(r["n"]),
                f"{r['bpp']:.3f}" if pd.notna(r["bpp"]) else "—",
                f"{r['cer']:.4f}", f"{r['wer']:.4f}", f"{r['bleu']:.4f}",
            ])
        elements.append(_make_table(data))
    else:
        elements.append(Paragraph("Aucune donnée neuronale disponible.", body_style))

    elements.append(Spacer(1, 0.5 * cm))

    # ------------------------------------------------------------------
    # Section 5 : Comparaison JPEG vs Neural
    # ------------------------------------------------------------------
    elements.append(Paragraph("5. Comparaison JPEG vs Neural à Iso-Bitrate", h1_style))
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#e0e0e0")))

    compressed = df[df["compression_type"].isin(["jpeg", "neural"])].copy()
    if not compressed.empty and "bitrate_bpp" in compressed.columns:
        iso_agg = (
            compressed.groupby(["vlm_name", "compression_type", "condition_label"])
            .agg(n=("cer", "size"), bpp=("bitrate_bpp", "mean"),
                 cer=("cer", "mean"), wer=("wer", "mean"),
                 bleu=("bleu", "mean"), ssim=("ssim", "mean"))
            .reset_index()
            .sort_values(["vlm_name", "bpp"])
        )
        data = [["VLM", "Type", "Condition", "bpp", "CER", "WER", "BLEU", "SSIM"]]
        for _, r in iso_agg.iterrows():
            data.append([
                r["vlm_name"], r["compression_type"], r["condition_label"],
                f"{r['bpp']:.3f}" if pd.notna(r["bpp"]) else "—",
                f"{r['cer']:.4f}", f"{r['wer']:.4f}", f"{r['bleu']:.4f}",
                f"{r['ssim']:.4f}" if pd.notna(r["ssim"]) else "—",
            ])
        elements.append(Paragraph(
            "Comparer les lignes JPEG et Neural au bitrate (bpp) le plus proche "
            "pour évaluer quelle méthode préserve mieux l'information textuelle.",
            body_style,
        ))
        elements.append(_make_table(data))
    else:
        elements.append(Paragraph("Données insuffisantes pour la comparaison.", body_style))

    elements.append(PageBreak())

    # ------------------------------------------------------------------
    # Section 6 : Heatmap numérique
    # ------------------------------------------------------------------
    elements.append(Paragraph("6. Heatmap : Catégorie × Condition", h1_style))
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#e0e0e0")))

    for vlm in vlm_names:
        vlm_df = df[(df["vlm_name"] == vlm) & (df["category"] != "Unknown")]
        if vlm_df.empty:
            continue

        elements.append(Paragraph(f"VLM : {vlm} — BLEU score moyen", h2_style))

        pivot = vlm_df.pivot_table(
            index="category",
            columns="condition_label",
            values="bleu",
            aggfunc="mean",
        )

        # Ordonner les colonnes
        col_order = ["baseline"]
        for qf in [90, 70, 50, 30, 10]:
            label = f"jpeg_QF{qf}"
            if label in pivot.columns:
                col_order.append(label)
        for ql in [1, 3, 6]:
            label = f"neural_q{ql}"
            if label in pivot.columns:
                col_order.append(label)
        col_order = [c for c in col_order if c in pivot.columns]
        pivot = pivot[col_order]

        # Construire le tableau
        header = ["Catégorie"] + col_order
        data = [header]
        for cat_name in sorted(pivot.index):
            row = [cat_name]
            for col in col_order:
                val = pivot.loc[cat_name, col]
                row.append(f"{val:.3f}" if pd.notna(val) else "—")
            data.append(row)

        elements.append(_make_heatmap_table(data))
        elements.append(Spacer(1, 0.5 * cm))

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    elements.append(Spacer(1, 1 * cm))
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#cccccc")))
    elements.append(Paragraph(
        f"Rapport généré automatiquement le {datetime.now():%d/%m/%Y à %H:%M} — "
        f"Plateforme VLM Compression v1.0",
        small_style,
    ))

    # ------------------------------------------------------------------
    # Générer le PDF
    # ------------------------------------------------------------------
    doc.build(elements)

    filename = f"rapport_synthese_{datetime.now():%Y%m%d_%H%M}.pdf"
    return FileResponse(
        tmp_path,
        media_type="application/pdf",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ============================================================================
# Helpers tableaux ReportLab
# ============================================================================

def _make_table(data: list):
    """Crée un tableau ReportLab stylé à partir d'une liste de listes."""
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import Table, TableStyle

    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        # Header
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        # Body
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 1), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
        # Grille
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        # Alternance couleur lignes
        *[
            ("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f5f5f5"))
            for i in range(2, len(data), 2)
        ],
    ]))
    return table


def _make_heatmap_table(data: list):
    """
    Crée un tableau avec coloration des cellules selon la valeur BLEU.
    Plus le BLEU est haut → plus vert. Plus bas → plus rouge.
    """
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import Table, TableStyle

    table = Table(data, repeatRows=1)

    style_commands = [
        # Header
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        # Body
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 1), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
        # Grille
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        # Première colonne (catégories) en gras
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("ALIGN", (0, 1), (0, -1), "LEFT"),
    ]

    # Coloration des cellules de données
    for row_idx in range(1, len(data)):
        for col_idx in range(1, len(data[row_idx])):
            cell_val = data[row_idx][col_idx]
            if cell_val == "—":
                continue
            try:
                val = float(cell_val)
                # BLEU : 0 → rouge, 0.5 → jaune, 1.0 → vert
                if val >= 0.7:
                    bg = colors.HexColor("#c8e6c9")  # vert clair
                elif val >= 0.5:
                    bg = colors.HexColor("#fff9c4")  # jaune clair
                elif val >= 0.3:
                    bg = colors.HexColor("#ffe0b2")  # orange clair
                else:
                    bg = colors.HexColor("#ffcdd2")  # rouge clair
                style_commands.append(
                    ("BACKGROUND", (col_idx, row_idx), (col_idx, row_idx), bg)
                )
            except ValueError:
                pass

    table.setStyle(TableStyle(style_commands))
    return table
