#!/usr/bin/env python3
"""
=============================================================================
compute_metrics.py — Prompt 4.1
=============================================================================
Calcul des métriques de transcription :
  - Lit les paires (texte prédit, ground-truth) depuis PostgreSQL
    en joignant les tables `predictions` et `ground_truth`
  - Calcule CER, WER, BLEU pour chaque prédiction
  - Sauvegarde les scores dans la table `metrics`
  - Génère un rapport de synthèse CSV avec agrégations par :
    VLM × compression_type × compression_level × catégorie documentaire

Usage :
    conda activate metrics
    python metrics/compute_metrics.py \
        --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression" \
        --gt-csv data/metadata/ground_truth.csv \
        --output-dir metrics/reports \
        --verbose

    # Recalculer uniquement pour un VLM
    python metrics/compute_metrics.py \
        --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression" \
        --gt-csv data/metadata/ground_truth.csv \
        --filter-vlm qwen2-vl

    # Forcer le recalcul (écrase les métriques existantes)
    python metrics/compute_metrics.py \
        --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression" \
        --gt-csv data/metadata/ground_truth.csv \
        --force

Dépendances : jiwer, sacrebleu, nltk, pandas, numpy, psycopg2, sqlalchemy, tqdm
=============================================================================
"""

import argparse
import logging
import os
import sys
import time
import unicodedata
import re
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================================
# Logging
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# SQLAlchemy — Tables
# ============================================================================
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    Text,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

Base = declarative_base()


class ImageRecord(Base):
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


class CompressionRecord(Base):
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
        Index("ix_comp_type_level", "compression_type", "compression_level"),
    )


class PredictionRecord(Base):
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
        Index("ix_pred_vlm_comp", "vlm_name", "compression_type", "compression_level"),
    )


class MetricRecord(Base):
    """Table des métriques calculées."""
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False, index=True)
    image_id = Column(Integer, ForeignKey("images.image_id"), nullable=False, index=True)
    vlm_name = Column(String(64), nullable=False, index=True)
    compression_type = Column(String(32), nullable=False)
    compression_level = Column(Integer, nullable=True)
    category = Column(String(128))
    cer = Column(Float)
    wer = Column(Float)
    bleu = Column(Float)
    gt_length = Column(Integer)          # Nombre de caractères du GT
    pred_length = Column(Integer)        # Nombre de caractères de la prédiction
    length_ratio = Column(Float)         # pred_length / gt_length
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("prediction_id", name="uq_metric_prediction"),
        Index("ix_metric_vlm_comp", "vlm_name", "compression_type", "compression_level"),
        Index("ix_metric_category", "category"),
    )


# ============================================================================
# Normalisation de texte
# ============================================================================

def normalize_text(text: str) -> str:
    """
    Normalise un texte avant le calcul des métriques.

    Opérations :
    - Normalisation Unicode (NFC)
    - Conversion en minuscules
    - Remplacement des sauts de ligne et tabulations par des espaces
    - Suppression des espaces multiples
    - Strip des espaces en début/fin

    La normalisation est identique pour GT et prédiction → comparaison équitable.
    """
    if not text:
        return ""

    # Unicode NFC
    text = unicodedata.normalize("NFC", text)

    # Minuscules
    text = text.lower()

    # Remplacer sauts de ligne et tabulations par espace
    text = re.sub(r"[\n\r\t]+", " ", text)

    # Supprimer les espaces multiples
    text = re.sub(r"\s+", " ", text)

    # Strip
    text = text.strip()

    return text


# ============================================================================
# Calcul des métriques
# ============================================================================

def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Character Error Rate (CER) via jiwer.

    CER = (S + D + I) / N
    où S = substitutions, D = suppressions, I = insertions,
    N = nombre de caractères dans la référence.

    Returns
    -------
    float : CER entre 0.0 et 1.0+ (peut dépasser 1.0 si plus d'erreurs que de chars).
    """
    import jiwer

    if not reference:
        return 1.0 if hypothesis else 0.0

    # jiwer.cer attend des strings
    cer_val = jiwer.cer(reference, hypothesis)
    return round(cer_val, 6)


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Word Error Rate (WER) via jiwer.

    WER = (S + D + I) / N
    au niveau des mots.

    Returns
    -------
    float : WER entre 0.0 et 1.0+.
    """
    import jiwer

    if not reference:
        return 1.0 if hypothesis else 0.0

    # jiwer gère la tokenisation par espaces
    wer_val = jiwer.wer(reference, hypothesis)
    return round(wer_val, 6)


def compute_bleu(reference: str, hypothesis: str) -> float:
    """
    BLEU score via sacrebleu.

    Utilise sacrebleu pour un calcul standardisé et reproductible.
    BLEU mesure la précision des n-grams du texte prédit par rapport
    au ground-truth. Score entre 0 et 100 (converti en 0-1 ici).

    Returns
    -------
    float : BLEU score entre 0.0 et 1.0.
    """
    import sacrebleu

    if not reference or not hypothesis:
        return 0.0

    # sacrebleu.sentence_bleu attend : hypothesis (str), references (list of str)
    bleu = sacrebleu.sentence_bleu(hypothesis, [reference])
    return round(bleu.score / 100.0, 6)  # Normaliser 0-100 → 0-1


def compute_all_metrics(gt_text: str, pred_text: str) -> dict:
    """
    Calcule CER, WER et BLEU entre le ground-truth et la prédiction.

    Les deux textes sont normalisés avant le calcul.

    Parameters
    ----------
    gt_text : str
        Texte ground-truth.
    pred_text : str
        Texte prédit par le VLM.

    Returns
    -------
    dict : {"cer": float, "wer": float, "bleu": float,
            "gt_length": int, "pred_length": int, "length_ratio": float}
    """
    gt_norm = normalize_text(gt_text or "")
    pred_norm = normalize_text(pred_text or "")

    gt_len = len(gt_norm)
    pred_len = len(pred_norm)
    length_ratio = pred_len / gt_len if gt_len > 0 else 0.0

    return {
        "cer": compute_cer(gt_norm, pred_norm),
        "wer": compute_wer(gt_norm, pred_norm),
        "bleu": compute_bleu(gt_norm, pred_norm),
        "gt_length": gt_len,
        "pred_length": pred_len,
        "length_ratio": round(length_ratio, 4),
    }


# ============================================================================
# Chargement du ground-truth
# ============================================================================

def load_ground_truth(gt_csv_path: str) -> dict:
    """
    Charge le CSV ground-truth et construit un index image_id → gt_text.

    Returns
    -------
    dict : {image_id (int) → gt_text (str)}
    """
    logger.info(f"Chargement du ground-truth : {gt_csv_path}")

    if not os.path.exists(gt_csv_path):
        logger.error(f"CSV introuvable : {gt_csv_path}")
        sys.exit(1)

    df = pd.read_csv(gt_csv_path)
    logger.info(f"  → {len(df)} entrées")

    # Vérifier les colonnes requises
    required_cols = ["image_id", "gt_text"]
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Colonne manquante dans le CSV : '{col}'")
            sys.exit(1)

    # Construire l'index
    gt_index = {}
    n_empty = 0
    for _, row in df.iterrows():
        img_id = int(row["image_id"])
        text = str(row["gt_text"]) if pd.notna(row["gt_text"]) else ""
        if not text.strip():
            n_empty += 1
        gt_index[img_id] = text

    logger.info(f"  → {len(gt_index)} images indexées, {n_empty} avec texte vide")
    return gt_index


# ============================================================================
# Pipeline principal
# ============================================================================

def run_pipeline(
    db_url: str,
    gt_csv_path: str,
    output_dir: str,
    filter_vlm: str,
    force: bool,
    batch_commit_size: int,
    verbose: bool,
):
    """Pipeline complet de calcul des métriques."""

    # ------------------------------------------------------------------
    # 1. Charger le ground-truth
    # ------------------------------------------------------------------
    gt_index = load_ground_truth(gt_csv_path)

    # ------------------------------------------------------------------
    # 2. Connexion base de données
    # ------------------------------------------------------------------
    logger.info("Connexion à la base de données...")
    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    logger.info("  → Connexion établie, table metrics créée/vérifiée")

    # ------------------------------------------------------------------
    # 3. Récupérer les prédictions à évaluer
    # ------------------------------------------------------------------
    logger.info("\nChargement des prédictions depuis la base...")
    query = session.query(PredictionRecord, ImageRecord).join(
        ImageRecord, PredictionRecord.image_id == ImageRecord.image_id
    )

    if filter_vlm:
        query = query.filter(PredictionRecord.vlm_name == filter_vlm)

    predictions = query.all()
    logger.info(f"  → {len(predictions)} prédictions trouvées")

    if not predictions:
        logger.warning("Aucune prédiction en base. Lancer d'abord les scripts d'inférence.")
        session.close()
        return

    # Résumé
    vlm_counts = {}
    for pred, img in predictions:
        key = pred.vlm_name
        vlm_counts[key] = vlm_counts.get(key, 0) + 1
    for vlm, count in sorted(vlm_counts.items()):
        logger.info(f"    {vlm:20s} : {count} prédictions")

    # ------------------------------------------------------------------
    # 4. Filtrer celles déjà calculées (sauf si --force)
    # ------------------------------------------------------------------
    if not force:
        existing_pred_ids = set(
            row[0] for row in session.query(MetricRecord.prediction_id).all()
        )
        tasks = [
            (pred, img) for pred, img in predictions
            if pred.id not in existing_pred_ids
        ]
        n_skipped = len(predictions) - len(tasks)
        logger.info(f"  → {n_skipped} déjà calculées (skip), {len(tasks)} restantes")
    else:
        tasks = predictions
        # Supprimer les métriques existantes pour recalcul
        if filter_vlm:
            session.query(MetricRecord).filter(
                MetricRecord.vlm_name == filter_vlm
            ).delete()
        else:
            session.query(MetricRecord).delete()
        session.commit()
        logger.info(f"  → Mode --force : métriques existantes supprimées, {len(tasks)} à recalculer")

    if not tasks:
        logger.info("Rien à calculer — toutes les métriques existent déjà.")
        # Aller directement à la génération du rapport
        generate_reports(session, output_dir)
        session.close()
        return

    # ------------------------------------------------------------------
    # 5. Calcul des métriques
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info(f"Calcul des métriques CER / WER / BLEU")
    logger.info(f"{'='*60}\n")

    stats = {
        "total": len(tasks),
        "success": 0,
        "no_gt": 0,
        "errors": 0,
        "cer_sum": 0,
        "wer_sum": 0,
        "bleu_sum": 0,
    }

    t_start = time.time()

    for idx, (pred, img) in enumerate(tqdm(tasks, desc="Calcul métriques", unit="pred")):
        try:
            # Récupérer le ground-truth
            gt_text = gt_index.get(pred.image_id, None)

            if gt_text is None:
                stats["no_gt"] += 1
                if verbose:
                    logger.warning(f"  Pas de GT pour image_id={pred.image_id}")
                continue

            # Calculer les métriques
            scores = compute_all_metrics(gt_text, pred.predicted_text)

            # Insérer en base
            metric_record = MetricRecord(
                prediction_id=pred.id,
                image_id=pred.image_id,
                vlm_name=pred.vlm_name,
                compression_type=pred.compression_type,
                compression_level=pred.compression_level,
                category=img.category,
                cer=scores["cer"],
                wer=scores["wer"],
                bleu=scores["bleu"],
                gt_length=scores["gt_length"],
                pred_length=scores["pred_length"],
                length_ratio=scores["length_ratio"],
            )
            session.add(metric_record)

            stats["success"] += 1
            stats["cer_sum"] += scores["cer"]
            stats["wer_sum"] += scores["wer"]
            stats["bleu_sum"] += scores["bleu"]

            if verbose:
                logger.info(
                    f"  [{pred.vlm_name}|{pred.compression_type}"
                    f"{'/' + str(pred.compression_level) if pred.compression_level else ''}] "
                    f"img={pred.image_id} → "
                    f"CER={scores['cer']:.4f} WER={scores['wer']:.4f} BLEU={scores['bleu']:.4f}"
                )

        except Exception as e:
            stats["errors"] += 1
            logger.error(f"  Erreur sur prediction_id={pred.id} : {e}")
            continue

        # Commit intermédiaire
        if (idx + 1) % batch_commit_size == 0:
            session.commit()

    # Commit final
    session.commit()

    t_elapsed = time.time() - t_start

    # ------------------------------------------------------------------
    # 6. Rapport console
    # ------------------------------------------------------------------
    n = stats["success"]
    avg_cer = stats["cer_sum"] / n if n > 0 else 0
    avg_wer = stats["wer_sum"] / n if n > 0 else 0
    avg_bleu = stats["bleu_sum"] / n if n > 0 else 0

    print(f"\n{'='*60}")
    print(f"RAPPORT — Calcul des Métriques")
    print(f"{'='*60}")
    print(f"Prédictions traitées : {stats['total']}")
    print(f"Métriques calculées  : {stats['success']}")
    print(f"Sans ground-truth    : {stats['no_gt']}")
    print(f"Erreurs              : {stats['errors']}")
    print(f"Temps                : {t_elapsed:.1f}s")
    print(f"")
    print(f"Moyennes globales :")
    print(f"  CER  : {avg_cer:.4f} ({avg_cer*100:.2f}%)")
    print(f"  WER  : {avg_wer:.4f} ({avg_wer*100:.2f}%)")
    print(f"  BLEU : {avg_bleu:.4f}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 7. Générer les rapports CSV
    # ------------------------------------------------------------------
    generate_reports(session, output_dir)

    session.close()

    print(f"\nProchaine étape : Prompt 5.1 (schéma SQL complet)")


# ============================================================================
# Génération des rapports CSV
# ============================================================================

def generate_reports(session: Session, output_dir: str):
    """
    Génère les rapports CSV de synthèse depuis la table metrics.

    Rapports générés :
    1. metrics_detail.csv       — une ligne par prédiction évaluée
    2. metrics_by_condition.csv — agrégé par VLM × compression_type × level
    3. metrics_by_category.csv  — agrégé par VLM × compression × catégorie
    4. metrics_heatmap.csv      — format pivot pour heatmap (catégorie × condition)
    5. metrics_iso_bitrate.csv  — comparaison JPEG vs Neural à bitrate proche
    """
    os.makedirs(output_dir, exist_ok=True)

    # Charger toutes les métriques
    all_metrics = session.query(MetricRecord).all()

    if not all_metrics:
        logger.warning("Aucune métrique en base — rapports vides.")
        return

    # Convertir en DataFrame
    rows = []
    for m in all_metrics:
        rows.append({
            "prediction_id": m.prediction_id,
            "image_id": m.image_id,
            "vlm_name": m.vlm_name,
            "compression_type": m.compression_type,
            "compression_level": m.compression_level,
            "category": m.category,
            "cer": m.cer,
            "wer": m.wer,
            "bleu": m.bleu,
            "gt_length": m.gt_length,
            "pred_length": m.pred_length,
            "length_ratio": m.length_ratio,
        })

    df = pd.DataFrame(rows)

    # Construire un label de condition lisible
    def make_condition_label(row):
        if row["compression_type"] == "baseline":
            return "baseline"
        elif row["compression_type"] == "jpeg":
            return f"jpeg_QF{row['compression_level']}"
        elif row["compression_type"] == "neural":
            return f"neural_q{row['compression_level']}"
        return f"{row['compression_type']}_{row['compression_level']}"

    df["condition"] = df.apply(make_condition_label, axis=1)

    # ---- Rapport 1 : Détail complet ----
    detail_path = os.path.join(output_dir, "metrics_detail.csv")
    df.to_csv(detail_path, index=False, float_format="%.6f")
    logger.info(f"  → {detail_path} ({len(df)} lignes)")

    # ---- Rapport 2 : Agrégé par VLM × condition ----
    agg_condition = (
        df.groupby(["vlm_name", "compression_type", "compression_level", "condition"])
        .agg(
            count=("cer", "size"),
            cer_mean=("cer", "mean"),
            cer_std=("cer", "std"),
            cer_median=("cer", "median"),
            wer_mean=("wer", "mean"),
            wer_std=("wer", "std"),
            wer_median=("wer", "median"),
            bleu_mean=("bleu", "mean"),
            bleu_std=("bleu", "std"),
            bleu_median=("bleu", "median"),
        )
        .reset_index()
        .sort_values(["vlm_name", "compression_type", "compression_level"])
    )

    condition_path = os.path.join(output_dir, "metrics_by_condition.csv")
    agg_condition.to_csv(condition_path, index=False, float_format="%.6f")
    logger.info(f"  → {condition_path} ({len(agg_condition)} lignes)")

    # Afficher un résumé
    print(f"\n{'─'*80}")
    print(f"RÉSUMÉ PAR CONDITION")
    print(f"{'─'*80}")
    print(f"{'VLM':>12} {'Condition':>16} {'N':>6} {'CER moy':>10} {'WER moy':>10} {'BLEU moy':>10}")
    print(f"{'─'*80}")
    for _, row in agg_condition.iterrows():
        print(
            f"{row['vlm_name']:>12} {row['condition']:>16} {row['count']:>6} "
            f"{row['cer_mean']:>10.4f} {row['wer_mean']:>10.4f} {row['bleu_mean']:>10.4f}"
        )
    print(f"{'─'*80}")

    # ---- Rapport 3 : Agrégé par VLM × condition × catégorie ----
    agg_category = (
        df.groupby(["vlm_name", "condition", "category"])
        .agg(
            count=("cer", "size"),
            cer_mean=("cer", "mean"),
            wer_mean=("wer", "mean"),
            bleu_mean=("bleu", "mean"),
        )
        .reset_index()
        .sort_values(["vlm_name", "condition", "category"])
    )

    category_path = os.path.join(output_dir, "metrics_by_category.csv")
    agg_category.to_csv(category_path, index=False, float_format="%.6f")
    logger.info(f"  → {category_path} ({len(agg_category)} lignes)")

    # ---- Rapport 4 : Heatmap (catégorie × condition) ----
    # Un fichier par VLM, pivot sur BLEU
    for vlm_name in df["vlm_name"].unique():
        vlm_df = df[df["vlm_name"] == vlm_name]

        for metric_name in ["cer", "wer", "bleu"]:
            pivot = vlm_df.pivot_table(
                index="category",
                columns="condition",
                values=metric_name,
                aggfunc="mean",
            )

            # Ordonner les colonnes logiquement
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

            heatmap_path = os.path.join(output_dir, f"heatmap_{vlm_name}_{metric_name}.csv")
            pivot.to_csv(heatmap_path, float_format="%.4f")
            logger.info(f"  → {heatmap_path}")

    # ---- Rapport 5 : Comparaison JPEG vs Neural à iso-bitrate ----
    generate_iso_bitrate_report(session, df, output_dir)

    print(f"\n{'='*60}")
    print(f"Tous les rapports CSV générés dans : {output_dir}")
    print(f"{'='*60}")


def generate_iso_bitrate_report(session: Session, metrics_df: pd.DataFrame, output_dir: str):
    """
    Génère le rapport de comparaison JPEG vs Neural à bitrate comparable.

    Récupère les bitrates réels depuis la table compressions et associe
    chaque paire JPEG/Neural par image pour une comparaison directe.
    """
    # Charger les bitrates depuis la table compressions
    compressions = session.query(CompressionRecord).all()
    bpp_index = {}
    for c in compressions:
        key = (c.image_id, c.compression_type, c.compression_level)
        bpp_index[key] = c.bitrate_bpp

    # Ajouter le bitrate au DataFrame des métriques
    def get_bpp(row):
        if row["compression_type"] == "baseline":
            return None
        key = (row["image_id"], row["compression_type"], row["compression_level"])
        return bpp_index.get(key, None)

    metrics_df = metrics_df.copy()
    metrics_df["bitrate_bpp"] = metrics_df.apply(get_bpp, axis=1)

    # Filtrer les lignes compressées avec bitrate connu
    compressed = metrics_df[metrics_df["bitrate_bpp"].notna()].copy()

    if compressed.empty:
        logger.warning("  Pas de données de bitrate — rapport iso-bitrate ignoré")
        return

    # Agrégation par VLM × type × level avec bitrate moyen
    iso_agg = (
        compressed.groupby(["vlm_name", "compression_type", "compression_level"])
        .agg(
            count=("cer", "size"),
            bpp_mean=("bitrate_bpp", "mean"),
            bpp_std=("bitrate_bpp", "std"),
            cer_mean=("cer", "mean"),
            wer_mean=("wer", "mean"),
            bleu_mean=("bleu", "mean"),
        )
        .reset_index()
        .sort_values(["vlm_name", "bpp_mean"])
    )

    iso_path = os.path.join(output_dir, "metrics_iso_bitrate.csv")
    iso_agg.to_csv(iso_path, index=False, float_format="%.6f")
    logger.info(f"  → {iso_path} ({len(iso_agg)} lignes)")

    # Afficher le résumé comparatif
    print(f"\n{'─'*80}")
    print(f"COMPARAISON ISO-BITRATE : JPEG vs NEURAL")
    print(f"{'─'*80}")
    print(f"{'VLM':>12} {'Type':>8} {'Level':>6} {'bpp moy':>10} {'CER':>8} {'WER':>8} {'BLEU':>8}")
    print(f"{'─'*80}")
    for _, row in iso_agg.iterrows():
        print(
            f"{row['vlm_name']:>12} {row['compression_type']:>8} "
            f"{row['compression_level']:>6} {row['bpp_mean']:>10.4f} "
            f"{row['cer_mean']:>8.4f} {row['wer_mean']:>8.4f} {row['bleu_mean']:>8.4f}"
        )
    print(f"{'─'*80}")
    print(f"→ Comparer les lignes JPEG et Neural à bpp_mean similaire")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Calcul des métriques CER/WER/BLEU et génération des rapports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  # Calcul complet
  python metrics/compute_metrics.py \\
      --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression" \\
      --gt-csv data/metadata/ground_truth.csv

  # Uniquement pour un VLM
  python metrics/compute_metrics.py \\
      --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression" \\
      --gt-csv data/metadata/ground_truth.csv \\
      --filter-vlm internvl2

  # Forcer le recalcul
  python metrics/compute_metrics.py \\
      --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression" \\
      --gt-csv data/metadata/ground_truth.csv \\
      --force --verbose

  # Sortie dans un dossier spécifique
  python metrics/compute_metrics.py \\
      --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression" \\
      --gt-csv data/metadata/ground_truth.csv \\
      --output-dir metrics/reports/run_20250315
        """,
    )

    parser.add_argument(
        "--db-url",
        required=True,
        help="URL PostgreSQL",
    )
    parser.add_argument(
        "--gt-csv",
        required=True,
        help="CSV ground-truth issu de prepare_ground_truth.py",
    )
    parser.add_argument(
        "--output-dir",
        default="metrics/reports",
        help="Dossier de sortie des rapports CSV (défaut: metrics/reports)",
    )
    parser.add_argument(
        "--filter-vlm",
        default=None,
        help="Ne calculer que pour ce VLM (ex: qwen2-vl, internvl2)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recalculer toutes les métriques (écrase les existantes)",
    )
    parser.add_argument(
        "--batch-commit",
        type=int,
        default=100,
        help="Nombre de métriques entre chaque commit SQL (défaut: 100)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Afficher les détails par prédiction",
    )

    args = parser.parse_args()

    run_pipeline(
        db_url=args.db_url,
        gt_csv_path=args.gt_csv,
        output_dir=args.output_dir,
        filter_vlm=args.filter_vlm,
        force=args.force,
        batch_commit_size=args.batch_commit,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
