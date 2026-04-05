#!/usr/bin/env python3
"""
=============================================================================
compress_jpeg.py — Prompt 2.1
=============================================================================
Pipeline de compression JPEG batch :
  - Lit un dossier d'images PNG (DocLayNet)
  - Génère 5 niveaux de qualité JPEG (QF = 90, 70, 50, 30, 10)
  - Calcule les métadonnées : taille Ko, SSIM, bitrate (bpp)
  - Sauvegarde les images compressées dans data/compressed/jpeg/
  - Insère les métadonnées dans la table `compressions` de PostgreSQL

Usage :
    conda activate compression
    python compression/jpeg_pipeline/compress_jpeg.py \
        --input-dir data/raw \
        --output-dir data/compressed/jpeg \
        --gt-csv data/metadata/ground_truth.csv \
        --quality-factors 90 70 50 30 10 \
        --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression" \
        --verbose

Dépendances : Pillow, opencv-python, piq, torch, psycopg2, sqlalchemy, pandas, tqdm
=============================================================================
"""

import argparse
import io
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
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
# SQLAlchemy — Modèle de table `images` et `compressions`
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
from datetime import datetime

Base = declarative_base()


class ImageRecord(Base):
    """Table des images originales."""
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
    """Table des images compressées."""
    __tablename__ = "compressions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey("images.image_id"), nullable=False, index=True)
    compression_type = Column(String(32), nullable=False)       # "jpeg" ou "neural"
    compression_level = Column(Integer)                          # QF pour JPEG
    quality_label = Column(String(32))                           # "QF90", "QF70", etc.
    bitrate_bpp = Column(Float)                                  # bits per pixel
    file_size_kb = Column(Float)
    compression_ratio = Column(Float)                            # original / compressé
    ssim = Column(Float)
    compressed_path = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("image_id", "compression_type", "compression_level",
                         name="uq_image_compression"),
        Index("ix_comp_type_level", "compression_type", "compression_level"),
    )


# ============================================================================
# Fonctions utilitaires
# ============================================================================

def compute_ssim_score(original_path: str, compressed_path: str) -> float:
    """
    Calcule le SSIM entre l'image originale et compressée.
    Utilise piq pour un calcul GPU-accéléré si disponible, sinon CPU.

    Returns
    -------
    float : Score SSIM entre 0 et 1.
    """
    import piq

    # Charger les images en tenseurs normalisés [0, 1]
    orig = cv2.imread(original_path, cv2.IMREAD_COLOR)
    comp = cv2.imread(compressed_path, cv2.IMREAD_COLOR)

    if orig is None or comp is None:
        logger.warning(f"Impossible de lire les images pour SSIM")
        return -1.0

    # Convertir BGR → RGB et normaliser
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    comp_rgb = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Convertir en tenseur PyTorch [B, C, H, W]
    orig_t = torch.from_numpy(orig_rgb).permute(2, 0, 1).unsqueeze(0)
    comp_t = torch.from_numpy(comp_rgb).permute(2, 0, 1).unsqueeze(0)

    # S'assurer que les dimensions correspondent
    if orig_t.shape != comp_t.shape:
        logger.warning(f"Dimensions différentes : {orig_t.shape} vs {comp_t.shape}")
        return -1.0

    # Calcul SSIM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    orig_t = orig_t.to(device)
    comp_t = comp_t.to(device)

    with torch.no_grad():
        ssim_val = piq.ssim(orig_t, comp_t, data_range=1.0).item()

    return round(ssim_val, 6)


def compute_bitrate(file_size_bytes: int, width: int, height: int) -> float:
    """
    Calcule le bitrate en bits per pixel (bpp).

    bpp = (taille_fichier_en_bits) / (largeur × hauteur)
    """
    if width <= 0 or height <= 0:
        return 0.0
    total_pixels = width * height
    total_bits = file_size_bytes * 8
    return round(total_bits / total_pixels, 4)


def compress_single_image_jpeg(
    input_path: str,
    output_path: str,
    quality: int,
) -> dict:
    """
    Compresse une image PNG en JPEG à un niveau de qualité donné.

    Parameters
    ----------
    input_path : str
        Chemin de l'image PNG originale.
    output_path : str
        Chemin de sauvegarde du JPEG.
    quality : int
        Facteur de qualité JPEG (1-100).

    Returns
    -------
    dict : Métadonnées de compression.
    """
    # Ouvrir avec Pillow
    img = Image.open(input_path)

    # Convertir en RGB si nécessaire (PNG peut être RGBA)
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")

    width, height = img.size

    # Taille originale
    original_size_bytes = os.path.getsize(input_path)

    # Compresser en JPEG
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, format="JPEG", quality=quality, optimize=True)

    # Taille compressée
    compressed_size_bytes = os.path.getsize(output_path)

    # Bitrate
    bpp = compute_bitrate(compressed_size_bytes, width, height)

    # Ratio de compression
    ratio = original_size_bytes / compressed_size_bytes if compressed_size_bytes > 0 else 0

    return {
        "width": width,
        "height": height,
        "original_size_kb": round(original_size_bytes / 1024, 2),
        "compressed_size_kb": round(compressed_size_bytes / 1024, 2),
        "bitrate_bpp": bpp,
        "compression_ratio": round(ratio, 2),
    }


# ============================================================================
# Pipeline principal
# ============================================================================

def ensure_image_in_db(session: Session, image_id: int, filename: str,
                       category: str, split: str, width: int, height: int,
                       original_size_kb: float, original_path: str) -> None:
    """Insère l'image dans la table `images` si elle n'existe pas déjà."""
    existing = session.query(ImageRecord).filter_by(image_id=image_id).first()
    if not existing:
        record = ImageRecord(
            image_id=image_id,
            filename=filename,
            category=category,
            split=split,
            width=width,
            height=height,
            original_size_kb=original_size_kb,
            original_path=original_path,
        )
        session.add(record)


def run_pipeline(
    input_dir: str,
    output_dir: str,
    gt_csv_path: str,
    quality_factors: list,
    db_url: str,
    compute_ssim: bool = True,
    verbose: bool = False,
    batch_commit_size: int = 50,
):
    """
    Pipeline complet de compression JPEG batch.

    Parameters
    ----------
    input_dir : str
        Dossier contenant les images PNG.
    output_dir : str
        Dossier racine de sortie pour les JPEG.
    gt_csv_path : str
        CSV du ground-truth (issu de prepare_ground_truth.py).
    quality_factors : list of int
        Niveaux de qualité JPEG à générer.
    db_url : str
        URL de connexion PostgreSQL.
    compute_ssim : bool
        Calculer le SSIM pour chaque image compressée.
    verbose : bool
        Afficher les détails.
    batch_commit_size : int
        Nombre d'images avant commit intermédiaire.
    """

    # ------------------------------------------------------------------
    # 1. Charger le CSV ground-truth pour le mapping image_id ↔ filename
    # ------------------------------------------------------------------
    logger.info(f"Chargement du CSV ground-truth : {gt_csv_path}")
    if not os.path.exists(gt_csv_path):
        logger.error(f"CSV introuvable : {gt_csv_path}")
        sys.exit(1)

    gt_df = pd.read_csv(gt_csv_path)
    logger.info(f"  → {len(gt_df)} images dans le CSV")

    # Index par filename pour lookup rapide
    gt_index = gt_df.set_index("filename").to_dict("index")

    # ------------------------------------------------------------------
    # 2. Lister les images PNG dans le dossier d'entrée
    # ------------------------------------------------------------------
    logger.info(f"Scan du dossier d'entrée : {input_dir}")
    png_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".PNG"))
    ])
    logger.info(f"  → {len(png_files)} images PNG trouvées")

    if not png_files:
        logger.error("Aucune image PNG trouvée. Vérifier le chemin.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Connexion à la base de données
    # ------------------------------------------------------------------
    logger.info(f"Connexion à la base de données...")
    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)  # Crée les tables si elles n'existent pas
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    logger.info(f"  → Connexion établie, tables créées/vérifiées")

    # ------------------------------------------------------------------
    # 4. Créer les sous-dossiers de sortie par QF
    # ------------------------------------------------------------------
    for qf in quality_factors:
        qf_dir = os.path.join(output_dir, f"QF{qf}")
        os.makedirs(qf_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 5. Compression batch
    # ------------------------------------------------------------------
    logger.info(f"\nDémarrage de la compression JPEG batch")
    logger.info(f"  Niveaux de qualité : {quality_factors}")
    logger.info(f"  SSIM : {'Oui' if compute_ssim else 'Non'}")
    logger.info(f"  Sortie : {output_dir}")
    logger.info(f"{'='*60}\n")

    stats = {
        "total_images": len(png_files),
        "total_compressions": 0,
        "skipped": 0,
        "errors": 0,
        "by_qf": {qf: {"count": 0, "total_size_kb": 0, "ssim_sum": 0} for qf in quality_factors},
    }

    t_start = time.time()

    for idx, png_filename in enumerate(tqdm(png_files, desc="Compression JPEG")):
        input_path = os.path.join(input_dir, png_filename)

        # Lookup dans le CSV GT
        gt_info = gt_index.get(png_filename, {})
        image_id = gt_info.get("image_id", idx)
        doc_category = gt_info.get("doc_category", "Unknown")
        split = gt_info.get("split", "unknown")

        for qf in quality_factors:
            # Chemin de sortie : data/compressed/jpeg/QF90/image.jpg
            out_filename = Path(png_filename).stem + ".jpg"
            out_path = os.path.join(output_dir, f"QF{qf}", out_filename)

            # Vérifier si déjà traité (reprise après interruption)
            existing = (
                session.query(CompressionRecord)
                .filter_by(
                    image_id=int(image_id),
                    compression_type="jpeg",
                    compression_level=qf,
                )
                .first()
            )
            if existing:
                stats["skipped"] += 1
                continue

            try:
                # Compresser
                meta = compress_single_image_jpeg(input_path, out_path, quality=qf)

                # SSIM
                ssim_val = None
                if compute_ssim:
                    ssim_val = compute_ssim_score(input_path, out_path)

                # Insérer l'image originale dans la table `images` (si pas déjà fait)
                ensure_image_in_db(
                    session=session,
                    image_id=int(image_id),
                    filename=png_filename,
                    category=doc_category,
                    split=split,
                    width=meta["width"],
                    height=meta["height"],
                    original_size_kb=meta["original_size_kb"],
                    original_path=input_path,
                )

                # Insérer le résultat de compression
                comp_record = CompressionRecord(
                    image_id=int(image_id),
                    compression_type="jpeg",
                    compression_level=qf,
                    quality_label=f"QF{qf}",
                    bitrate_bpp=meta["bitrate_bpp"],
                    file_size_kb=meta["compressed_size_kb"],
                    compression_ratio=meta["compression_ratio"],
                    ssim=ssim_val,
                    compressed_path=out_path,
                )
                session.add(comp_record)

                stats["total_compressions"] += 1
                stats["by_qf"][qf]["count"] += 1
                stats["by_qf"][qf]["total_size_kb"] += meta["compressed_size_kb"]
                if ssim_val and ssim_val > 0:
                    stats["by_qf"][qf]["ssim_sum"] += ssim_val

                if verbose:
                    logger.info(
                        f"  {png_filename} → QF{qf} : "
                        f"{meta['compressed_size_kb']:.1f} Ko, "
                        f"{meta['bitrate_bpp']:.3f} bpp, "
                        f"SSIM={ssim_val:.4f}" if ssim_val else ""
                    )

            except Exception as e:
                stats["errors"] += 1
                logger.error(f"Erreur sur {png_filename} (QF{qf}) : {e}")
                continue

        # Commit intermédiaire tous les N images
        if (idx + 1) % batch_commit_size == 0:
            session.commit()
            if verbose:
                logger.info(f"  → Commit intermédiaire ({idx + 1}/{len(png_files)})")

    # Commit final
    session.commit()
    session.close()

    t_elapsed = time.time() - t_start

    # ------------------------------------------------------------------
    # 6. Rapport de synthèse
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"RAPPORT — Compression JPEG Batch")
    print(f"{'='*60}")
    print(f"Images traitées     : {stats['total_images']}")
    print(f"Compressions créées : {stats['total_compressions']}")
    print(f"Skipped (déjà fait) : {stats['skipped']}")
    print(f"Erreurs             : {stats['errors']}")
    print(f"Temps total         : {t_elapsed:.1f}s ({t_elapsed/60:.1f} min)")
    print()
    print(f"{'QF':>6} {'Count':>8} {'Taille moy (Ko)':>16} {'SSIM moy':>10}")
    print(f"{'-'*44}")
    for qf in quality_factors:
        qf_stats = stats["by_qf"][qf]
        n = qf_stats["count"]
        if n > 0:
            avg_size = qf_stats["total_size_kb"] / n
            avg_ssim = qf_stats["ssim_sum"] / n if compute_ssim else 0
            print(f"{qf:>6} {n:>8} {avg_size:>16.1f} {avg_ssim:>10.4f}")
    print(f"{'='*60}")
    print(f"Images compressées dans : {output_dir}")
    print(f"Métadonnées en base PostgreSQL")
    print(f"{'='*60}")
    print(f"\nProchaine étape : Prompt 2.2 (compression neuronale CompressAI)")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de compression JPEG batch avec métadonnées en PostgreSQL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  # Compression complète
  python compression/jpeg_pipeline/compress_jpeg.py \\
      --input-dir data/raw \\
      --output-dir data/compressed/jpeg \\
      --gt-csv data/metadata/ground_truth.csv \\
      --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression"

  # Test rapide sans SSIM
  python compression/jpeg_pipeline/compress_jpeg.py \\
      --input-dir data/raw \\
      --output-dir data/compressed/jpeg \\
      --gt-csv data/metadata/ground_truth.csv \\
      --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression" \\
      --quality-factors 90 50 10 \\
      --no-ssim

  # Avec qualités personnalisées
  python compression/jpeg_pipeline/compress_jpeg.py \\
      --input-dir data/raw \\
      --output-dir data/compressed/jpeg \\
      --gt-csv data/metadata/ground_truth.csv \\
      --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression" \\
      --quality-factors 95 80 60 40 20 5
        """,
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Dossier contenant les images PNG originales",
    )
    parser.add_argument(
        "--output-dir",
        default="data/compressed/jpeg",
        help="Dossier racine de sortie (défaut: data/compressed/jpeg)",
    )
    parser.add_argument(
        "--gt-csv",
        required=True,
        help="CSV ground-truth issu de prepare_ground_truth.py",
    )
    parser.add_argument(
        "--quality-factors",
        nargs="+",
        type=int,
        default=[90, 70, 50, 30, 10],
        help="Niveaux de qualité JPEG (défaut: 90 70 50 30 10)",
    )
    parser.add_argument(
        "--db-url",
        required=True,
        help="URL PostgreSQL (ex: postgresql://user:pass@localhost:5432/db)",
    )
    parser.add_argument(
        "--no-ssim",
        action="store_true",
        help="Désactiver le calcul SSIM (plus rapide)",
    )
    parser.add_argument(
        "--batch-commit",
        type=int,
        default=50,
        help="Nombre d'images entre chaque commit SQL (défaut: 50)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Afficher les détails par image",
    )

    args = parser.parse_args()

    # Validation
    if not os.path.isdir(args.input_dir):
        logger.error(f"Dossier introuvable : {args.input_dir}")
        sys.exit(1)

    for qf in args.quality_factors:
        if not 1 <= qf <= 100:
            logger.error(f"Quality factor invalide : {qf} (doit être entre 1 et 100)")
            sys.exit(1)

    run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        gt_csv_path=args.gt_csv,
        quality_factors=sorted(args.quality_factors, reverse=True),
        db_url=args.db_url,
        compute_ssim=not args.no_ssim,
        verbose=args.verbose,
        batch_commit_size=args.batch_commit,
    )


if __name__ == "__main__":
    main()
