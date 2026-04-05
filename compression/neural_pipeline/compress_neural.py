#!/usr/bin/env python3
"""
=============================================================================
compress_neural.py — Prompt 2.2
=============================================================================
Pipeline de compression neuronale batch avec CompressAI :
  - Lit un dossier d'images PNG (DocLayNet)
  - Compresse avec le modèle Cheng2020 (ou ELIC) à 3 niveaux de qualité
    correspondant à ~0.1, ~0.25, ~0.5 bpp
  - Calcule les métadonnées : taille Ko, bitrate réel (bpp), SSIM
  - Sauvegarde les images reconstruites dans data/compressed/neural/
  - Insère les métadonnées dans la table `compressions` de PostgreSQL
    (même schéma que compress_jpeg.py)

Usage :
    conda activate compression
    python compression/neural_pipeline/compress_neural.py \
        --input-dir data/raw \
        --output-dir data/compressed/neural \
        --gt-csv data/metadata/ground_truth.csv \
        --model cheng2020-anchor \
        --quality-levels 1 3 6 \
        --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression" \
        --verbose

Modèles CompressAI disponibles :
    - cheng2020-anchor  (par défaut, bon compromis qualité/vitesse)
    - cheng2020-attn    (meilleur qualité, plus lent)
    - mbt2018           (plus léger)

Niveaux de qualité CompressAI → bitrate approximatif :
    Niveau 1 → ~0.05–0.12 bpp (compression agressive)
    Niveau 3 → ~0.15–0.30 bpp (compression modérée)
    Niveau 6 → ~0.35–0.60 bpp (haute qualité)
    (varie selon le contenu de l'image)

Dépendances : compressai, torch, torchvision, Pillow, opencv-python, piq,
              psycopg2, sqlalchemy, pandas, tqdm
=============================================================================
"""

import argparse
import io
import logging
import math
import os
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
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
# SQLAlchemy — Réutilise le même schéma que compress_jpeg.py
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


# ============================================================================
# Mapping qualité CompressAI → label lisible
# ============================================================================
QUALITY_LABELS = {
    1: "neural_q1",    # ~0.1 bpp — compression agressive
    2: "neural_q2",
    3: "neural_q3",    # ~0.25 bpp — compression modérée
    4: "neural_q4",
    5: "neural_q5",
    6: "neural_q6",    # ~0.5 bpp — haute qualité
}

# Mapping modèle → nombre max de niveaux de qualité
MODEL_MAX_QUALITY = {
    "cheng2020-anchor": 6,
    "cheng2020-attn": 6,
    "mbt2018": 8,
    "mbt2018-mean": 8,
}


# ============================================================================
# Chargement du modèle CompressAI
# ============================================================================

def load_compressai_model(model_name: str, quality: int, device: str = "cuda"):
    """
    Charge un modèle pré-entraîné CompressAI.

    Parameters
    ----------
    model_name : str
        Nom du modèle (ex: "cheng2020-anchor").
    quality : int
        Niveau de qualité (1–6 pour Cheng2020, 1–8 pour mbt2018).
    device : str
        "cuda" ou "cpu".

    Returns
    -------
    nn.Module : Modèle CompressAI prêt pour l'inférence.
    """
    import compressai
    from compressai.zoo import models as compressai_models

    # Mapping nom → fonction de chargement
    model_registry = {
        "cheng2020-anchor": compressai_models["cheng2020-anchor"],
        "cheng2020-attn": compressai_models["cheng2020-attn"],
        "mbt2018": compressai_models["mbt2018"],
        "mbt2018-mean": compressai_models["mbt2018-mean"],
    }

    if model_name not in model_registry:
        raise ValueError(
            f"Modèle inconnu : {model_name}. "
            f"Disponibles : {list(model_registry.keys())}"
        )

    max_q = MODEL_MAX_QUALITY.get(model_name, 6)
    if not 1 <= quality <= max_q:
        raise ValueError(
            f"Quality {quality} hors limites pour {model_name} (1–{max_q})"
        )

    logger.info(f"  Chargement de {model_name} (quality={quality})...")
    model = model_registry[model_name](quality=quality, pretrained=True)
    model = model.to(device)
    model.eval()
    model.update()  # Mise à jour des tables d'entropie

    # Compter les paramètres
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  → Modèle chargé ({n_params/1e6:.1f}M paramètres) sur {device}")

    return model


# ============================================================================
# Fonctions de compression / décompression neuronale
# ============================================================================

def pad_to_multiple(tensor: torch.Tensor, multiple: int = 64) -> tuple:
    """
    Pad un tenseur image pour que H et W soient multiples de `multiple`.
    CompressAI nécessite des dimensions divisibles par 64 (ou 32 selon le modèle).

    Returns
    -------
    tuple : (tensor_padded, padding_info) pour pouvoir dé-padder ensuite.
    """
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

    return tensor, {"pad_h": pad_h, "pad_w": pad_w, "orig_h": h, "orig_w": w}


def unpad(tensor: torch.Tensor, padding_info: dict) -> torch.Tensor:
    """Retire le padding ajouté par pad_to_multiple."""
    h = padding_info["orig_h"]
    w = padding_info["orig_w"]
    return tensor[:, :, :h, :w]


def compress_and_reconstruct(
    model,
    image_tensor: torch.Tensor,
    device: str = "cuda",
) -> dict:
    """
    Compresse et reconstruit une image avec un modèle CompressAI.

    Parameters
    ----------
    model : nn.Module
        Modèle CompressAI chargé.
    image_tensor : torch.Tensor
        Image [1, C, H, W] normalisée [0, 1].
    device : str
        Device.

    Returns
    -------
    dict :
        - "reconstructed" : torch.Tensor [1, C, H, W] en [0, 1]
        - "bitrate_bpp" : float, bits per pixel réels
        - "encoded_size_bytes" : int, taille du bitstream compressé
    """
    _, _, orig_h, orig_w = image_tensor.shape
    total_pixels = orig_h * orig_w

    # Padding
    x_padded, pad_info = pad_to_multiple(image_tensor.to(device), multiple=64)

    with torch.no_grad():
        # Méthode 1 : compress + decompress (donne le vrai bitstream)
        try:
            compressed = model.compress(x_padded)
            decompressed = model.decompress(
                compressed["strings"],
                compressed["shape"],
            )
            x_hat = decompressed["x_hat"]

            # Calculer la taille réelle du bitstream
            total_bytes = sum(
                sum(len(s) for s in string_list)
                for string_list in compressed["strings"]
            )

        except Exception:
            # Fallback : forward pass (estime le bitrate via les likelihoods)
            logger.debug("  Fallback vers forward pass (pas de bitstream réel)")
            output = model(x_padded)
            x_hat = output["x_hat"]

            # Estimer le bitrate via les likelihoods
            total_bits = 0
            for key in ["likelihoods"]:
                if key in output:
                    for name, likelihood in output[key].items():
                        total_bits += torch.sum(
                            -torch.log2(torch.clamp(likelihood, min=1e-10))
                        ).item()

            total_bytes = int(total_bits / 8)

    # Dé-padding
    x_hat = unpad(x_hat, pad_info)

    # Clamp [0, 1]
    x_hat = torch.clamp(x_hat, 0.0, 1.0)

    # Bitrate réel
    bpp = (total_bytes * 8) / total_pixels

    return {
        "reconstructed": x_hat,
        "bitrate_bpp": round(bpp, 4),
        "encoded_size_bytes": total_bytes,
    }


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convertit un tenseur [1, C, H, W] en [0,1] vers une PIL Image."""
    img_np = tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_np)


def image_to_tensor(image_path: str) -> tuple:
    """
    Charge une image et la convertit en tenseur [1, C, H, W] normalisé [0, 1].

    Returns
    -------
    tuple : (tensor, width, height)
    """
    img = Image.open(image_path)

    # Convertir en RGB
    if img.mode in ("RGBA", "P", "LA", "L"):
        img = img.convert("RGB")

    width, height = img.size

    to_tensor = transforms.ToTensor()  # Normalise automatiquement en [0, 1]
    tensor = to_tensor(img).unsqueeze(0)  # [1, C, H, W]

    return tensor, width, height


# ============================================================================
# SSIM
# ============================================================================

def compute_ssim_score(original_tensor: torch.Tensor, reconstructed_tensor: torch.Tensor,
                       device: str = "cuda") -> float:
    """Calcule le SSIM entre l'original et la reconstruction (tenseurs [1,C,H,W])."""
    import piq

    orig = original_tensor.to(device)
    recon = reconstructed_tensor.to(device)

    if orig.shape != recon.shape:
        # Redimensionner si nécessaire
        recon = F.interpolate(recon, size=orig.shape[2:], mode="bilinear", align_corners=False)

    with torch.no_grad():
        ssim_val = piq.ssim(orig, recon, data_range=1.0).item()

    return round(ssim_val, 6)


# ============================================================================
# Insertion base de données
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


# ============================================================================
# Pipeline principal
# ============================================================================

def run_pipeline(
    input_dir: str,
    output_dir: str,
    gt_csv_path: str,
    model_name: str,
    quality_levels: list,
    db_url: str,
    compute_ssim: bool = True,
    save_format: str = "png",
    verbose: bool = False,
    batch_commit_size: int = 20,
):
    """
    Pipeline complet de compression neuronale batch.

    Parameters
    ----------
    input_dir : str
        Dossier contenant les images PNG.
    output_dir : str
        Dossier racine de sortie.
    gt_csv_path : str
        CSV du ground-truth (issu de prepare_ground_truth.py).
    model_name : str
        Nom du modèle CompressAI.
    quality_levels : list of int
        Niveaux de qualité CompressAI (ex: [1, 3, 6]).
    db_url : str
        URL de connexion PostgreSQL.
    compute_ssim : bool
        Calculer le SSIM.
    save_format : str
        Format de sauvegarde de la reconstruction ("png" ou "bmp").
        PNG est lossless → la reconstruction est fidèle.
    verbose : bool
        Afficher les détails.
    batch_commit_size : int
        Nombre d'images avant commit intermédiaire.
    """

    # ------------------------------------------------------------------
    # 1. Device & GPU
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        logger.info(f"GPU détecté : {gpu_name} ({gpu_mem:.1f} Go)")
    else:
        logger.warning("Pas de GPU détecté — compression CPU (très lent)")

    # ------------------------------------------------------------------
    # 2. Charger le CSV ground-truth
    # ------------------------------------------------------------------
    logger.info(f"Chargement du CSV ground-truth : {gt_csv_path}")
    if not os.path.exists(gt_csv_path):
        logger.error(f"CSV introuvable : {gt_csv_path}")
        sys.exit(1)

    gt_df = pd.read_csv(gt_csv_path)
    logger.info(f"  → {len(gt_df)} images dans le CSV")
    gt_index = gt_df.set_index("filename").to_dict("index")

    # ------------------------------------------------------------------
    # 3. Lister les images PNG
    # ------------------------------------------------------------------
    logger.info(f"Scan du dossier d'entrée : {input_dir}")
    png_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".PNG"))
    ])
    logger.info(f"  → {len(png_files)} images PNG trouvées")

    if not png_files:
        logger.error("Aucune image PNG trouvée.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Connexion base de données
    # ------------------------------------------------------------------
    logger.info(f"Connexion à la base de données...")
    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    logger.info(f"  → Connexion établie")

    # ------------------------------------------------------------------
    # 5. Créer les sous-dossiers de sortie par niveau
    # ------------------------------------------------------------------
    for ql in quality_levels:
        ql_dir = os.path.join(output_dir, f"q{ql}")
        os.makedirs(ql_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 6. Boucle sur les niveaux de qualité (un modèle par niveau)
    # ------------------------------------------------------------------
    logger.info(f"\nDémarrage de la compression neuronale batch")
    logger.info(f"  Modèle   : {model_name}")
    logger.info(f"  Niveaux  : {quality_levels}")
    logger.info(f"  Device   : {device}")
    logger.info(f"  SSIM     : {'Oui' if compute_ssim else 'Non'}")
    logger.info(f"  Sortie   : {output_dir}")
    logger.info(f"{'='*60}\n")

    stats = {
        "total_images": len(png_files),
        "total_compressions": 0,
        "skipped": 0,
        "errors": 0,
        "by_quality": {
            ql: {"count": 0, "total_bpp": 0, "total_size_kb": 0, "ssim_sum": 0}
            for ql in quality_levels
        },
    }

    t_start = time.time()

    for ql in quality_levels:
        logger.info(f"\n{'─'*60}")
        logger.info(f"Niveau de qualité : {ql} ({model_name})")
        logger.info(f"{'─'*60}")

        # Charger le modèle pour ce niveau
        model = load_compressai_model(model_name, quality=ql, device=device)

        for idx, png_filename in enumerate(tqdm(
            png_files, desc=f"Neural q{ql}", unit="img"
        )):
            input_path = os.path.join(input_dir, png_filename)

            # Lookup GT
            gt_info = gt_index.get(png_filename, {})
            image_id = gt_info.get("image_id", idx)
            doc_category = gt_info.get("doc_category", "Unknown")
            split = gt_info.get("split", "unknown")

            # Chemin de sortie
            out_filename = Path(png_filename).stem + f".{save_format}"
            out_path = os.path.join(output_dir, f"q{ql}", out_filename)

            # Vérifier si déjà traité
            existing = (
                session.query(CompressionRecord)
                .filter_by(
                    image_id=int(image_id),
                    compression_type="neural",
                    compression_level=ql,
                )
                .first()
            )
            if existing:
                stats["skipped"] += 1
                continue

            try:
                # Charger l'image en tenseur
                img_tensor, width, height = image_to_tensor(input_path)
                original_size_bytes = os.path.getsize(input_path)
                original_size_kb = original_size_bytes / 1024

                # Compresser et reconstruire
                result = compress_and_reconstruct(model, img_tensor, device=device)

                # Sauvegarder la reconstruction en PNG (lossless)
                recon_img = tensor_to_image(result["reconstructed"])
                recon_img.save(out_path)
                saved_size_bytes = os.path.getsize(out_path)

                # SSIM entre original et reconstruction
                ssim_val = None
                if compute_ssim:
                    ssim_val = compute_ssim_score(
                        img_tensor, result["reconstructed"], device=device
                    )

                # Bitrate et taille du bitstream compressé (pas du fichier PNG sauvé)
                bitrate_bpp = result["bitrate_bpp"]
                encoded_size_kb = result["encoded_size_bytes"] / 1024
                compression_ratio = (
                    original_size_bytes / result["encoded_size_bytes"]
                    if result["encoded_size_bytes"] > 0 else 0
                )

                # Insérer l'image originale
                ensure_image_in_db(
                    session=session,
                    image_id=int(image_id),
                    filename=png_filename,
                    category=doc_category,
                    split=split,
                    width=width,
                    height=height,
                    original_size_kb=round(original_size_kb, 2),
                    original_path=input_path,
                )

                # Insérer le résultat de compression
                comp_record = CompressionRecord(
                    image_id=int(image_id),
                    compression_type="neural",
                    compression_level=ql,
                    quality_label=QUALITY_LABELS.get(ql, f"neural_q{ql}"),
                    bitrate_bpp=bitrate_bpp,
                    file_size_kb=round(encoded_size_kb, 2),
                    compression_ratio=round(compression_ratio, 2),
                    ssim=ssim_val,
                    compressed_path=out_path,
                )
                session.add(comp_record)

                # Stats
                stats["total_compressions"] += 1
                ql_stats = stats["by_quality"][ql]
                ql_stats["count"] += 1
                ql_stats["total_bpp"] += bitrate_bpp
                ql_stats["total_size_kb"] += encoded_size_kb
                if ssim_val and ssim_val > 0:
                    ql_stats["ssim_sum"] += ssim_val

                if verbose:
                    logger.info(
                        f"  {png_filename} → q{ql} : "
                        f"{bitrate_bpp:.4f} bpp, "
                        f"{encoded_size_kb:.1f} Ko (bitstream), "
                        f"SSIM={ssim_val:.4f}" if ssim_val else ""
                    )

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                stats["errors"] += 1
                logger.error(
                    f"OOM sur {png_filename} (q{ql}) — "
                    f"image trop grande pour le GPU. "
                    f"Essayer --resize ou réduire les images."
                )
                continue

            except Exception as e:
                stats["errors"] += 1
                logger.error(f"Erreur sur {png_filename} (q{ql}) : {e}")
                continue

            # Commit intermédiaire
            if (idx + 1) % batch_commit_size == 0:
                session.commit()
                # Libérer la mémoire GPU périodiquement
                if device == "cuda":
                    torch.cuda.empty_cache()

        # Commit après chaque niveau de qualité
        session.commit()

        # Libérer le modèle avant de charger le suivant
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        logger.info(f"  → Niveau q{ql} terminé, modèle libéré de la mémoire GPU")

    session.close()
    t_elapsed = time.time() - t_start

    # ------------------------------------------------------------------
    # 7. Rapport de synthèse
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"RAPPORT — Compression Neuronale Batch ({model_name})")
    print(f"{'='*60}")
    print(f"Images traitées     : {stats['total_images']}")
    print(f"Compressions créées : {stats['total_compressions']}")
    print(f"Skipped (déjà fait) : {stats['skipped']}")
    print(f"Erreurs             : {stats['errors']}")
    print(f"Temps total         : {t_elapsed:.1f}s ({t_elapsed/60:.1f} min)")
    print()
    print(f"{'Qualité':>8} {'Count':>8} {'bpp moy':>10} {'Taille moy (Ko)':>16} {'SSIM moy':>10}")
    print(f"{'-'*56}")
    for ql in quality_levels:
        ql_stats = stats["by_quality"][ql]
        n = ql_stats["count"]
        if n > 0:
            avg_bpp = ql_stats["total_bpp"] / n
            avg_size = ql_stats["total_size_kb"] / n
            avg_ssim = ql_stats["ssim_sum"] / n if compute_ssim else 0
            print(f"  q{ql:>5} {n:>8} {avg_bpp:>10.4f} {avg_size:>16.1f} {avg_ssim:>10.4f}")
    print(f"{'='*60}")
    print(f"Reconstructions dans : {output_dir}")
    print(f"Métadonnées en base PostgreSQL (table compressions, type='neural')")
    print(f"{'='*60}")
    print()

    # ------------------------------------------------------------------
    # 8. Table de correspondance JPEG ↔ Neural pour iso-bitrate
    # ------------------------------------------------------------------
    print(f"{'─'*60}")
    print(f"CORRESPONDANCE ISO-BITRATE (approximative)")
    print(f"{'─'*60}")
    print(f"  Neural q1 (~0.1 bpp)   ↔  JPEG QF ~10")
    print(f"  Neural q3 (~0.25 bpp)  ↔  JPEG QF ~30")
    print(f"  Neural q6 (~0.5 bpp)   ↔  JPEG QF ~50–70")
    print(f"{'─'*60}")
    print(f"Note : Les bitrates réels varient par image. Pour une comparaison")
    print(f"rigoureuse à iso-bitrate, utiliser les valeurs bpp réelles en base.")
    print(f"{'─'*60}")
    print(f"\nProchaine étape : Prompt 3.1 (inférence Qwen2-VL)")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de compression neuronale batch avec CompressAI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  # Compression complète (3 niveaux, Cheng2020)
  python compression/neural_pipeline/compress_neural.py \\
      --input-dir data/raw \\
      --output-dir data/compressed/neural \\
      --gt-csv data/metadata/ground_truth.csv \\
      --model cheng2020-anchor \\
      --quality-levels 1 3 6 \\
      --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression"

  # Test rapide sur un seul niveau, sans SSIM
  python compression/neural_pipeline/compress_neural.py \\
      --input-dir data/raw \\
      --output-dir data/compressed/neural \\
      --gt-csv data/metadata/ground_truth.csv \\
      --model cheng2020-anchor \\
      --quality-levels 3 \\
      --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression" \\
      --no-ssim

  # Avec le modèle attention (meilleure qualité, plus lent)
  python compression/neural_pipeline/compress_neural.py \\
      --input-dir data/raw \\
      --output-dir data/compressed/neural \\
      --gt-csv data/metadata/ground_truth.csv \\
      --model cheng2020-attn \\
      --quality-levels 1 3 6 \\
      --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression"
        """,
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Dossier contenant les images PNG originales",
    )
    parser.add_argument(
        "--output-dir",
        default="data/compressed/neural",
        help="Dossier racine de sortie (défaut: data/compressed/neural)",
    )
    parser.add_argument(
        "--gt-csv",
        required=True,
        help="CSV ground-truth issu de prepare_ground_truth.py",
    )
    parser.add_argument(
        "--model",
        default="cheng2020-anchor",
        choices=["cheng2020-anchor", "cheng2020-attn", "mbt2018", "mbt2018-mean"],
        help="Modèle CompressAI (défaut: cheng2020-anchor)",
    )
    parser.add_argument(
        "--quality-levels",
        nargs="+",
        type=int,
        default=[1, 3, 6],
        help="Niveaux de qualité CompressAI (défaut: 1 3 6 → ~0.1/0.25/0.5 bpp)",
    )
    parser.add_argument(
        "--db-url",
        required=True,
        help="URL PostgreSQL (ex: postgresql://user:pass@localhost:5432/db)",
    )
    parser.add_argument(
        "--save-format",
        default="png",
        choices=["png", "bmp"],
        help="Format de sauvegarde des reconstructions (défaut: png, lossless)",
    )
    parser.add_argument(
        "--no-ssim",
        action="store_true",
        help="Désactiver le calcul SSIM (plus rapide)",
    )
    parser.add_argument(
        "--batch-commit",
        type=int,
        default=20,
        help="Nombre d'images entre chaque commit SQL (défaut: 20)",
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

    max_q = MODEL_MAX_QUALITY.get(args.model, 6)
    for ql in args.quality_levels:
        if not 1 <= ql <= max_q:
            logger.error(f"Quality level {ql} invalide pour {args.model} (1–{max_q})")
            sys.exit(1)

    run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        gt_csv_path=args.gt_csv,
        model_name=args.model,
        quality_levels=sorted(args.quality_levels),
        db_url=args.db_url,
        compute_ssim=not args.no_ssim,
        save_format=args.save_format,
        verbose=args.verbose,
        batch_commit_size=args.batch_commit,
    )


if __name__ == "__main__":
    main()
