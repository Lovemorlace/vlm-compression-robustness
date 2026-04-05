#!/usr/bin/env python3
"""
=============================================================================
infer_internvl2.py — Prompt 3.2
=============================================================================
Script d'inférence pour InternVL2-8B :
  - Même interface CLI que infer_qwen2vl.py (Prompt 3.1)
  - Même schéma de sortie dans la table `predictions`
  - Même prompt de transcription → résultats directement comparables
  - Lit la liste des images depuis PostgreSQL (baseline + compressées)
  - Sauvegarde le texte prédit dans la table `predictions` (vlm_name="internvl2")

Usage :
    conda activate vlm_internvl
    python inference/vlm_internvl/infer_internvl2.py \
        --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression" \
        --batch-size 1 \
        --quantize 4bit \
        --verbose

    # Uniquement les images baseline
    python inference/vlm_internvl/infer_internvl2.py \
        --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression" \
        --filter-type baseline

    # Test rapide
    python inference/vlm_internvl/infer_internvl2.py \
        --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression" \
        --limit 20 --quantize 4bit --verbose

Dépendances : transformers, torch, accelerate, bitsandbytes, timm, einops,
              sentencepiece, Pillow, psycopg2, sqlalchemy, pandas, tqdm
=============================================================================
"""

import argparse
import gc
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
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
# SQLAlchemy — Même schéma que infer_qwen2vl.py
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


# ============================================================================
# Constantes
# ============================================================================
VLM_NAME = "internvl2"
MODEL_ID = "OpenGVLab/InternVL2-8B"

# Même prompt que Qwen2-VL — identique pour assurer la comparabilité
DEFAULT_PROMPT = (
    "Transcribe all visible text in this document image exactly as written, "
    "preserving the reading order from top to bottom, left to right. "
    "Output only the transcribed text, nothing else."
)

# InternVL2 utilise des tuiles dynamiques de 448×448
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================================
# Prétraitement d'image InternVL2 — Dynamic Tiling
# ============================================================================

def build_transform(input_size: int = 448):
    """Construit le pipeline de transformation pour une tuile InternVL2."""
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Trouve le ratio d'aspect le plus proche parmi les ratios cibles."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            # Préférer le ratio avec plus de tuiles (meilleure résolution)
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """
    Découpe dynamiquement une image en tuiles de taille image_size×image_size
    selon le ratio d'aspect optimal pour InternVL2.

    Parameters
    ----------
    image : PIL.Image
    min_num : int
        Nombre minimum de tuiles.
    max_num : int
        Nombre maximum de tuiles.
    image_size : int
        Taille d'une tuile (448 pour InternVL2).
    use_thumbnail : bool
        Ajouter une vignette redimensionnée de l'image complète.

    Returns
    -------
    list of PIL.Image : Liste de tuiles.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Générer les ratios cibles possibles
    target_ratios = set()
    for n in range(min_num, max_num + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i * j <= max_num and i * j >= min_num:
                    target_ratios.add((i, j))

    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Trouver le meilleur ratio
    best_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Redimensionner l'image selon le ratio choisi
    target_width = best_ratio[0] * image_size
    target_height = best_ratio[1] * image_size
    blocks = best_ratio[0] * best_ratio[1]

    resized = image.resize((target_width, target_height))

    # Découper en tuiles
    processed_images = []
    for i in range(best_ratio[1]):      # lignes
        for j in range(best_ratio[0]):  # colonnes
            box = (
                j * image_size,
                i * image_size,
                (j + 1) * image_size,
                (i + 1) * image_size,
            )
            tile = resized.crop(box)
            processed_images.append(tile)

    # Ajouter une vignette globale
    if use_thumbnail and len(processed_images) != 1:
        thumbnail = image.resize((image_size, image_size))
        processed_images.append(thumbnail)

    return processed_images


def load_image_for_internvl(image_path: str, input_size: int = 448,
                            max_num: int = 12) -> torch.Tensor:
    """
    Charge et prétraite une image pour InternVL2 avec dynamic tiling.

    Returns
    -------
    torch.Tensor : [N, C, H, W] avec N = nombre de tuiles.
    """
    image = Image.open(image_path).convert("RGB")
    transform = build_transform(input_size=input_size)

    tiles = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )

    pixel_values = torch.stack([transform(tile) for tile in tiles])
    return pixel_values


# ============================================================================
# Chargement du modèle InternVL2
# ============================================================================

def load_model(quantize: str = "none", device_map: str = "auto"):
    """
    Charge InternVL2-8B.

    Parameters
    ----------
    quantize : str
        "none" : FP16 (~16 Go VRAM)
        "4bit" : Quantization 4-bit (~8 Go VRAM)
        "8bit" : Quantization 8-bit (~10 Go VRAM)

    Returns
    -------
    tuple : (model, tokenizer)
    """
    from transformers import AutoModel, AutoTokenizer

    logger.info(f"Chargement de {MODEL_ID}...")
    logger.info(f"  Quantization : {quantize}")

    load_kwargs = {
        "pretrained_model_name_or_path": MODEL_ID,
        "device_map": device_map,
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
    }

    if quantize == "4bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs.pop("torch_dtype", None)

    elif quantize == "8bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        load_kwargs.pop("torch_dtype", None)

    # Charger le modèle
    model = AutoModel.from_pretrained(**load_kwargs)
    model.eval()

    # Charger le tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        use_fast=False,
    )

    # Info mémoire
    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated() / (1024**3)
        mem_total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        logger.info(f"  → Modèle chargé : {mem_alloc:.1f} / {mem_total:.1f} Go VRAM")
    else:
        logger.info(f"  → Modèle chargé sur CPU")

    return model, tokenizer


# ============================================================================
# Inférence sur une image
# ============================================================================

def run_inference_single(
    model,
    tokenizer,
    image_path: str,
    prompt: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    max_tiles: int = 12,
) -> dict:
    """
    Exécute l'inférence InternVL2 sur une seule image.

    Parameters
    ----------
    model : AutoModel (InternVL2)
    tokenizer : AutoTokenizer
    image_path : str
    prompt : str
    max_new_tokens : int
    temperature : float
    max_tiles : int
        Nombre max de tuiles dynamiques.

    Returns
    -------
    dict : {"predicted_text": str, "inference_time_s": float, "num_tokens": int}
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    # Charger et prétraiter l'image (dynamic tiling)
    pixel_values = load_image_for_internvl(
        image_path, input_size=448, max_num=max_tiles
    )

    # Déplacer sur le device du modèle
    device = next(model.parameters()).device
    pixel_values = pixel_values.to(device).to(torch.float16)

    # Construire le prompt InternVL2
    # Format : "<image>\n{prompt}" avec un token <image> par tuile
    num_tiles = pixel_values.shape[0]
    image_tokens = "<image>\n" * num_tiles
    full_prompt = image_tokens + prompt

    # Paramètres de génération
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_config["temperature"] = temperature

    # Inférence
    t_start = time.time()

    with torch.no_grad():
        response = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=full_prompt,
            generation_config=generation_config,
        )

    t_elapsed = time.time() - t_start

    # model.chat() retourne directement le texte décodé
    predicted_text = response.strip() if isinstance(response, str) else str(response).strip()

    # Estimer le nombre de tokens générés
    num_tokens = len(tokenizer.encode(predicted_text))

    return {
        "predicted_text": predicted_text,
        "inference_time_s": round(t_elapsed, 3),
        "num_tokens": num_tokens,
    }


# ============================================================================
# Construction de la liste de tâches (identique à infer_qwen2vl.py)
# ============================================================================

def build_task_list(session: Session, filter_type: str, filter_level: int,
                    category: str, limit: int) -> list:
    """Construit la liste des tâches d'inférence depuis la base."""
    tasks = []

    # --- Baseline ---
    if filter_type is None or filter_type == "baseline":
        query = session.query(ImageRecord)
        if category:
            query = query.filter(ImageRecord.category == category)

        for img in query.all():
            if img.original_path and os.path.exists(img.original_path):
                tasks.append({
                    "image_id": img.image_id,
                    "image_path": img.original_path,
                    "compression_type": "baseline",
                    "compression_level": None,
                    "compression_id": None,
                    "filename": img.filename,
                    "category": img.category,
                })

    # --- Images compressées ---
    if filter_type is None or filter_type in ("jpeg", "neural"):
        query = session.query(CompressionRecord, ImageRecord).join(
            ImageRecord, CompressionRecord.image_id == ImageRecord.image_id
        )

        if filter_type and filter_type != "baseline":
            query = query.filter(CompressionRecord.compression_type == filter_type)
        if filter_level is not None:
            query = query.filter(CompressionRecord.compression_level == filter_level)
        if category:
            query = query.filter(ImageRecord.category == category)

        for comp, img in query.all():
            if comp.compressed_path and os.path.exists(comp.compressed_path):
                tasks.append({
                    "image_id": img.image_id,
                    "image_path": comp.compressed_path,
                    "compression_type": comp.compression_type,
                    "compression_level": comp.compression_level,
                    "compression_id": comp.id,
                    "filename": img.filename,
                    "category": img.category,
                })

    if limit:
        tasks = tasks[:limit]

    return tasks


def filter_already_done(session: Session, tasks: list, vlm_name: str) -> list:
    """Retire les tâches déjà traitées."""
    filtered = []
    for task in tasks:
        existing = (
            session.query(PredictionRecord)
            .filter_by(
                image_id=task["image_id"],
                vlm_name=vlm_name,
                compression_type=task["compression_type"],
                compression_level=task["compression_level"],
            )
            .first()
        )
        if not existing:
            filtered.append(task)
    return filtered


# ============================================================================
# Pipeline principal
# ============================================================================

def run_pipeline(
    db_url: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    quantize: str,
    max_tiles: int,
    filter_type: str,
    filter_level: int,
    category: str,
    limit: int,
    batch_commit_size: int,
    verbose: bool,
):
    """Pipeline complet d'inférence InternVL2."""

    # ------------------------------------------------------------------
    # 1. Connexion base de données
    # ------------------------------------------------------------------
    logger.info("Connexion à la base de données...")
    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    logger.info("  → Connexion établie")

    # ------------------------------------------------------------------
    # 2. Construire la liste de tâches
    # ------------------------------------------------------------------
    logger.info("\nConstruction de la liste de tâches...")
    all_tasks = build_task_list(session, filter_type, filter_level, category, limit)
    logger.info(f"  → {len(all_tasks)} tâches identifiées")

    tasks = filter_already_done(session, all_tasks, VLM_NAME)
    n_skipped = len(all_tasks) - len(tasks)
    logger.info(f"  → {n_skipped} déjà traitées (skip)")
    logger.info(f"  → {len(tasks)} tâches restantes")

    if not tasks:
        logger.info("Rien à faire — toutes les prédictions existent déjà.")
        session.close()
        return

    # Résumé par type
    type_counts = {}
    for t in tasks:
        key = f"{t['compression_type']}"
        if t["compression_level"] is not None:
            key += f" (level={t['compression_level']})"
        type_counts[key] = type_counts.get(key, 0) + 1

    logger.info("\n  Répartition des tâches :")
    for key, count in sorted(type_counts.items()):
        logger.info(f"    {key:30s} : {count}")

    # ------------------------------------------------------------------
    # 3. Charger le modèle
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    model, tokenizer = load_model(quantize=quantize)
    logger.info(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 4. Boucle d'inférence
    # ------------------------------------------------------------------
    logger.info(f"Démarrage de l'inférence {VLM_NAME}")
    logger.info(f"  Prompt     : {prompt[:80]}...")
    logger.info(f"  Max tokens : {max_new_tokens}")
    logger.info(f"  Temp.      : {temperature}")
    logger.info(f"  Max tuiles : {max_tiles}")
    logger.info(f"{'='*60}\n")

    stats = {
        "total": len(tasks),
        "success": 0,
        "errors": 0,
        "total_tokens": 0,
        "total_time": 0,
    }

    t_pipeline_start = time.time()

    for idx, task in enumerate(tqdm(tasks, desc=f"Inférence {VLM_NAME}", unit="img")):
        try:
            result = run_inference_single(
                model=model,
                tokenizer=tokenizer,
                image_path=task["image_path"],
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                max_tiles=max_tiles,
            )

            # Insérer en base
            pred_record = PredictionRecord(
                image_id=task["image_id"],
                compression_id=task["compression_id"],
                vlm_name=VLM_NAME,
                compression_type=task["compression_type"],
                compression_level=task["compression_level"],
                prompt_used=prompt,
                predicted_text=result["predicted_text"],
                inference_time_s=result["inference_time_s"],
                num_tokens_generated=result["num_tokens"],
            )
            session.add(pred_record)

            stats["success"] += 1
            stats["total_tokens"] += result["num_tokens"]
            stats["total_time"] += result["inference_time_s"]

            if verbose:
                preview = result["predicted_text"][:100].replace("\n", " ")
                logger.info(
                    f"  [{task['compression_type']}"
                    f"{'/' + str(task['compression_level']) if task['compression_level'] else ''}] "
                    f"{task['filename']} → {result['num_tokens']} tokens, "
                    f"{result['inference_time_s']:.1f}s — \"{preview}...\""
                )

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            stats["errors"] += 1
            logger.error(
                f"  OOM sur {task['filename']} — "
                f"essayer --max-tiles plus bas ou --quantize 4bit"
            )
            continue

        except Exception as e:
            stats["errors"] += 1
            logger.error(f"  Erreur sur {task['filename']} : {e}")
            continue

        # Commit intermédiaire
        if (idx + 1) % batch_commit_size == 0:
            session.commit()
            if verbose:
                logger.info(f"  → Commit ({idx + 1}/{len(tasks)})")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Commit final
    session.commit()
    session.close()

    t_pipeline_total = time.time() - t_pipeline_start

    # ------------------------------------------------------------------
    # 5. Libérer le modèle
    # ------------------------------------------------------------------
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ------------------------------------------------------------------
    # 6. Rapport
    # ------------------------------------------------------------------
    avg_time = stats["total_time"] / stats["success"] if stats["success"] > 0 else 0
    avg_tokens = stats["total_tokens"] / stats["success"] if stats["success"] > 0 else 0

    print(f"\n{'='*60}")
    print(f"RAPPORT — Inférence {VLM_NAME}")
    print(f"{'='*60}")
    print(f"Modèle               : {MODEL_ID}")
    print(f"Tâches totales       : {stats['total']}")
    print(f"Succès               : {stats['success']}")
    print(f"Erreurs              : {stats['errors']}")
    print(f"Tokens générés       : {stats['total_tokens']}")
    print(f"Temps total inférence: {stats['total_time']:.1f}s ({stats['total_time']/60:.1f} min)")
    print(f"Temps pipeline total : {t_pipeline_total:.1f}s ({t_pipeline_total/60:.1f} min)")
    print(f"Moyenne par image    : {avg_time:.2f}s, {avg_tokens:.0f} tokens")
    print(f"{'='*60}")
    print(f"Prédictions sauvegardées en base (table predictions, vlm='{VLM_NAME}')")
    print(f"{'='*60}")
    print()

    # ------------------------------------------------------------------
    # 7. Comparaison rapide avec Qwen2-VL si disponible
    # ------------------------------------------------------------------
    try:
        engine2 = create_engine(db_url, echo=False)
        session2 = sessionmaker(bind=engine2)()

        n_qwen = session2.query(PredictionRecord).filter_by(vlm_name="qwen2-vl").count()
        n_intern = session2.query(PredictionRecord).filter_by(vlm_name="internvl2").count()

        print(f"{'─'*60}")
        print(f"ÉTAT DES PRÉDICTIONS EN BASE")
        print(f"{'─'*60}")
        print(f"  qwen2-vl   : {n_qwen:>6} prédictions")
        print(f"  internvl2  : {n_intern:>6} prédictions")

        if n_qwen > 0 and n_intern > 0:
            print(f"\n  ✓ Les deux VLMs ont des prédictions — prêt pour le calcul des métriques")
        elif n_qwen == 0:
            print(f"\n  ⚠ Aucune prédiction Qwen2-VL — lancer infer_qwen2vl.py d'abord")

        print(f"{'─'*60}")
        session2.close()
    except Exception:
        pass

    print(f"\nProchaine étape : Prompt 4.1 (calcul des métriques CER/WER/BLEU)")


# ============================================================================
# CLI — Même interface que infer_qwen2vl.py
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=f"Inférence {VLM_NAME} sur les images DocLayNet (originales + compressées).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  # Inférence complète (baseline + JPEG + neural)
  python inference/vlm_internvl/infer_internvl2.py \\
      --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression"

  # Uniquement baseline
  python inference/vlm_internvl/infer_internvl2.py \\
      --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression" \\
      --filter-type baseline

  # Uniquement JPEG QF=50
  python inference/vlm_internvl/infer_internvl2.py \\
      --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression" \\
      --filter-type jpeg --filter-level 50

  # Test rapide, 10 images, 4-bit, moins de tuiles
  python inference/vlm_internvl/infer_internvl2.py \\
      --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression" \\
      --limit 10 --quantize 4bit --max-tiles 6 --verbose

  # Filtrer par catégorie
  python inference/vlm_internvl/infer_internvl2.py \\
      --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression" \\
      --category Financial
        """,
    )

    parser.add_argument(
        "--db-url",
        required=True,
        help="URL PostgreSQL",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt de transcription (défaut: prompt standard, identique à Qwen2-VL)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Nombre max de tokens à générer (défaut: 2048)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Température. 0.0 = déterministe (défaut: 0.0)",
    )
    parser.add_argument(
        "--quantize",
        choices=["none", "4bit", "8bit"],
        default="none",
        help="Quantization du modèle (défaut: none → FP16)",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=12,
        help="Nombre max de tuiles dynamiques InternVL2 (défaut: 12). "
             "Réduire si OOM (ex: 6).",
    )
    parser.add_argument(
        "--filter-type",
        choices=["baseline", "jpeg", "neural"],
        default=None,
        help="Ne traiter que ce type de compression",
    )
    parser.add_argument(
        "--filter-level",
        type=int,
        default=None,
        help="Ne traiter que ce niveau de compression",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Filtrer par catégorie documentaire",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limiter le nombre total de tâches",
    )
    parser.add_argument(
        "--batch-commit",
        type=int,
        default=10,
        help="Nombre d'images entre chaque commit SQL (défaut: 10)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Afficher les détails par image",
    )

    args = parser.parse_args()

    run_pipeline(
        db_url=args.db_url,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        quantize=args.quantize,
        max_tiles=args.max_tiles,
        filter_type=args.filter_type,
        filter_level=args.filter_level,
        category=args.category,
        limit=args.limit,
        batch_commit_size=args.batch_commit,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
