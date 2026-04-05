#!/usr/bin/env python3
"""
=============================================================================
infer_qwen2vl.py — Prompt 3.1
=============================================================================
Script d'inférence pour Qwen2-VL-7B-Instruct :
  - Lit la liste des images à traiter depuis la base PostgreSQL
    (originales + toutes les versions compressées JPEG et neuronales)
  - Envoie chaque image au VLM avec un prompt de transcription identique
  - Sauvegarde le texte prédit dans la table `predictions` de PostgreSQL
  - Gère la reprise après interruption (skip les prédictions existantes)
  - Support quantization 4-bit pour GPU avec 16 Go VRAM

Usage :
    conda activate vlm_qwen
    python inference/vlm_qwen/infer_qwen2vl.py \
        --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression" \
        --batch-size 1 \
        --quantize 4bit \
        --verbose

    # Uniquement les images baseline (originales PNG)
    python inference/vlm_qwen/infer_qwen2vl.py \
        --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression" \
        --filter-type baseline

    # Uniquement JPEG QF=50
    python inference/vlm_qwen/infer_qwen2vl.py \
        --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression" \
        --filter-type jpeg --filter-level 50

    # Limiter à N images (test rapide)
    python inference/vlm_qwen/infer_qwen2vl.py \
        --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression" \
        --limit 20

Dépendances : transformers, torch, accelerate, bitsandbytes, qwen-vl-utils,
              Pillow, psycopg2, sqlalchemy, pandas, tqdm
=============================================================================
"""

import argparse
import gc
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

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
from sqlalchemy import text as sql_text

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
    """Table des prédictions VLM."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey("images.image_id"), nullable=False, index=True)
    compression_id = Column(Integer, ForeignKey("compressions.id"), nullable=True)
    vlm_name = Column(String(64), nullable=False, index=True)
    compression_type = Column(String(32), nullable=False)       # "baseline", "jpeg", "neural"
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
VLM_NAME = "qwen2-vl"
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

DEFAULT_PROMPT = (
    "Transcribe all visible text in this document image exactly as written, "
    "preserving the reading order from top to bottom, left to right. "
    "Output only the transcribed text, nothing else."
)


# ============================================================================
# Chargement du modèle Qwen2-VL
# ============================================================================

def load_model(quantize: str = "none", device_map: str = "auto"):
    """
    Charge Qwen2-VL-7B-Instruct.

    Parameters
    ----------
    quantize : str
        "none" : FP16 (nécessite ~16 Go VRAM)
        "4bit" : Quantization 4-bit via bitsandbytes (~8 Go VRAM)
        "8bit" : Quantization 8-bit (~10 Go VRAM)
    device_map : str
        Stratégie de placement (défaut: "auto").

    Returns
    -------
    tuple : (model, processor)
    """
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

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
        # Retirer torch_dtype quand on utilise quantization
        load_kwargs.pop("torch_dtype", None)

    elif quantize == "8bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        load_kwargs.pop("torch_dtype", None)

    # Charger le modèle
    model = Qwen2VLForConditionalGeneration.from_pretrained(**load_kwargs)
    model.eval()

    # Charger le processor
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    # Info mémoire
    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated() / (1024**3)
        mem_total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        logger.info(f"  → Modèle chargé : {mem_alloc:.1f} / {mem_total:.1f} Go VRAM")
    else:
        logger.info(f"  → Modèle chargé sur CPU")

    return model, processor


# ============================================================================
# Inférence sur une image
# ============================================================================

def run_inference_single(
    model,
    processor,
    image_path: str,
    prompt: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
) -> dict:
    """
    Exécute l'inférence Qwen2-VL sur une seule image.

    Parameters
    ----------
    model : Qwen2VLForConditionalGeneration
    processor : AutoProcessor
    image_path : str
        Chemin vers l'image.
    prompt : str
        Prompt de transcription.
    max_new_tokens : int
        Nombre max de tokens à générer.
    temperature : float
        Température de génération (0.0 = déterministe).

    Returns
    -------
    dict : {"predicted_text": str, "inference_time_s": float, "num_tokens": int}
    """
    # Vérifier que l'image existe
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    # Charger l'image
    image = Image.open(image_path).convert("RGB")

    # Construire le message au format Qwen2-VL
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Appliquer le template de chat
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Préparer les inputs
    inputs = processor(
        text=[text_input],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    # Déplacer sur le device du modèle
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    # Génération
    t_start = time.time()

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    t_elapsed = time.time() - t_start

    # Décoder — retirer les tokens d'entrée
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, input_len:]
    predicted_text = processor.decode(generated_ids, skip_special_tokens=True).strip()

    num_tokens = len(generated_ids)

    return {
        "predicted_text": predicted_text,
        "inference_time_s": round(t_elapsed, 3),
        "num_tokens": num_tokens,
    }


# ============================================================================
# Construction de la liste de tâches depuis la base
# ============================================================================

def build_task_list(session: Session, filter_type: str, filter_level: int,
                    category: str, limit: int) -> list:
    """
    Construit la liste des (image_id, image_path, compression_type,
    compression_level, compression_id) à traiter.

    Inclut :
    - Les images baseline (PNG originales) → compression_type="baseline"
    - Les images compressées JPEG → compression_type="jpeg"
    - Les images compressées neural → compression_type="neural"

    Parameters
    ----------
    session : Session
    filter_type : str or None
        Filtrer par type : "baseline", "jpeg", "neural", ou None (tout).
    filter_level : int or None
        Filtrer par niveau de compression.
    category : str or None
        Filtrer par catégorie documentaire.
    limit : int or None
        Limiter le nombre total de tâches.

    Returns
    -------
    list of dict : Tâches d'inférence.
    """
    tasks = []

    # --- Baseline (images originales) ---
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

    # Limiter si demandé
    if limit:
        tasks = tasks[:limit]

    return tasks


def filter_already_done(session: Session, tasks: list, vlm_name: str) -> list:
    """Retire les tâches dont la prédiction existe déjà en base."""
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
    filter_type: str,
    filter_level: int,
    category: str,
    limit: int,
    batch_commit_size: int,
    verbose: bool,
):
    """Pipeline complet d'inférence Qwen2-VL."""

    # ------------------------------------------------------------------
    # 1. Connexion base de données
    # ------------------------------------------------------------------
    logger.info("Connexion à la base de données...")
    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    logger.info("  → Connexion établie, table predictions créée/vérifiée")

    # ------------------------------------------------------------------
    # 2. Construire la liste de tâches
    # ------------------------------------------------------------------
    logger.info("\nConstruction de la liste de tâches...")
    all_tasks = build_task_list(session, filter_type, filter_level, category, limit)
    logger.info(f"  → {len(all_tasks)} tâches identifiées")

    # Filtrer celles déjà faites
    tasks = filter_already_done(session, all_tasks, VLM_NAME)
    n_skipped = len(all_tasks) - len(tasks)
    logger.info(f"  → {n_skipped} déjà traitées (skip)")
    logger.info(f"  → {len(tasks)} tâches restantes")

    if not tasks:
        logger.info("Rien à faire — toutes les prédictions existent déjà.")
        session.close()
        return

    # Résumé des tâches par type
    type_counts = {}
    for t in tasks:
        key = f"{t['compression_type']}"
        if t['compression_level'] is not None:
            key += f" (level={t['compression_level']})"
        type_counts[key] = type_counts.get(key, 0) + 1

    logger.info("\n  Répartition des tâches :")
    for key, count in sorted(type_counts.items()):
        logger.info(f"    {key:30s} : {count}")

    # ------------------------------------------------------------------
    # 3. Charger le modèle
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    model, processor = load_model(quantize=quantize)
    logger.info(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 4. Boucle d'inférence
    # ------------------------------------------------------------------
    logger.info(f"Démarrage de l'inférence {VLM_NAME}")
    logger.info(f"  Prompt    : {prompt[:80]}...")
    logger.info(f"  Max tokens: {max_new_tokens}")
    logger.info(f"  Temp.     : {temperature}")
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
            # Inférence
            result = run_inference_single(
                model=model,
                processor=processor,
                image_path=task["image_path"],
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
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
                    f"  [{task['compression_type']}{'/' + str(task['compression_level']) if task['compression_level'] else ''}] "
                    f"{task['filename']} → {result['num_tokens']} tokens, "
                    f"{result['inference_time_s']:.1f}s — \"{preview}...\""
                )

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            stats["errors"] += 1
            logger.error(f"  OOM sur {task['filename']} — skip")
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

            # Libérer la mémoire GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Commit final
    session.commit()
    session.close()

    t_pipeline_total = time.time() - t_pipeline_start

    # ------------------------------------------------------------------
    # 5. Libérer le modèle
    # ------------------------------------------------------------------
    del model, processor
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
    print(f"Modèle              : {MODEL_ID}")
    print(f"Tâches totales      : {stats['total']}")
    print(f"Succès              : {stats['success']}")
    print(f"Erreurs             : {stats['errors']}")
    print(f"Tokens générés      : {stats['total_tokens']}")
    print(f"Temps total inférence: {stats['total_time']:.1f}s ({stats['total_time']/60:.1f} min)")
    print(f"Temps pipeline total : {t_pipeline_total:.1f}s ({t_pipeline_total/60:.1f} min)")
    print(f"Moyenne par image   : {avg_time:.2f}s, {avg_tokens:.0f} tokens")
    print(f"{'='*60}")
    print(f"Prédictions sauvegardées en base (table predictions, vlm='{VLM_NAME}')")
    print(f"{'='*60}")
    print(f"\nProchaine étape : Prompt 3.2 (inférence InternVL2)")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=f"Inférence {VLM_NAME} sur les images DocLayNet (originales + compressées).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  # Inférence complète (baseline + JPEG + neural)
  python inference/vlm_qwen/infer_qwen2vl.py \\
      --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression"

  # Uniquement baseline (images originales)
  python inference/vlm_qwen/infer_qwen2vl.py \\
      --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression" \\
      --filter-type baseline

  # Uniquement JPEG QF=50
  python inference/vlm_qwen/infer_qwen2vl.py \\
      --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression" \\
      --filter-type jpeg --filter-level 50

  # Test rapide sur 10 images, 4-bit
  python inference/vlm_qwen/infer_qwen2vl.py \\
      --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression" \\
      --limit 10 --quantize 4bit --verbose

  # Filtrer par catégorie documentaire
  python inference/vlm_qwen/infer_qwen2vl.py \\
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
        help="Prompt de transcription (défaut: prompt standard)",
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
        help="Température de génération. 0.0 = déterministe (défaut: 0.0)",
    )
    parser.add_argument(
        "--quantize",
        choices=["none", "4bit", "8bit"],
        default="none",
        help="Quantization du modèle (défaut: none → FP16)",
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
        help="Ne traiter que ce niveau de compression (ex: 50 pour JPEG QF50)",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Filtrer par catégorie documentaire (ex: Financial, Scientific)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limiter le nombre total de tâches (pour tests rapides)",
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
        filter_type=args.filter_type,
        filter_level=args.filter_level,
        category=args.category,
        limit=args.limit,
        batch_commit_size=args.batch_commit,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
