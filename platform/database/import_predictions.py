#!/usr/bin/env python3
"""
=============================================================================
import_predictions.py
=============================================================================
Importe les CSV de prédictions générés sur Google Colab dans la table
`predictions` de PostgreSQL local.

Usage :
    conda activate platform
    python platform/database/import_predictions.py \
        --csv results/predictions_qwen2-vl.csv \
        --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression"

    # Importer les deux VLMs d'un coup
    python platform/database/import_predictions.py \
        --csv results/predictions_qwen2-vl.csv results/predictions_internvl2.csv \
        --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression"
=============================================================================
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, Column, Integer, Float, String, Text, DateTime, ForeignKey, UniqueConstraint, Index
from sqlalchemy.orm import declarative_base, sessionmaker
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

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
    __table_args__ = {"extend_existing": True}
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey("images.image_id"), nullable=False)
    compression_type = Column(String(32), nullable=False)
    compression_level = Column(Integer, nullable=True)
    compressed_path = Column(Text)
    file_size_kb = Column(Float)
    bitrate_bpp = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

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
    )


def main():
    parser = argparse.ArgumentParser(description="Import des CSV Colab dans PostgreSQL")
    parser.add_argument("--csv", nargs="+", required=True, help="Chemin(s) CSV de prédictions")
    parser.add_argument("--db-url", required=True, help="URL PostgreSQL")
    args = parser.parse_args()

    engine = create_engine(args.db_url, echo=False)
    Session = sessionmaker(bind=engine)
    session = Session()

    total_inserted = 0
    total_skipped = 0
    total_errors = 0

    for csv_path in args.csv:
        if not os.path.exists(csv_path):
            logger.error(f"Fichier introuvable : {csv_path}")
            continue

        logger.info(f"\nImport de {csv_path}...")
        df = pd.read_csv(csv_path)
        logger.info(f"  → {len(df)} lignes")

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Import {os.path.basename(csv_path)}"):
            image_id = int(row["image_id"])
            vlm_name = str(row["vlm_name"])
            comp_type = str(row["compression_type"])
            comp_level = int(row["compression_level"]) if pd.notna(row.get("compression_level")) else None

            # Vérifier que l'image existe en base
            img = session.query(ImageRecord).filter_by(image_id=image_id).first()
            if not img:
                total_errors += 1
                continue

            # Vérifier doublon
            existing = session.query(PredictionRecord).filter_by(
                image_id=image_id,
                vlm_name=vlm_name,
                compression_type=comp_type,
                compression_level=comp_level,
            ).first()

            if existing:
                total_skipped += 1
                continue

            # Trouver le compression_id correspondant si ce n'est pas baseline
            compression_id = None
            if comp_type != "baseline" and comp_level is not None:
                from sqlalchemy import text as sql_text
                result = session.execute(
                    sql_text(
                        "SELECT id FROM compressions WHERE image_id = :img_id "
                        "AND compression_type = :ctype AND compression_level = :clevel"
                    ),
                    {"img_id": image_id, "ctype": comp_type, "clevel": comp_level},
                ).fetchone()
                if result:
                    compression_id = result[0]

            pred = PredictionRecord(
                image_id=image_id,
                compression_id=compression_id,
                vlm_name=vlm_name,
                compression_type=comp_type,
                compression_level=comp_level,
                prompt_used=str(row.get("prompt_used", "")) if pd.notna(row.get("prompt_used")) else None,
                predicted_text=str(row.get("predicted_text", "")) if pd.notna(row.get("predicted_text")) else "",
                inference_time_s=float(row["inference_time_s"]) if pd.notna(row.get("inference_time_s")) else None,
                num_tokens_generated=int(row["num_tokens_generated"]) if pd.notna(row.get("num_tokens_generated")) else None,
            )
            session.add(pred)
            total_inserted += 1

            if total_inserted % 100 == 0:
                session.commit()

    session.commit()
    session.close()

    print(f"\n{'='*60}")
    print(f"IMPORT TERMINÉ")
    print(f"{'='*60}")
    print(f"  Insérés  : {total_inserted}")
    print(f"  Skippés  : {total_skipped} (doublons)")
    print(f"  Erreurs  : {total_errors} (image_id manquant en base)")
    print(f"{'='*60}")
    print(f"\nProchaine étape : lancer compute_metrics.py pour calculer CER/WER/BLEU")


if __name__ == "__main__":
    main()
