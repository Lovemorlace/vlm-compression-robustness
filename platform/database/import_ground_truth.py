#!/usr/bin/env python3
"""
=============================================================================
import_ground_truth.py
=============================================================================
Importe le CSV ground-truth (issu de prepare_ground_truth.py) directement
dans les tables `images` et `ground_truth` de PostgreSQL.

Alternative à l'import via \COPY + SQL function pour ceux qui préfèrent
un script Python.

Usage :
    conda activate platform
    python platform/database/import_ground_truth.py \
        --gt-csv data/metadata/ground_truth.csv \
        --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression"
=============================================================================
"""

import argparse
import logging
import os
import re
import sys
import unicodedata

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, Float, String, Text, DateTime
from datetime import datetime
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


class GroundTruthRecord(Base):
    __tablename__ = "ground_truth"
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, unique=True, nullable=False, index=True)
    gt_text = Column(Text)
    gt_text_normalized = Column(Text)
    num_annotations = Column(Integer, default=0)
    num_text_annotations = Column(Integer, default=0)
    num_characters = Column(Integer, default=0)
    num_words = Column(Integer, default=0)
    layout_types = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Import ground-truth CSV vers PostgreSQL")
    parser.add_argument("--gt-csv", required=True, help="Chemin du CSV ground-truth")
    parser.add_argument("--db-url", required=True, help="URL PostgreSQL")
    parser.add_argument("--images-dir", default=None, help="Dossier des images PNG (pour remplir original_path)")
    args = parser.parse_args()

    if not os.path.exists(args.gt_csv):
        logger.error(f"CSV introuvable : {args.gt_csv}")
        sys.exit(1)

    # Charger le CSV
    df = pd.read_csv(args.gt_csv)
    logger.info(f"CSV chargé : {len(df)} lignes")

    # Connexion
    engine = create_engine(args.db_url, echo=False)
    Session = sessionmaker(bind=engine)
    session = Session()

    n_images = 0
    n_gt = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Import GT"):
        image_id = int(row["image_id"])
        filename = str(row["filename"])
        gt_text = str(row["gt_text"]) if pd.notna(row["gt_text"]) else ""
        gt_norm = normalize_text(gt_text)

        original_path = None
        if args.images_dir:
            candidate = os.path.join(args.images_dir, filename)
            if os.path.exists(candidate):
                original_path = candidate

        # Image
        existing_img = session.query(ImageRecord).filter_by(image_id=image_id).first()
        if not existing_img:
            img = ImageRecord(
                image_id=image_id,
                filename=filename,
                category=str(row.get("doc_category", "Unknown")),
                split=str(row.get("split", "unknown")),
                width=int(row["width"]) if pd.notna(row.get("width")) else None,
                height=int(row["height"]) if pd.notna(row.get("height")) else None,
                original_path=original_path,
            )
            session.add(img)
            n_images += 1

        # Ground-truth
        existing_gt = session.query(GroundTruthRecord).filter_by(image_id=image_id).first()
        if not existing_gt:
            gt = GroundTruthRecord(
                image_id=image_id,
                gt_text=gt_text,
                gt_text_normalized=gt_norm,
                num_annotations=int(row.get("num_annotations", 0)) if pd.notna(row.get("num_annotations")) else 0,
                num_text_annotations=int(row.get("num_text_annotations", 0)) if pd.notna(row.get("num_text_annotations")) else 0,
                num_characters=int(row.get("num_characters", 0)) if pd.notna(row.get("num_characters")) else 0,
                num_words=len(gt_norm.split()) if gt_norm else 0,
                layout_types=str(row.get("layout_types", "")) if pd.notna(row.get("layout_types")) else "",
            )
            session.add(gt)
            n_gt += 1

    session.commit()
    session.close()

    print(f"\n{'='*60}")
    print(f"Import terminé")
    print(f"  Images insérées       : {n_images}")
    print(f"  Ground-truths insérés : {n_gt}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
