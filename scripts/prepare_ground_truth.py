#!/usr/bin/env python3
"""
=============================================================================
prepare_ground_truth.py — Prompt 1.2
=============================================================================
Lit les annotations COCO JSON de DocLayNet, reconstruit le texte ground-truth
par ordre spatial des bounding boxes (haut→bas, gauche→droite), et sauvegarde
dans un CSV propre.

Usage :
    conda activate metrics
    python scripts/prepare_ground_truth.py \
        --coco-json data/metadata/COCO/val.json \
        --images-dir data/raw \
        --output data/metadata/ground_truth.csv \
        --verbose

Entrée :
    - Fichier(s) COCO JSON de DocLayNet (train.json, val.json, test.json)
      Chaque annotation contient : bbox [x, y, w, h], text, category_id, image_id

Sortie :
    - CSV avec colonnes : image_id, filename, category, doc_category, gt_text,
      num_annotations, num_characters
=============================================================================
"""

import argparse
import json
import os
import sys
import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

# ============================================================================
# Configuration du logging
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Mapping des catégories DocLayNet (11 catégories de layout)
# ============================================================================
DOCLAYNET_CATEGORIES = {
    1: "Caption",
    2: "Footnote",
    3: "Formula",
    4: "List-item",
    5: "Page-footer",
    6: "Page-header",
    7: "Picture",
    8: "Section-header",
    9: "Table",
    10: "Text",
    11: "Title",
}

# Catégories documentaires de DocLayNet (extraites du nom de fichier ou metadata)
DOC_CATEGORIES = [
    "Financial",
    "Scientific",
    "Patent",
    "Law",
    "Government",
    "Manual",
]


def load_coco_json(coco_path: str) -> dict:
    """Charge et valide un fichier COCO JSON de DocLayNet."""
    logger.info(f"Chargement de {coco_path}...")

    if not os.path.exists(coco_path):
        logger.error(f"Fichier introuvable : {coco_path}")
        sys.exit(1)

    with open(coco_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # Validation de la structure COCO
    required_keys = ["images", "annotations"]
    for key in required_keys:
        if key not in coco_data:
            logger.error(f"Clé manquante dans le JSON COCO : '{key}'")
            sys.exit(1)

    n_images = len(coco_data["images"])
    n_annotations = len(coco_data["annotations"])
    logger.info(f"  → {n_images} images, {n_annotations} annotations")

    # Log des catégories si présentes
    if "categories" in coco_data:
        cats = {c["id"]: c["name"] for c in coco_data["categories"]}
        logger.info(f"  → Catégories de layout : {cats}")

    return coco_data


def extract_doc_category(filename: str) -> str:
    """
    Extrait la catégorie documentaire depuis le nom de fichier DocLayNet.

    Les noms DocLayNet suivent souvent le pattern :
        <hash>_<category>_<page>.png
    ou contiennent la catégorie dans le chemin.

    Si impossible à détecter, retourne 'Unknown'.
    """
    filename_lower = filename.lower()

    for cat in DOC_CATEGORIES:
        if cat.lower() in filename_lower:
            return cat

    # Tentative via le chemin (si organisé par dossier)
    return "Unknown"


def sort_annotations_spatial(annotations: list, line_threshold_ratio: float = 0.015) -> list:
    """
    Trie les annotations par ordre de lecture spatial :
    1. Regroupe en lignes (bboxes dont le y est proche)
    2. Trie les lignes de haut en bas
    3. Au sein de chaque ligne, trie de gauche à droite

    Parameters
    ----------
    annotations : list
        Liste d'annotations avec clé 'bbox' au format [x, y, w, h].
    line_threshold_ratio : float
        Seuil relatif pour regrouper en lignes. Deux bboxes sont sur la
        même ligne si |y1 - y2| < seuil * hauteur_page_estimée.

    Returns
    -------
    list : Annotations triées par ordre de lecture.
    """
    if not annotations:
        return []

    # Extraire les coordonnées y (top) de chaque bbox
    y_coords = [ann["bbox"][1] for ann in annotations]
    page_height_est = max(y_coords) - min(y_coords) + 1 if y_coords else 1
    line_threshold = page_height_est * line_threshold_ratio

    # Trier d'abord par y (haut en bas)
    sorted_anns = sorted(annotations, key=lambda a: (a["bbox"][1], a["bbox"][0]))

    # Regrouper en lignes
    lines = []
    current_line = [sorted_anns[0]]

    for ann in sorted_anns[1:]:
        y_current = ann["bbox"][1]
        y_prev = current_line[-1]["bbox"][1]

        if abs(y_current - y_prev) <= line_threshold:
            # Même ligne
            current_line.append(ann)
        else:
            # Nouvelle ligne
            lines.append(current_line)
            current_line = [ann]

    lines.append(current_line)

    # Trier chaque ligne par x (gauche à droite), puis concaténer
    result = []
    for line in lines:
        line_sorted = sorted(line, key=lambda a: a["bbox"][0])
        result.extend(line_sorted)

    return result


def reconstruct_text(annotations_sorted: list, separator: str = " ") -> str:
    """
    Reconstruit le texte complet à partir des annotations triées.

    Gère les cas où :
    - Le champ 'text' est absent ou vide
    - Les annotations sont des éléments non-textuels (Picture, Formula)

    Parameters
    ----------
    annotations_sorted : list
        Annotations triées par sort_annotations_spatial().
    separator : str
        Séparateur entre les blocs de texte.

    Returns
    -------
    str : Texte ground-truth reconstruit.
    """
    text_parts = []

    for ann in annotations_sorted:
        text = ann.get("text", "")

        if text is None:
            text = ""

        # Nettoyer le texte
        text = text.strip()

        # Ignorer les entrées vides
        if not text:
            continue

        # Ajouter un marqueur pour les éléments structurels si pertinent
        cat_id = ann.get("category_id", -1)
        cat_name = DOCLAYNET_CATEGORIES.get(cat_id, "Unknown")

        # Les sections-header et titres ajoutent un saut de ligne avant
        if cat_name in ("Section-header", "Title") and text_parts:
            text_parts.append("\n")

        text_parts.append(text)

        # Saut de ligne après les paragraphes et headers
        if cat_name in ("Text", "Section-header", "Title", "Caption", "Footnote"):
            text_parts.append("\n")

    # Joindre et nettoyer
    full_text = separator.join(text_parts)
    # Normaliser les espaces multiples et sauts de ligne
    full_text = "\n".join(
        line.strip() for line in full_text.splitlines() if line.strip()
    )

    return full_text


def build_image_index(coco_data: dict) -> dict:
    """Construit un index image_id → metadata image."""
    return {img["id"]: img for img in coco_data["images"]}


def build_annotations_by_image(coco_data: dict) -> dict:
    """Regroupe les annotations par image_id."""
    anns_by_image = defaultdict(list)
    for ann in coco_data["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)
    return anns_by_image


def process_coco_file(coco_data: dict, images_dir: str = None, verbose: bool = False) -> pd.DataFrame:
    """
    Traitement principal : reconstruit le GT texte pour chaque image.

    Parameters
    ----------
    coco_data : dict
        Données COCO JSON chargées.
    images_dir : str
        Chemin vers les images (pour vérification d'existence).
    verbose : bool
        Afficher les détails par image.

    Returns
    -------
    pd.DataFrame : DataFrame avec le ground-truth par image.
    """
    image_index = build_image_index(coco_data)
    anns_by_image = build_annotations_by_image(coco_data)

    records = []
    n_skipped = 0
    n_no_text = 0

    for image_id in tqdm(sorted(image_index.keys()), desc="Reconstruction GT"):
        img_info = image_index[image_id]
        filename = img_info.get("file_name", f"unknown_{image_id}")

        # Vérifier l'existence de l'image si le dossier est fourni
        if images_dir:
            img_path = os.path.join(images_dir, filename)
            if not os.path.exists(img_path):
                if verbose:
                    logger.warning(f"Image manquante : {img_path}")

        # Récupérer les annotations de cette image
        image_anns = anns_by_image.get(image_id, [])

        if not image_anns:
            n_skipped += 1
            if verbose:
                logger.warning(f"Aucune annotation pour image {image_id} ({filename})")
            continue

        # Filtrer les annotations qui ont du texte
        text_anns = [a for a in image_anns if a.get("text", "").strip()]

        if not text_anns:
            n_no_text += 1
            if verbose:
                logger.warning(f"Aucun texte dans les annotations de {filename}")
            # On garde quand même l'entrée avec texte vide
            records.append({
                "image_id": image_id,
                "filename": filename,
                "doc_category": extract_doc_category(filename),
                "gt_text": "",
                "num_annotations": len(image_anns),
                "num_text_annotations": 0,
                "num_characters": 0,
                "width": img_info.get("width", 0),
                "height": img_info.get("height", 0),
            })
            continue

        # Trier spatialement
        sorted_anns = sort_annotations_spatial(text_anns)

        # Reconstruire le texte
        gt_text = reconstruct_text(sorted_anns)

        # Extraire les types de layout présents
        layout_types = list(set(
            DOCLAYNET_CATEGORIES.get(a.get("category_id", -1), "Unknown")
            for a in image_anns
        ))

        records.append({
            "image_id": image_id,
            "filename": filename,
            "doc_category": extract_doc_category(filename),
            "gt_text": gt_text,
            "num_annotations": len(image_anns),
            "num_text_annotations": len(text_anns),
            "num_characters": len(gt_text),
            "width": img_info.get("width", 0),
            "height": img_info.get("height", 0),
            "layout_types": "|".join(sorted(layout_types)),
        })

    logger.info(f"  → {len(records)} images traitées")
    logger.info(f"  → {n_skipped} images sans annotation, {n_no_text} sans texte")

    return pd.DataFrame(records)


def generate_summary(df: pd.DataFrame, output_dir: str) -> None:
    """Génère un rapport de synthèse du ground-truth extrait."""
    summary_path = os.path.join(output_dir, "ground_truth_summary.txt")

    lines = [
        "=" * 60,
        "RAPPORT DE SYNTHÈSE — Ground-Truth DocLayNet",
        "=" * 60,
        "",
        f"Nombre total d'images : {len(df)}",
        f"Images avec texte     : {(df['num_characters'] > 0).sum()}",
        f"Images sans texte     : {(df['num_characters'] == 0).sum()}",
        "",
        "--- Statistiques texte ---",
        f"Caractères (moyenne)  : {df['num_characters'].mean():.0f}",
        f"Caractères (médiane)  : {df['num_characters'].median():.0f}",
        f"Caractères (min)      : {df['num_characters'].min()}",
        f"Caractères (max)      : {df['num_characters'].max()}",
        "",
        "--- Répartition par catégorie documentaire ---",
    ]

    cat_counts = df["doc_category"].value_counts()
    for cat, count in cat_counts.items():
        subset = df[df["doc_category"] == cat]
        avg_chars = subset["num_characters"].mean()
        lines.append(f"  {cat:15s} : {count:5d} images  (moy. {avg_chars:.0f} chars)")

    lines += [
        "",
        "--- Annotations par image ---",
        f"Annotations (moyenne) : {df['num_annotations'].mean():.1f}",
        f"Annotations texte (moy) : {df['num_text_annotations'].mean():.1f}",
        "",
        "=" * 60,
    ]

    report = "\n".join(lines)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Rapport de synthèse sauvegardé : {summary_path}")
    print("\n" + report)


def main():
    parser = argparse.ArgumentParser(
        description="Prépare le ground-truth texte à partir des annotations COCO DocLayNet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  # Traiter le split val
  python scripts/prepare_ground_truth.py \\
      --coco-json data/metadata/COCO/val.json \\
      --output data/metadata/ground_truth_val.csv

  # Traiter plusieurs splits
  python scripts/prepare_ground_truth.py \\
      --coco-json data/metadata/COCO/val.json data/metadata/COCO/test.json \\
      --output data/metadata/ground_truth.csv

  # Avec vérification d'existence des images
  python scripts/prepare_ground_truth.py \\
      --coco-json data/metadata/COCO/val.json \\
      --images-dir data/raw \\
      --output data/metadata/ground_truth_val.csv \\
      --verbose
        """,
    )

    parser.add_argument(
        "--coco-json",
        nargs="+",
        required=True,
        help="Chemin(s) vers le(s) fichier(s) COCO JSON DocLayNet",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Dossier contenant les images PNG (pour vérification d'existence)",
    )
    parser.add_argument(
        "--output",
        default="data/metadata/ground_truth.csv",
        help="Chemin du fichier CSV de sortie (défaut: data/metadata/ground_truth.csv)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Afficher les détails de traitement par image",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Ne traiter qu'un échantillon de N images (pour tests rapides)",
    )

    args = parser.parse_args()

    # Créer le dossier de sortie si nécessaire
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Traiter chaque fichier COCO JSON
    all_dfs = []

    for coco_path in args.coco_json:
        logger.info(f"\n{'='*60}")
        logger.info(f"Traitement de : {coco_path}")
        logger.info(f"{'='*60}")

        # Extraire le nom du split depuis le nom de fichier
        split_name = Path(coco_path).stem  # ex: "val", "test", "train"

        coco_data = load_coco_json(coco_path)

        # Échantillonnage si demandé
        if args.sample:
            logger.info(f"  → Échantillon limité à {args.sample} images")
            coco_data["images"] = coco_data["images"][: args.sample]
            # Filtrer les annotations correspondantes
            sampled_ids = {img["id"] for img in coco_data["images"]}
            coco_data["annotations"] = [
                a for a in coco_data["annotations"] if a["image_id"] in sampled_ids
            ]

        df = process_coco_file(coco_data, args.images_dir, args.verbose)
        df["split"] = split_name
        all_dfs.append(df)

    # Concaténer tous les splits
    final_df = pd.concat(all_dfs, ignore_index=True)

    # Réordonner les colonnes
    col_order = [
        "image_id", "filename", "split", "doc_category",
        "gt_text", "num_annotations", "num_text_annotations",
        "num_characters", "width", "height", "layout_types",
    ]
    col_order = [c for c in col_order if c in final_df.columns]
    final_df = final_df[col_order]

    # Sauvegarder le CSV
    final_df.to_csv(args.output, index=False, encoding="utf-8")
    logger.info(f"\n✓ CSV sauvegardé : {args.output}")
    logger.info(f"  → {len(final_df)} lignes, {final_df['num_characters'].sum():,} caractères totaux")

    # Générer le rapport de synthèse
    generate_summary(final_df, output_dir or ".")

    # Aperçu rapide
    print(f"\n--- Aperçu du CSV ({args.output}) ---")
    print(final_df[["image_id", "filename", "doc_category", "num_characters"]].head(10).to_string(index=False))
    print(f"\n{'='*60}")
    print(f"Prompt 1.2 terminé. Prochaine étape : Prompt 2.1 (compression JPEG)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
