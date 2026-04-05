# generate_ground_truth.py
import os
import csv
import pytesseract
from PIL import Image
from tqdm import tqdm
import psycopg2

RAW_DIR = "/home/mor/projet/data/raw"
DB_URL  = "postgresql://vlm_user:changeme@localhost:5432/vlm_compression"

# Récupérer les images en base
conn = psycopg2.connect(DB_URL)
cur  = conn.cursor()
cur.execute("SELECT image_id, filename, original_path FROM images ORDER BY image_id")
images = cur.fetchall()
print(f"{len(images)} images à traiter")

updated = 0
for image_id, filename, original_path in tqdm(images):
    # Chemin absolu
    if not os.path.isabs(original_path):
        full_path = os.path.join("/home/mor/projet", original_path)
    else:
        full_path = original_path

    if not os.path.exists(full_path):
        print(f"⚠ Introuvable : {full_path}")
        continue

    # OCR
    try:
        img  = Image.open(full_path).convert("RGB")
        text = pytesseract.image_to_string(img, lang="eng").strip()
    except Exception as e:
        print(f"⚠ OCR échoué sur {filename}: {e}")
        text = ""

    if not text:
        continue

    num_chars = len(text)
    num_words = len(text.split())

    # Mettre à jour ground_truth
    cur.execute("""
        UPDATE ground_truth
        SET gt_text = %s, num_characters = %s, num_words = %s
        WHERE image_id = %s
    """, (text, num_chars, num_words, image_id))

    # Insérer si absent
    if cur.rowcount == 0:
        cur.execute("""
            INSERT INTO ground_truth (image_id, gt_text, num_characters, num_words)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """, (image_id, text, num_chars, num_words))

    updated += 1
    if updated % 10 == 0:
        conn.commit()

conn.commit()
cur.close()
conn.close()
print(f"\n✓ {updated} images avec GT texte généré")
