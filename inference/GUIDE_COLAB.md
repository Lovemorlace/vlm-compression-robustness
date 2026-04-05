# Guide : Inférence VLM sur Google Colab

## Le workflow complet

```
VOTRE PC (WSL)                    GOOGLE COLAB (GPU T4 gratuit)
──────────────────                ─────────────────────────────
1. Préparer les images
2. Compresser (JPEG + Neural)
3. Uploader vers Google Drive  →  4. Lancer l'inférence VLM
                                  5. Sauvegarder les CSV
6. Télécharger les CSV         ←  
7. Importer dans PostgreSQL
8. Calculer les métriques
9. Lancer la plateforme
```

## Étape 1 — Préparer les images sur votre PC

```bash
# 1. Préparer le ground-truth
conda activate metrics
python scripts/prepare_ground_truth.py \
    --coco-json data/metadata/COCO/test.json \
    --output data/metadata/ground_truth.csv \
    --sample 100  # Commencer petit !

# 2. Initialiser la base
psql -U vlm_user -d vlm_compression -f platform/database/init_schema.sql

# 3. Importer le ground-truth
conda activate platform
python platform/database/import_ground_truth.py \
    --gt-csv data/metadata/ground_truth.csv \
    --images-dir data/raw \
    --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression"

# 4. Compresser JPEG (tourne en CPU, c'est rapide)
conda activate compression
python compression/jpeg_pipeline/compress_jpeg.py \
    --input-dir data/raw \
    --output-dir data/compressed/jpeg \
    --gt-csv data/metadata/ground_truth.csv \
    --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression" \
    --no-ssim  # Plus rapide sans GPU

# 5. Compresser Neural (tourne en CPU, plus lent mais faisable)
python compression/neural_pipeline/compress_neural.py \
    --input-dir data/raw \
    --output-dir data/compressed/neural \
    --gt-csv data/metadata/ground_truth.csv \
    --model cheng2020-anchor \
    --quality-levels 1 3 6 \
    --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression" \
    --no-ssim
```

## Étape 2 — Uploader vers Google Drive

Créer cette structure dans Google Drive :

```
Mon Drive/
└── vlm_projet/
    ├── images/
    │   ├── raw/              ← Copier data/raw/*.png
    │   ├── jpeg/
    │   │   ├── QF90/         ← Copier data/compressed/jpeg/QF90/*.jpg
    │   │   ├── QF70/
    │   │   ├── QF50/
    │   │   ├── QF30/
    │   │   └── QF10/
    │   └── neural/
    │       ├── q1/           ← Copier data/compressed/neural/q1/*.png
    │       ├── q3/
    │       └── q6/
    ├── ground_truth.csv      ← Copier data/metadata/ground_truth.csv
    └── results/              ← Créer ce dossier vide
```

**Astuce** : Avec 100 images × 9 conditions = ~900 fichiers. 
Créer un zip sur votre PC et le dézipper dans Colab est plus rapide :

```bash
# Sur votre PC
cd ~/projet
zip -r images_for_colab.zip data/raw data/compressed data/metadata/ground_truth.csv
# Uploader images_for_colab.zip sur Google Drive
```

## Étape 3 — Lancer les notebooks Colab

1. Uploader `colab_qwen2vl.ipynb` sur Google Colab
2. `Exécution > Modifier le type d'exécution > GPU T4`
3. Exécuter toutes les cellules
4. Le CSV `predictions_qwen2-vl.csv` est sauvegardé dans Drive
5. Répéter avec `colab_internvl2.ipynb`

**Temps estimé** (GPU T4, 100 images × 9 conditions = 900 inférences) :
- Qwen2-VL : ~3-5 secondes/image → ~1h
- InternVL2 : ~4-6 secondes/image → ~1h30

**⚠ Colab gratuit** : session de 12h max, peut couper aléatoirement.
Les sauvegardes intermédiaires (toutes les 50 images) protègent contre ça.

## Étape 4 — Récupérer et importer les résultats

```bash
# Télécharger les CSV depuis Google Drive vers ~/projet/inference/results/

# Importer dans PostgreSQL
conda activate platform
python platform/database/import_predictions.py \
    --csv inference/results/predictions_qwen2-vl.csv \
           inference/results/predictions_internvl2.csv \
    --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression"
```

## Étape 5 — Calculer les métriques

```bash
conda activate metrics
python metrics/compute_metrics.py \
    --db-url "postgresql://vlm_user:changeme@localhost:5432/vlm_compression" \
    --gt-csv data/metadata/ground_truth.csv \
    --output-dir metrics/reports
```

## Étape 6 — Lancer la plateforme

```bash
# Terminal 1 — Backend
conda activate platform
cd platform/backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Frontend
cd platform/frontend
npm run dev
```

Ouvrir http://localhost:5173

## FAQ

**Q : Combien d'images utiliser ?**
Commencer avec 50-100 images du test set. C'est suffisant pour valider le pipeline et avoir des résultats préliminaires. Augmenter ensuite.

**Q : Colab a coupé en plein milieu ?**
Le CSV partiel est sauvé dans Drive. Relancez le notebook — grâce à la structure de la boucle, vous pouvez facilement reprendre.

**Q : Le GPU T4 a 15 Go VRAM, c'est assez ?**
Oui, en 4-bit les deux modèles tiennent (~8 Go). Si OOM sur de grandes images, réduire `MAX_TILES` à 4 pour InternVL2.

**Q : Et si j'ai accès à Colab Pro ?**
Utiliser un GPU A100 ou L4 → inférence 3-4x plus rapide, et vous pouvez monter à FP16 (meilleure qualité).
