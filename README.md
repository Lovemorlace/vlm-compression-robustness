# Compression d'Images & Robustesse des VLMs

Plateforme d'évaluation de l'impact de la compression d'images (JPEG et neuronale) sur les performances de transcription des Vision Language Models (VLMs) appliqués à des documents.

---

## Structure du Projet

```
projet/
│
├── data/                              # Données DocLayNet
│   ├── raw/                           # Images PNG originales par catégorie
│   ├── compressed/
│   │   ├── jpeg/                      # QF = 90, 70, 50, 30, 10
│   │   └── neural/                    # Bitrates ≈ 0.1, 0.25, 0.5 bpp
│   └── metadata/                      # Annotations COCO JSON + texte GT
│
├── inference/                         # Scripts d'inférence VLM
│   ├── vlm_qwen/                      # Qwen2-VL-7B
│   ├── vlm_internvl/                  # InternVL2-8B
│   └── results/                       # Résultats JSON bruts
│
├── compression/                       # Pipelines de compression
│   ├── jpeg_pipeline/
│   └── neural_pipeline/
│
├── metrics/                           # Calcul CER, WER, BLEU, SSIM, LPIPS
│
├── platform/                          # Application web de visualisation
│   ├── backend/                       # FastAPI + SQLAlchemy
│   ├── frontend/                      # React + Recharts/Plotly
│   └── database/                      # Scripts SQL d'initialisation
│
├── envs/                              # Fichiers requirements par environnement
│   ├── requirements_compression.txt
│   ├── requirements_vlm_qwen.txt
│   ├── requirements_vlm_internvl.txt
│   ├── requirements_metrics.txt
│   └── requirements_platform.txt
│
├── scripts/                           # Scripts utilitaires (setup, export, etc.)
├── logs/                              # Logs d'exécution des pipelines
├── .env.example                       # Template de configuration
└── README.md                          # Ce fichier
```

---

## Prérequis Système

| Composant | Version requise |
|-----------|----------------|
| OS | Ubuntu 22.04 LTS |
| GPU | NVIDIA avec ≥ 16 Go VRAM (24 Go recommandé) |
| CUDA | 12.x |
| cuDNN | 8.9+ |
| Conda | Miniconda ou Anaconda |
| Node.js | 18+ (pour le frontend) |
| PostgreSQL | 15+ |

---

## Installation Pas à Pas

### Étape 0 — Vérifier les prérequis

```bash
# Vérifier CUDA
nvidia-smi

# Vérifier conda
conda --version

# Vérifier Node.js
node --version

# Vérifier PostgreSQL
psql --version
```

### Étape 1 — Cloner et configurer

```bash
# Se placer dans le répertoire du projet
cd ~/projet

# Copier et éditer le fichier de configuration
cp .env.example .env
nano .env    # Adapter POSTGRES_PASSWORD et les chemins si nécessaire
```

### Étape 2 — Installer PostgreSQL et créer la base

```bash
# Installer PostgreSQL (si pas déjà fait)
sudo apt update
sudo apt install -y postgresql postgresql-contrib

# Démarrer le service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Créer l'utilisateur et la base
sudo -u postgres psql -c "CREATE USER vlm_user WITH PASSWORD 'changeme_secure_password';"
sudo -u postgres psql -c "CREATE DATABASE vlm_compression OWNER vlm_user;"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE vlm_compression TO vlm_user;"
```

### Étape 3 — Créer les environnements Conda

Chaque composant du pipeline a son propre environnement pour éviter les conflits de dépendances.

```bash
# --- Environnement : compression ---
conda create -n compression python=3.10 -y
conda activate compression
pip install -r envs/requirements_compression.txt
conda deactivate

# --- Environnement : vlm_qwen ---
conda create -n vlm_qwen python=3.10 -y
conda activate vlm_qwen
pip install -r envs/requirements_vlm_qwen.txt
conda deactivate

# --- Environnement : vlm_internvl ---
conda create -n vlm_internvl python=3.10 -y
conda activate vlm_internvl
pip install -r envs/requirements_vlm_internvl.txt
conda deactivate

# --- Environnement : metrics ---
conda create -n metrics python=3.10 -y
conda activate metrics
pip install -r envs/requirements_metrics.txt
conda deactivate

# --- Environnement : platform ---
conda create -n platform python=3.10 -y
conda activate platform
pip install -r envs/requirements_platform.txt
conda deactivate
```

### Étape 4 — Télécharger DocLayNet

```bash
# Créer le dossier de destination
mkdir -p data/raw data/metadata

# Option A : via Hugging Face (recommandé)
# Installer git-lfs si nécessaire
sudo apt install -y git-lfs
git lfs install

# Cloner le dataset (attention : ~30 Go)
git clone https://huggingface.co/datasets/ds4sd/DocLayNet data/doclaynet_download

# Copier les images PNG dans data/raw/
# Copier les JSONs COCO dans data/metadata/

# Option B : téléchargement direct depuis IBM
# Voir : https://github.com/DS4SD/DocLayNet
```

### Étape 5 — Télécharger les modèles VLM

```bash
# Qwen2-VL-7B
conda activate vlm_qwen
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = 'Qwen/Qwen2-VL-7B-Instruct'
print('Téléchargement Qwen2-VL-7B...')
AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', trust_remote_code=True)
print('Terminé.')
"
conda deactivate

# InternVL2-8B
conda activate vlm_internvl
python -c "
from transformers import AutoModel, AutoTokenizer
model_name = 'OpenGVLab/InternVL2-8B'
print('Téléchargement InternVL2-8B...')
AutoModel.from_pretrained(model_name, device_map='auto', trust_remote_code=True)
print('Terminé.')
"
conda deactivate
```

### Étape 6 — Installer le frontend

```bash
cd platform/frontend
npm install
cd ../..
```

---

## Ordre d'Exécution du Pipeline

Suivre cet ordre strict — chaque étape dépend de la précédente.

```
 1. Préparer le ground-truth DocLayNet     →  scripts/prepare_ground_truth.py
 2. Initialiser la base de données          →  platform/database/init_schema.sql
 3. Compression JPEG batch                  →  compression/jpeg_pipeline/compress.py
 4. Compression neuronale batch             →  compression/neural_pipeline/compress.py
 5. Inférence Qwen2-VL                     →  inference/vlm_qwen/infer.py
 6. Inférence InternVL2                     →  inference/vlm_internvl/infer.py
 7. Calcul des métriques                    →  metrics/compute_metrics.py
 8. Lancer le backend                       →  platform/backend/
 9. Lancer le frontend                      →  platform/frontend/
```

---

## Lancer la Plateforme

```bash
# Terminal 1 — Backend
conda activate platform
cd platform/backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Frontend
cd platform/frontend
npm run dev
```

Ouvrir le navigateur à `http://localhost:5173`

---

## Schéma JSON Standardisé des Résultats

Chaque résultat d'inférence suit ce format avant insertion en base :

```json
{
  "image_id": "financial_0042",
  "vlm": "qwen2-vl",
  "compression_type": "jpeg",
  "compression_level": 50,
  "bitrate_bpp": 0.83,
  "file_size_kb": 124,
  "predicted_text": "...",
  "gt_text": "...",
  "scores": {
    "cer": 0.083,
    "wer": 0.121,
    "bleu": 0.71
  },
  "ssim": 0.92,
  "timestamp": "2025-03-15T14:30:00Z"
}
```

---

## Notes Importantes

- **Mémoire GPU** : Qwen2-VL-7B et InternVL2-8B nécessitent chacun ~16 Go VRAM. Lancer les inférences séquentiellement, jamais en parallèle. En cas de mémoire insuffisante, utiliser la quantization 4-bit via bitsandbytes.
- **Volume de données** : DocLayNet contient ~80k images. Pour les premiers tests, utiliser un sous-ensemble (ex. 500 images du test set). Le pipeline complet sera long (~24-48h selon le GPU).
- **Reproductibilité** : Tous les paramètres sont dans `.env`. Le prompt VLM est unique et identique pour les deux modèles. La température est à 0.0 pour des résultats déterministes.
- **Journal de bord** : Maintenir `logs/notes_resultats.md` au fil des expériences.
