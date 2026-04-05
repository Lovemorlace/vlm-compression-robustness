#!/bin/bash
# ============================================
# setup_envs.sh — Création automatique de tous les environnements conda
# Usage : bash scripts/setup_envs.sh
# ============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVS_DIR="$PROJECT_ROOT/envs"

echo "============================================"
echo " Projet Compression & VLMs — Setup"
echo " Racine : $PROJECT_ROOT"
echo "============================================"

# Liste des environnements
declare -A ENVS=(
    ["compression"]="requirements_compression.txt"
    ["vlm_qwen"]="requirements_vlm_qwen.txt"
    ["vlm_internvl"]="requirements_vlm_internvl.txt"
    ["metrics"]="requirements_metrics.txt"
    ["platform"]="requirements_platform.txt"
)

for env_name in "${!ENVS[@]}"; do
    req_file="${ENVS[$env_name]}"
    echo ""
    echo "--- Création de l'environnement : $env_name ---"

    # Vérifier si l'env existe déjà
    if conda info --envs | grep -q "^$env_name "; then
        echo "  ⚠ L'environnement '$env_name' existe déjà. Skipping."
        continue
    fi

    conda create -n "$env_name" python=3.10 -y
    echo "  ✓ Environnement créé"

    # Installer les dépendances
    eval "$(conda shell.bash hook)"
    conda activate "$env_name"
    pip install -r "$ENVS_DIR/$req_file"
    conda deactivate
    echo "  ✓ Dépendances installées depuis $req_file"
done

echo ""
echo "============================================"
echo " Tous les environnements sont prêts."
echo " Vérifier avec : conda info --envs"
echo "============================================"
