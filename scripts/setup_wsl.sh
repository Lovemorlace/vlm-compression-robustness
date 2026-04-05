#!/bin/bash
# =============================================================================
# setup_wsl.sh — Installation complète sur WSL (sans GPU)
# =============================================================================
# Usage :
#   bash scripts/setup_wsl.sh
#
# Ce script installe :
#   1. Mises à jour système
#   2. PostgreSQL 15
#   3. Miniconda (Python 3.10)
#   4. Node.js 20 (via nvm)
#   5. Dépendances système
# =============================================================================

set -e

echo "============================================"
echo " Installation WSL — Projet VLM Compression"
echo "============================================"
echo ""

# ------------------------------------------------------------------
# 1. Mises à jour système + dépendances de base
# ------------------------------------------------------------------
echo "--- [1/5] Mises à jour système ---"
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    build-essential \
    curl \
    wget \
    git \
    git-lfs \
    unzip \
    software-properties-common \
    libpq-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

echo "  ✓ Système mis à jour"

# ------------------------------------------------------------------
# 2. PostgreSQL 15
# ------------------------------------------------------------------
echo ""
echo "--- [2/5] Installation PostgreSQL ---"

if command -v psql &> /dev/null; then
    echo "  PostgreSQL déjà installé : $(psql --version)"
else
    sudo apt install -y postgresql postgresql-contrib
    echo "  ✓ PostgreSQL installé"
fi

# Démarrer le service (WSL n'utilise pas systemd par défaut)
sudo service postgresql start
echo "  ✓ Service PostgreSQL démarré"

# Créer l'utilisateur et la base
sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname='vlm_user'" | grep -q 1 || \
    sudo -u postgres psql -c "CREATE USER vlm_user WITH PASSWORD 'changeme_secure_password';"

sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname='vlm_compression'" | grep -q 1 || \
    sudo -u postgres psql -c "CREATE DATABASE vlm_compression OWNER vlm_user;"

sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE vlm_compression TO vlm_user;" 2>/dev/null || true

echo "  ✓ Base 'vlm_compression' prête (user: vlm_user)"

# ------------------------------------------------------------------
# 3. Miniconda
# ------------------------------------------------------------------
echo ""
echo "--- [3/5] Installation Miniconda ---"

if command -v conda &> /dev/null; then
    echo "  Conda déjà installé : $(conda --version)"
else
    CONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
    wget -q "https://repo.anaconda.com/miniconda/${CONDA_INSTALLER}" -O /tmp/${CONDA_INSTALLER}
    bash /tmp/${CONDA_INSTALLER} -b -p "$HOME/miniconda3"
    rm /tmp/${CONDA_INSTALLER}

    # Ajouter au PATH
    echo '' >> ~/.bashrc
    echo '# >>> conda initialize >>>' >> ~/.bashrc
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
    echo '# <<< conda initialize <<<' >> ~/.bashrc

    # Aussi pour zsh si présent
    if [ -f ~/.zshrc ]; then
        echo '' >> ~/.zshrc
        echo '# >>> conda initialize >>>' >> ~/.zshrc
        echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.zshrc
        echo '# <<< conda initialize <<<' >> ~/.zshrc
    fi

    # Activer dans la session courante
    export PATH="$HOME/miniconda3/bin:$PATH"

    # Initialiser conda pour bash et zsh
    conda init bash
    if [ -f ~/.zshrc ]; then
        conda init zsh
    fi

    echo "  ✓ Miniconda installé dans ~/miniconda3"
fi

echo "  Conda version : $(conda --version 2>/dev/null || echo 'rechargez le terminal')"

# ------------------------------------------------------------------
# 4. Node.js 20 via nvm
# ------------------------------------------------------------------
echo ""
echo "--- [4/5] Installation Node.js ---"

if command -v node &> /dev/null; then
    echo "  Node.js déjà installé : $(node --version)"
else
    # Installer nvm
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

    # Charger nvm dans la session courante
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

    # Installer Node.js 20 LTS
    nvm install 20
    nvm use 20
    nvm alias default 20

    echo "  ✓ Node.js $(node --version) installé via nvm"
fi

# ------------------------------------------------------------------
# 5. Vérification finale
# ------------------------------------------------------------------
echo ""
echo "============================================"
echo " VÉRIFICATION FINALE"
echo "============================================"

echo ""
echo -n "  PostgreSQL : "
psql --version 2>/dev/null || echo "❌ NON TROUVÉ"

echo -n "  Conda      : "
conda --version 2>/dev/null || echo "❌ NON TROUVÉ (fermer et rouvrir le terminal)"

echo -n "  Python     : "
python3 --version 2>/dev/null || echo "❌ NON TROUVÉ"

echo -n "  Node.js    : "
node --version 2>/dev/null || echo "❌ NON TROUVÉ (fermer et rouvrir le terminal)"

echo -n "  npm        : "
npm --version 2>/dev/null || echo "❌ NON TROUVÉ"

echo -n "  git        : "
git --version 2>/dev/null || echo "❌ NON TROUVÉ"

echo -n "  GPU NVIDIA : "
nvidia-smi &>/dev/null && nvidia-smi --query-gpu=name --format=csv,noheader || echo "Aucun (mode CPU uniquement)"

echo ""
echo "============================================"
echo " INSTALLATION TERMINÉE"
echo "============================================"
echo ""
echo " ⚠  IMPORTANT : Fermez et rouvrez votre terminal"
echo "    (ou tapez : source ~/.bashrc)"
echo "    pour que conda et node soient disponibles."
echo ""
echo " Prochaines étapes :"
echo "   1. source ~/.bashrc   (ou relancer le terminal)"
echo "   2. cd ~/projet"
echo "   3. conda --version    (vérifier)"
echo "   4. node --version     (vérifier)"
echo "   5. bash scripts/setup_envs.sh  (créer les envs conda)"
echo ""
echo " Note : Sans GPU, l'inférence VLM sera lente."
echo " Options :"
echo "   - Travailler sur un petit subset (50-100 images)"
echo "   - Utiliser Google Colab / un serveur cloud pour l'inférence"
echo "   - Les parties compression + plateforme tournent bien en CPU"
echo "============================================"
