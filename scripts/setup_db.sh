#!/bin/bash
# ============================================
# setup_db.sh — Initialisation de la base PostgreSQL
# Usage : bash scripts/setup_db.sh
# ============================================

set -e

# Charger la configuration
source .env

echo "Création de la base de données..."
sudo -u postgres psql <<EOF
-- Créer l'utilisateur (ignore si existe)
DO \$\$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '${POSTGRES_USER}') THEN
        CREATE ROLE ${POSTGRES_USER} WITH LOGIN PASSWORD '${POSTGRES_PASSWORD}';
    END IF;
END
\$\$;

-- Créer la base (ignore si existe)
SELECT 'CREATE DATABASE ${POSTGRES_DB} OWNER ${POSTGRES_USER}'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${POSTGRES_DB}')\gexec

GRANT ALL PRIVILEGES ON DATABASE ${POSTGRES_DB} TO ${POSTGRES_USER};
EOF

echo "✓ Base '${POSTGRES_DB}' prête pour l'utilisateur '${POSTGRES_USER}'"
echo ""
echo "Prochaine étape : exécuter le schéma SQL (Phase 5)"
echo "  psql -U ${POSTGRES_USER} -d ${POSTGRES_DB} -f platform/database/init_schema.sql"
