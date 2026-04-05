"""
=============================================================================
main.py — Point d'entrée FastAPI — Prompt 6.1
=============================================================================
Plateforme de visualisation : Compression d'Images & Robustesse des VLMs

Lancement :
    conda activate platform
    cd platform/backend
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Documentation API interactive :
    http://localhost:8000/docs      (Swagger UI)
    http://localhost:8000/redoc     (ReDoc)
=============================================================================
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import engine, Base
from routers import meta, results, charts, export


# ============================================================================
# Lifespan — création des tables au démarrage
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Crée les tables si elles n'existent pas au démarrage."""
    Base.metadata.create_all(bind=engine)
    yield


# ============================================================================
# Application FastAPI
# ============================================================================

app = FastAPI(
    title="VLM Compression Platform",
    description=(
        "API de la plateforme d'évaluation de l'impact de la compression d'images "
        "(JPEG et neuronale) sur les performances de transcription des Vision Language Models. "
        "Permet de filtrer, visualiser et comparer les résultats par VLM, "
        "condition de compression et catégorie documentaire."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ============================================================================
# CORS — Autoriser le frontend React (localhost:5173)
# ============================================================================

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Enregistrement des routers
# ============================================================================

app.include_router(meta.router)
app.include_router(results.router)
app.include_router(charts.router)
app.include_router(export.router)


# ============================================================================
# Root endpoint
# ============================================================================

@app.get("/", tags=["root"])
def root():
    return {
        "service": "VLM Compression Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "meta": {
                "GET /api/health": "Health check",
                "GET /api/stats": "Statistiques globales du projet",
                "GET /api/filters": "Options de filtres disponibles",
            },
            "results": {
                "GET /api/results": "Liste filtrée et paginée des résultats",
                "GET /api/results/{image_id}": "Tous les résultats pour une image",
                "GET /api/results/{image_id}/detail": "Résultat détaillé (1 VLM + 1 condition)",
                "GET /api/images/serve": "Servir un fichier image",
                "GET /api/images/list": "Liste des images avec résumé",
            },
            "charts": {
                "GET /api/charts/degradation": "Courbes de dégradation (score vs compression)",
                "GET /api/charts/heatmap": "Heatmap catégorie × condition",
                "GET /api/charts/iso-bitrate": "Comparaison JPEG vs Neural à iso-bitrate",
                "GET /api/charts/distribution": "Distribution des scores (histogramme)",
                "GET /api/charts/vlm-comparison": "Comparaison inter-VLMs",
            },
            "export": {
                "GET /api/export/csv": "Export CSV des résultats filtrés",
                "GET /api/export/csv/aggregated": "Export CSV agrégé par condition",
                "GET /api/export/csv/heatmap": "Export CSV format pivot (heatmap)",
                "GET /api/export/csv/iso-bitrate": "Export CSV comparaison iso-bitrate",
                "GET /api/export/report": "Rapport de synthèse PDF complet",
            },
        },
    }
