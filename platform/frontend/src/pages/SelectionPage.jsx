/**
 * SelectionPage.jsx — Écran 1 : Sélection de l'expérience
 *
 * Permet de choisir :
 *   - Catégorie documentaire DocLayNet
 *   - VLM (Qwen2-VL, InternVL2, ou les deux)
 *   - Condition de compression (Baseline, JPEG QF, Neural quality)
 *
 * Les filtres sont chargés dynamiquement depuis l'API /api/filters
 * et /api/stats pour n'afficher que les options avec des résultats.
 */

import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { fetchStats, fetchFilters } from "../services/api";
import "./SelectionPage.css";

// ============================================================================
// Icônes SVG inline (pas de dépendance externe)
// ============================================================================

const IconImage = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
    <circle cx="8.5" cy="8.5" r="1.5" />
    <polyline points="21 15 16 10 5 21" />
  </svg>
);

const IconBrain = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 2a7 7 0 0 0-7 7c0 2.38 1.19 4.47 3 5.74V17a2 2 0 0 0 2 2h4a2 2 0 0 0 2-2v-2.26c1.81-1.27 3-3.36 3-5.74a7 7 0 0 0-7-7z" />
    <path d="M10 21v1a1 1 0 0 0 1 1h2a1 1 0 0 0 1-1v-1" />
  </svg>
);

const IconCompress = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="4 14 10 14 10 20" />
    <polyline points="20 10 14 10 14 4" />
    <line x1="14" y1="10" x2="21" y2="3" />
    <line x1="3" y1="21" x2="10" y2="14" />
  </svg>
);

const IconChart = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="18" y1="20" x2="18" y2="10" />
    <line x1="12" y1="20" x2="12" y2="4" />
    <line x1="6" y1="20" x2="6" y2="14" />
  </svg>
);

const IconArrow = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <line x1="5" y1="12" x2="19" y2="12" />
    <polyline points="12 5 19 12 12 19" />
  </svg>
);

const IconGrid = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="7" height="7" />
    <rect x="14" y="3" width="7" height="7" />
    <rect x="3" y="14" width="7" height="7" />
    <rect x="14" y="14" width="7" height="7" />
  </svg>
);


// ============================================================================
// Composant principal
// ============================================================================

export default function SelectionPage() {
  const navigate = useNavigate();

  // État
  const [stats, setStats] = useState(null);
  const [filters, setFilters] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Sélections utilisateur
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [selectedVlm, setSelectedVlm] = useState("both");
  const [selectedConditionType, setSelectedConditionType] = useState("baseline");
  const [selectedJpegLevel, setSelectedJpegLevel] = useState(null);
  const [selectedNeuralLevel, setSelectedNeuralLevel] = useState(null);

  // Chargement initial
  useEffect(() => {
    Promise.all([fetchStats(), fetchFilters()])
      .then(([statsData, filtersData]) => {
        setStats(statsData);
        setFilters(filtersData);

        // Pré-sélection par défaut
        if (statsData.jpeg_levels?.length > 0) {
          setSelectedJpegLevel(statsData.jpeg_levels[0]);
        }
        if (statsData.neural_levels?.length > 0) {
          setSelectedNeuralLevel(statsData.neural_levels[0]);
        }

        setLoading(false);
      })
      .catch((err) => {
        setError(
          "Impossible de se connecter au backend. Vérifiez que le serveur FastAPI tourne sur le port 8000."
        );
        setLoading(false);
      });
  }, []);

  // Navigation
  const handleViewResults = () => {
    const params = new URLSearchParams();

    if (selectedCategory !== "all") {
      params.set("category", selectedCategory);
    }
    if (selectedVlm !== "both") {
      params.set("vlm_name", selectedVlm);
    }

    params.set("compression_type", selectedConditionType);

    if (selectedConditionType === "jpeg" && selectedJpegLevel !== null) {
      params.set("compression_level", selectedJpegLevel);
    }
    if (selectedConditionType === "neural" && selectedNeuralLevel !== null) {
      params.set("compression_level", selectedNeuralLevel);
    }

    navigate(`/analysis?${params.toString()}`);
  };

  const handleViewImages = () => {
    const params = new URLSearchParams();
    if (selectedCategory !== "all") {
      params.set("category", selectedCategory);
    }
    navigate(`/image/browse?${params.toString()}`);
  };

  // ============================================================================
  // Rendu
  // ============================================================================

  if (loading) {
    return (
      <div className="sel-loading">
        <div className="sel-loading-spinner" />
        <p>Connexion à la plateforme...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="sel-error">
        <div className="sel-error-icon">!</div>
        <h2>Erreur de connexion</h2>
        <p>{error}</p>
        <button onClick={() => window.location.reload()} className="sel-btn-retry">
          Réessayer
        </button>
      </div>
    );
  }

  return (
    <div className="sel-page">
      {/* ---------------------------------------------------------------- */}
      {/* Header */}
      {/* ---------------------------------------------------------------- */}
      <header className="sel-header">
        <div className="sel-header-content">
          <div className="sel-header-title">
            <div className="sel-logo">
              <span className="sel-logo-icon">◆</span>
              <span>VLM × Compression</span>
            </div>
            <h1>Plateforme d'Évaluation</h1>
            <p className="sel-subtitle">
              Impact de la compression d'images sur la transcription des Vision Language Models
            </p>
          </div>

          {/* Stats rapides */}
          {stats && (
            <div className="sel-stats-bar">
              <StatBadge label="Images" value={stats.total_images} />
              <StatBadge label="Prédictions" value={stats.total_predictions} />
              <StatBadge label="Métriques" value={stats.total_metrics} />
              <StatBadge
                label="VLMs"
                value={stats.vlm_names?.length || 0}
              />
            </div>
          )}
        </div>
      </header>

      {/* ---------------------------------------------------------------- */}
      {/* Corps — Filtres */}
      {/* ---------------------------------------------------------------- */}
      <main className="sel-main">
        <div className="sel-filters-grid">
          {/* ---- Section 1 : Catégorie ---- */}
          <section className="sel-section">
            <div className="sel-section-header">
              <IconImage />
              <h2>Catégorie Documentaire</h2>
            </div>
            <p className="sel-section-desc">
              DocLayNet — Sélectionnez le type de document à analyser
            </p>
            <div className="sel-options-grid sel-options-categories">
              <OptionCard
                label="Toutes"
                description="Toutes les catégories"
                selected={selectedCategory === "all"}
                onClick={() => setSelectedCategory("all")}
                count={stats?.total_images}
              />
              {filters?.categories?.map((cat) => {
                const cond = filters.compression_conditions?.find(
                  (c) => c.type === "baseline"
                );
                return (
                  <OptionCard
                    key={cat}
                    label={cat}
                    selected={selectedCategory === cat}
                    onClick={() => setSelectedCategory(cat)}
                  />
                );
              })}
            </div>
          </section>

          {/* ---- Section 2 : VLM ---- */}
          <section className="sel-section">
            <div className="sel-section-header">
              <IconBrain />
              <h2>Vision Language Model</h2>
            </div>
            <p className="sel-section-desc">
              Choisissez le modèle à évaluer ou comparez les deux
            </p>
            <div className="sel-options-grid sel-options-vlm">
              <OptionCard
                label="Les deux"
                description="Comparaison inter-VLMs"
                selected={selectedVlm === "both"}
                onClick={() => setSelectedVlm("both")}
                accent="compare"
              />
              {filters?.vlm_names?.map((vlm) => (
                <OptionCard
                  key={vlm}
                  label={formatVlmName(vlm)}
                  tag={vlm}
                  description={vlmDescriptions[vlm] || ""}
                  selected={selectedVlm === vlm}
                  onClick={() => setSelectedVlm(vlm)}
                />
              ))}
            </div>
          </section>

          {/* ---- Section 3 : Condition de compression ---- */}
          <section className="sel-section">
            <div className="sel-section-header">
              <IconCompress />
              <h2>Condition de Compression</h2>
            </div>
            <p className="sel-section-desc">
              Sélectionnez le type et le niveau de compression
            </p>

            {/* Sélection du type */}
            <div className="sel-compression-types">
              <CompTypeButton
                label="Baseline"
                sublabel="PNG original"
                active={selectedConditionType === "baseline"}
                onClick={() => setSelectedConditionType("baseline")}
                color="baseline"
              />
              <CompTypeButton
                label="JPEG"
                sublabel="DCT lossy"
                active={selectedConditionType === "jpeg"}
                onClick={() => setSelectedConditionType("jpeg")}
                color="jpeg"
              />
              <CompTypeButton
                label="Neural"
                sublabel="CompressAI"
                active={selectedConditionType === "neural"}
                onClick={() => setSelectedConditionType("neural")}
                color="neural"
              />
            </div>

            {/* Niveaux JPEG */}
            {selectedConditionType === "jpeg" && stats?.jpeg_levels?.length > 0 && (
              <div className="sel-levels-section">
                <h3 className="sel-levels-title">Facteur de Qualité JPEG</h3>
                <div className="sel-levels-row">
                  {stats.jpeg_levels
                    .sort((a, b) => b - a)
                    .map((qf) => (
                      <LevelChip
                        key={qf}
                        label={`QF ${qf}`}
                        sublabel={jpegQualityLabel(qf)}
                        selected={selectedJpegLevel === qf}
                        onClick={() => setSelectedJpegLevel(qf)}
                        color="jpeg"
                      />
                    ))}
                </div>
                <div className="sel-levels-hint">
                  QF élevé = meilleure qualité, fichier plus gros. QF bas = compression agressive.
                </div>
              </div>
            )}

            {/* Niveaux Neural */}
            {selectedConditionType === "neural" && stats?.neural_levels?.length > 0 && (
              <div className="sel-levels-section">
                <h3 className="sel-levels-title">Niveau de Qualité Neural</h3>
                <div className="sel-levels-row">
                  {stats.neural_levels.map((ql) => (
                    <LevelChip
                      key={ql}
                      label={`q${ql}`}
                      sublabel={neuralQualityLabel(ql)}
                      selected={selectedNeuralLevel === ql}
                      onClick={() => setSelectedNeuralLevel(ql)}
                      color="neural"
                    />
                  ))}
                </div>
                <div className="sel-levels-hint">
                  Niveau bas = compression agressive (~0.1 bpp). Niveau haut = haute fidélité (~0.5 bpp).
                </div>
              </div>
            )}

            {/* Baseline info */}
            {selectedConditionType === "baseline" && (
              <div className="sel-baseline-info">
                Images PNG originales sans aucune compression — plafond de performance des VLMs.
              </div>
            )}
          </section>
        </div>

        {/* ---------------------------------------------------------------- */}
        {/* Résumé de sélection + actions */}
        {/* ---------------------------------------------------------------- */}
        <div className="sel-actions-panel">
          <div className="sel-selection-summary">
            <h3>Sélection</h3>
            <div className="sel-summary-tags">
              <SummaryTag
                label="Catégorie"
                value={selectedCategory === "all" ? "Toutes" : selectedCategory}
              />
              <SummaryTag
                label="VLM"
                value={
                  selectedVlm === "both"
                    ? "Comparaison"
                    : formatVlmName(selectedVlm)
                }
              />
              <SummaryTag
                label="Compression"
                value={formatConditionSummary(
                  selectedConditionType,
                  selectedJpegLevel,
                  selectedNeuralLevel
                )}
              />
            </div>
          </div>

          <div className="sel-actions-buttons">
            <button className="sel-btn sel-btn-secondary" onClick={handleViewImages}>
              <IconGrid />
              <span>Parcourir les images</span>
            </button>
            <button className="sel-btn sel-btn-primary" onClick={handleViewResults}>
              <IconChart />
              <span>Voir les résultats</span>
              <IconArrow />
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}


// ============================================================================
// Sous-composants
// ============================================================================

function StatBadge({ label, value }) {
  return (
    <div className="sel-stat-badge">
      <span className="sel-stat-value">
        {typeof value === "number" ? value.toLocaleString("fr-FR") : value}
      </span>
      <span className="sel-stat-label">{label}</span>
    </div>
  );
}

function OptionCard({ label, description, tag, selected, onClick, count, accent }) {
  return (
    <button
      className={`sel-option-card ${selected ? "selected" : ""} ${accent ? `accent-${accent}` : ""}`}
      onClick={onClick}
    >
      <span className="sel-option-label">{label}</span>
      {description && <span className="sel-option-desc">{description}</span>}
      {tag && <span className="sel-option-tag">{tag}</span>}
      {count != null && <span className="sel-option-count">{count.toLocaleString("fr-FR")}</span>}
      <span className="sel-option-check">{selected ? "●" : "○"}</span>
    </button>
  );
}

function CompTypeButton({ label, sublabel, active, onClick, color }) {
  return (
    <button
      className={`sel-comptype-btn ${active ? "active" : ""} color-${color}`}
      onClick={onClick}
    >
      <span className="sel-comptype-label">{label}</span>
      <span className="sel-comptype-sub">{sublabel}</span>
    </button>
  );
}

function LevelChip({ label, sublabel, selected, onClick, color }) {
  return (
    <button
      className={`sel-level-chip ${selected ? "selected" : ""} color-${color}`}
      onClick={onClick}
    >
      <span className="sel-level-label">{label}</span>
      {sublabel && <span className="sel-level-sub">{sublabel}</span>}
    </button>
  );
}

function SummaryTag({ label, value }) {
  return (
    <div className="sel-summary-tag">
      <span className="sel-summary-tag-label">{label}</span>
      <span className="sel-summary-tag-value">{value}</span>
    </div>
  );
}


// ============================================================================
// Helpers
// ============================================================================

const vlmDescriptions = {
  "qwen2-vl": "Qwen2-VL-7B — Fort sur OCR document",
  internvl2: "InternVL2-8B — Architecture vision robuste",
};

function formatVlmName(vlm) {
  const names = {
    "qwen2-vl": "Qwen2-VL",
    internvl2: "InternVL2",
  };
  return names[vlm] || vlm;
}

function jpegQualityLabel(qf) {
  if (qf >= 90) return "Très haute";
  if (qf >= 70) return "Haute";
  if (qf >= 50) return "Moyenne";
  if (qf >= 30) return "Basse";
  return "Très basse";
}

function neuralQualityLabel(ql) {
  if (ql <= 1) return "~0.1 bpp";
  if (ql <= 3) return "~0.25 bpp";
  if (ql <= 6) return "~0.5 bpp";
  return `q${ql}`;
}

function formatConditionSummary(type, jpegLevel, neuralLevel) {
  if (type === "baseline") return "Baseline (PNG)";
  if (type === "jpeg") return `JPEG QF=${jpegLevel}`;
  if (type === "neural") return `Neural q${neuralLevel}`;
  return type;
}
