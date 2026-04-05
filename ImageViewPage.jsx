/**
 * ImageViewPage.jsx — Écran 2 : Vue par image (Prompt 7.2)
 *
 * Fonctionnalités :
 *   - Slider visuel original / compressé (avant/après)
 *   - Affichage texte GT vs texte prédit avec diff coloré
 *   - Scores CER / WER / BLEU
 *   - Navigation entre images
 *   - Sélection VLM et condition de compression
 */

import React, { useState, useEffect, useRef, useCallback } from "react";
import { useParams, useSearchParams, Link } from "react-router-dom";
import {
  fetchImageResults,
  fetchImageList,
  getImageUrl,
} from "../services/api";
import "./ImageViewPage.css";

// ============================================================================
// Composant principal
// ============================================================================

export default function ImageViewPage() {
  const { imageId } = useParams();
  const [searchParams] = useSearchParams();

  const [imageData, setImageData] = useState(null);
  const [imageList, setImageList] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Sélection active
  const [selectedVlm, setSelectedVlm] = useState(null);
  const [selectedCondition, setSelectedCondition] = useState(null);
  const [activeResult, setActiveResult] = useState(null);

  // Charger les données de l'image
  useEffect(() => {
    if (!imageId || imageId === "browse") return;

    setLoading(true);
    setError(null);

    fetchImageResults(parseInt(imageId))
      .then((data) => {
        setImageData(data);

        if (data.results && data.results.length > 0) {
          const vlms = [...new Set(data.results.map((r) => r.vlm_name))];
          const firstVlm = vlms[0];
          setSelectedVlm(firstVlm);

          const conditions = [
            ...new Set(
              data.results
                .filter((r) => r.vlm_name === firstVlm)
                .map((r) => r.condition_label)
            ),
          ];
          setSelectedCondition(conditions[0] || "baseline");

          const first = data.results.find(
            (r) =>
              r.vlm_name === firstVlm &&
              r.condition_label === (conditions[0] || "baseline")
          );
          setActiveResult(first || data.results[0]);
        }

        setLoading(false);
      })
      .catch(() => {
        setError("Image non trouvée ou erreur de chargement.");
        setLoading(false);
      });
  }, [imageId]);

  // Charger la liste des images pour navigation
  useEffect(() => {
    const category = searchParams.get("category");
    fetchImageList({ category, page_size: 200, has_results: true })
      .then((data) => setImageList(data.data || []))
      .catch(() => {});
  }, [searchParams]);

  // Mettre à jour le résultat actif quand VLM ou condition change
  useEffect(() => {
    if (!imageData || !selectedVlm || !selectedCondition) return;

    const match = imageData.results.find(
      (r) =>
        r.vlm_name === selectedVlm && r.condition_label === selectedCondition
    );
    setActiveResult(match || null);
  }, [selectedVlm, selectedCondition, imageData]);

  // Navigation
  const currentIndex = imageList.findIndex(
    (img) => img.image_id === parseInt(imageId)
  );
  const prevImage = currentIndex > 0 ? imageList[currentIndex - 1] : null;
  const nextImage =
    currentIndex < imageList.length - 1 ? imageList[currentIndex + 1] : null;

  // Listes dérivées
  const vlmList = imageData
    ? [...new Set(imageData.results.map((r) => r.vlm_name))]
    : [];
  const conditionList = imageData
    ? [
        ...new Set(
          imageData.results
            .filter((r) => r.vlm_name === selectedVlm)
            .map((r) => r.condition_label)
        ),
      ]
    : [];

  // Browse mode
  if (imageId === "browse") {
    return <BrowseView imageList={imageList} searchParams={searchParams} />;
  }

  if (loading) {
    return (
      <div className="iv-loading">
        <div className="iv-spinner" />
        <p>Chargement de l'image...</p>
      </div>
    );
  }

  if (error || !imageData) {
    return (
      <div className="iv-error">
        <h2>Erreur</h2>
        <p>{error || "Données non disponibles."}</p>
        <Link to="/" className="iv-back-link">← Retour</Link>
      </div>
    );
  }

  const img = imageData.image;
  const gt = imageData.ground_truth;

  return (
    <div className="iv-page">
      {/* Header */}
      <header className="iv-header">
        <div className="iv-header-left">
          <Link to="/" className="iv-back-link">← Sélection</Link>
          <div className="iv-header-info">
            <h1>{img.filename}</h1>
            <div className="iv-header-tags">
              <span className="iv-tag iv-tag-cat">{img.category}</span>
              <span className="iv-tag iv-tag-id">ID: {img.image_id}</span>
              <span className="iv-tag">{img.width}×{img.height}</span>
              {img.original_size_kb && (
                <span className="iv-tag">{img.original_size_kb.toFixed(0)} Ko</span>
              )}
            </div>
          </div>
        </div>
        <div className="iv-nav">
          {prevImage ? (
            <Link to={`/image/${prevImage.image_id}?${searchParams.toString()}`} className="iv-nav-btn">← Préc</Link>
          ) : (
            <span className="iv-nav-btn disabled">← Préc</span>
          )}
          <span className="iv-nav-counter">{currentIndex >= 0 ? currentIndex + 1 : "?"} / {imageList.length}</span>
          {nextImage ? (
            <Link to={`/image/${nextImage.image_id}?${searchParams.toString()}`} className="iv-nav-btn">Suiv →</Link>
          ) : (
            <span className="iv-nav-btn disabled">Suiv →</span>
          )}
        </div>
      </header>

      {/* Sélecteurs VLM + Condition */}
      <div className="iv-selectors">
        <div className="iv-selector-group">
          <label>VLM</label>
          <div className="iv-selector-btns">
            {vlmList.map((vlm) => (
              <button
                key={vlm}
                className={`iv-sel-btn ${selectedVlm === vlm ? "active" : ""}`}
                onClick={() => setSelectedVlm(vlm)}
              >
                {formatVlm(vlm)}
              </button>
            ))}
          </div>
        </div>
        <div className="iv-selector-group">
          <label>Condition</label>
          <div className="iv-selector-btns">
            {conditionList.map((cond) => (
              <button
                key={cond}
                className={`iv-sel-btn ${selectedCondition === cond ? "active" : ""} ${condColor(cond)}`}
                onClick={() => setSelectedCondition(cond)}
              >
                {cond}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Contenu principal */}
      {activeResult ? (
        <div className="iv-content">
          <div className="iv-row-top">
            <div className="iv-slider-panel">
              <ImageSlider
                originalPath={activeResult.original_path}
                compressedPath={activeResult.compressed_path}
                conditionLabel={activeResult.condition_label}
              />
              {activeResult.compression_type !== "baseline" && (
                <div className="iv-compression-info">
                  {activeResult.bitrate_bpp != null && <InfoChip label="Bitrate" value={`${activeResult.bitrate_bpp.toFixed(3)} bpp`} />}
                  {activeResult.compressed_size_kb != null && <InfoChip label="Taille" value={`${activeResult.compressed_size_kb.toFixed(1)} Ko`} />}
                  {activeResult.compression_ratio != null && <InfoChip label="Ratio" value={`×${activeResult.compression_ratio.toFixed(1)}`} />}
                  {activeResult.ssim != null && <InfoChip label="SSIM" value={activeResult.ssim.toFixed(4)} />}
                </div>
              )}
            </div>
            <div className="iv-scores-panel">
              <h3>Scores de Transcription</h3>
              <div className="iv-scores-grid">
                <ScoreCard label="CER" value={activeResult.cer} description="Character Error Rate" format="percent" inverted />
                <ScoreCard label="WER" value={activeResult.wer} description="Word Error Rate" format="percent" inverted />
                <ScoreCard label="BLEU" value={activeResult.bleu} description="Bilingual Evaluation" format="score" />
              </div>
              <div className="iv-scores-meta">
                {activeResult.inference_time_s != null && <span>Inférence : {activeResult.inference_time_s.toFixed(2)}s</span>}
                {activeResult.num_tokens_generated != null && <span>{activeResult.num_tokens_generated} tokens</span>}
              </div>
            </div>
          </div>
          <div className="iv-row-bottom">
            <TextDiffPanel gtText={gt?.gt_text || ""} predText={activeResult.predicted_text || ""} />
          </div>
        </div>
      ) : (
        <div className="iv-no-result">Aucun résultat pour cette combinaison VLM / condition.</div>
      )}
    </div>
  );
}

// ============================================================================
// ImageSlider
// ============================================================================

function ImageSlider({ originalPath, compressedPath, conditionLabel }) {
  const containerRef = useRef(null);
  const [sliderPos, setSliderPos] = useState(50);
  const [isDragging, setIsDragging] = useState(false);

  const isBaseline = conditionLabel === "baseline";
  const leftSrc = originalPath ? getImageUrl(originalPath) : null;
  const rightSrc = !isBaseline && compressedPath ? getImageUrl(compressedPath) : null;

  const handleMove = useCallback((clientX) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const percent = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));
    setSliderPos(percent);
  }, []);

  useEffect(() => {
    if (!isDragging) return;
    const onUp = () => setIsDragging(false);
    const onMove = (e) => handleMove(e.clientX);
    document.addEventListener("mouseup", onUp);
    document.addEventListener("mousemove", onMove);
    return () => {
      document.removeEventListener("mouseup", onUp);
      document.removeEventListener("mousemove", onMove);
    };
  }, [isDragging, handleMove]);

  if (!leftSrc) {
    return (
      <div className="iv-slider-empty">
        <p>Image non disponible</p>
        <span>Vérifiez que les chemins d'images sont accessibles</span>
      </div>
    );
  }

  if (isBaseline || !rightSrc) {
    return (
      <div className="iv-slider-container">
        <div className="iv-slider-label-solo">Image originale (PNG)</div>
        <img src={leftSrc} alt="Original" className="iv-slider-img-solo" />
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="iv-slider-container iv-slider-compare"
      onMouseDown={() => setIsDragging(true)}
      onTouchMove={(e) => handleMove(e.touches[0].clientX)}
    >
      <img src={rightSrc} alt="Compressé" className="iv-slider-img-full" />
      <div className="iv-slider-clip" style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}>
        <img src={leftSrc} alt="Original" className="iv-slider-img-full" />
      </div>
      <div className="iv-slider-bar" style={{ left: `${sliderPos}%` }}>
        <div className="iv-slider-handle"><span>⟨⟩</span></div>
      </div>
      <div className="iv-slider-labels">
        <span className="iv-slider-label-left">Original</span>
        <span className="iv-slider-label-right">{conditionLabel}</span>
      </div>
    </div>
  );
}

// ============================================================================
// ScoreCard
// ============================================================================

function ScoreCard({ label, value, description, format, inverted }) {
  if (value == null) {
    return (
      <div className="iv-score-card">
        <div className="iv-score-label">{label}</div>
        <div className="iv-score-value na">—</div>
        <div className="iv-score-desc">{description}</div>
      </div>
    );
  }

  const displayValue = format === "percent" ? `${(value * 100).toFixed(1)}%` : value.toFixed(3);

  let quality;
  if (inverted) {
    quality = value <= 0.05 ? "excellent" : value <= 0.15 ? "good" : value <= 0.3 ? "medium" : "poor";
  } else {
    quality = value >= 0.8 ? "excellent" : value >= 0.5 ? "good" : value >= 0.3 ? "medium" : "poor";
  }

  return (
    <div className={`iv-score-card quality-${quality}`}>
      <div className="iv-score-label">{label}</div>
      <div className="iv-score-value">{displayValue}</div>
      <div className="iv-score-bar-bg">
        <div
          className="iv-score-bar-fill"
          style={{ width: `${Math.min(100, (inverted ? 1 - value : value) * 100)}%` }}
        />
      </div>
      <div className="iv-score-desc">{description}</div>
    </div>
  );
}

// ============================================================================
// TextDiffPanel
// ============================================================================

function TextDiffPanel({ gtText, predText }) {
  const [viewMode, setViewMode] = useState("diff");
  const diffResult = computeDiff(gtText, predText);

  return (
    <div className="iv-diff-panel">
      <div className="iv-diff-header">
        <h3>Texte Ground-Truth vs Prédit</h3>
        <div className="iv-diff-modes">
          {["diff", "side", "gt", "pred"].map((mode) => (
            <button key={mode} className={viewMode === mode ? "active" : ""} onClick={() => setViewMode(mode)}>
              {{ diff: "Diff", side: "Côte à côte", gt: "GT seul", pred: "Prédit seul" }[mode]}
            </button>
          ))}
        </div>
      </div>

      <div className="iv-diff-legend">
        <span className="iv-legend-correct">■ Correct</span>
        <span className="iv-legend-deletion">■ Manquant</span>
        <span className="iv-legend-insertion">■ Ajouté</span>
      </div>

      <div className="iv-diff-body">
        {viewMode === "diff" && (
          <div className="iv-diff-content">
            {diffResult.map((part, i) => (
              <span key={i} className={`iv-diff-word ${part.type === "delete" ? "iv-diff-del" : part.type === "insert" ? "iv-diff-ins" : ""}`}>
                {part.text}{" "}
              </span>
            ))}
          </div>
        )}
        {viewMode === "side" && (
          <div className="iv-side-container">
            <div className="iv-side-col"><div className="iv-side-label">Ground-Truth</div><div className="iv-side-text">{gtText || "(vide)"}</div></div>
            <div className="iv-side-col"><div className="iv-side-label">Prédit</div><div className="iv-side-text">{predText || "(vide)"}</div></div>
          </div>
        )}
        {viewMode === "gt" && <div className="iv-single-text"><div className="iv-single-label">Ground-Truth</div><div className="iv-single-content">{gtText || "(vide)"}</div></div>}
        {viewMode === "pred" && <div className="iv-single-text"><div className="iv-single-label">Texte Prédit</div><div className="iv-single-content">{predText || "(vide)"}</div></div>}
      </div>

      <div className="iv-diff-stats">
        <span>GT : {gtText ? gtText.length : 0} chars, {gtText ? gtText.split(/\s+/).filter(Boolean).length : 0} mots</span>
        <span>Prédit : {predText ? predText.length : 0} chars, {predText ? predText.split(/\s+/).filter(Boolean).length : 0} mots</span>
      </div>
    </div>
  );
}

// ============================================================================
// BrowseView
// ============================================================================

function BrowseView({ imageList, searchParams }) {
  return (
    <div className="iv-page">
      <header className="iv-header">
        <div className="iv-header-left">
          <Link to="/" className="iv-back-link">← Sélection</Link>
          <h1>Parcourir les images ({imageList.length})</h1>
        </div>
      </header>
      <div className="iv-browse-grid">
        {imageList.length === 0 ? (
          <p className="iv-browse-empty">Aucune image avec des résultats en base.</p>
        ) : (
          imageList.map((img) => (
            <Link key={img.image_id} to={`/image/${img.image_id}?${searchParams.toString()}`} className="iv-browse-card">
              <div className="iv-browse-name">{img.filename}</div>
              <div className="iv-browse-meta">
                <span className="iv-tag iv-tag-cat">{img.category}</span>
                <span>{img.n_results} résultats</span>
              </div>
            </Link>
          ))
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Helpers
// ============================================================================

function InfoChip({ label, value }) {
  return <div className="iv-info-chip"><span className="iv-info-label">{label}</span><span className="iv-info-value">{value}</span></div>;
}

function formatVlm(vlm) {
  return { "qwen2-vl": "Qwen2-VL", internvl2: "InternVL2" }[vlm] || vlm;
}

function condColor(cond) {
  if (cond === "baseline") return "cond-baseline";
  if (cond.startsWith("jpeg")) return "cond-jpeg";
  if (cond.startsWith("neural")) return "cond-neural";
  return "";
}

function computeDiff(gt, pred) {
  if (!gt && !pred) return [];
  if (!gt) return [{ type: "insert", text: pred }];
  if (!pred) return [{ type: "delete", text: gt }];

  const gtWords = gt.split(/\s+/).filter(Boolean);
  const predWords = pred.split(/\s+/).filter(Boolean);
  const m = gtWords.length;
  const n = predWords.length;

  if (m > 500 || n > 500) return simpleDiff(gtWords, predWords);

  const dp = Array(m + 1).fill(null).map(() => Array(n + 1).fill(0));
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] = gtWords[i-1].toLowerCase() === predWords[j-1].toLowerCase()
        ? dp[i-1][j-1] + 1
        : Math.max(dp[i-1][j], dp[i][j-1]);
    }
  }

  const parts = [];
  let i = m, j = n;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && gtWords[i-1].toLowerCase() === predWords[j-1].toLowerCase()) {
      parts.unshift({ type: "equal", text: predWords[j-1] });
      i--; j--;
    } else if (j > 0 && (i === 0 || dp[i][j-1] >= dp[i-1][j])) {
      parts.unshift({ type: "insert", text: predWords[j-1] });
      j--;
    } else {
      parts.unshift({ type: "delete", text: gtWords[i-1] });
      i--;
    }
  }
  return parts;
}

function simpleDiff(gtWords, predWords) {
  const parts = [];
  const predSet = new Set(predWords.map((w) => w.toLowerCase()));
  const gtSet = new Set(gtWords.map((w) => w.toLowerCase()));
  for (const w of gtWords) parts.push({ type: predSet.has(w.toLowerCase()) ? "equal" : "delete", text: w });
  for (const w of predWords) if (!gtSet.has(w.toLowerCase())) parts.push({ type: "insert", text: w });
  return parts;
}
