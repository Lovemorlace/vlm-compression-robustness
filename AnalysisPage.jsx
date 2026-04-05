/**
 * AnalysisPage.jsx — Écran 3 : Vue agrégée (Prompt 7.3)
 *
 * Graphiques :
 *   1. Courbes de dégradation (score vs niveau de compression)
 *   2. Comparaison JPEG vs Neural à iso-bitrate
 *   3. Heatmap catégorie × condition
 *   4. Comparaison inter-VLMs
 *
 * Utilise Recharts pour tous les graphiques.
 */

import React, { useState, useEffect } from "react";
import { useSearchParams, Link } from "react-router-dom";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter,
  Cell, ReferenceLine,
} from "recharts";
import {
  fetchDegradation,
  fetchHeatmap,
  fetchIsoBitrate,
  fetchVlmComparison,
  fetchFilters,
  getExportCsvUrl,
  getExportAggregatedUrl,
  getExportReportUrl,
} from "../services/api";
import "./AnalysisPage.css";

// ============================================================================
// Couleurs
// ============================================================================
const COLORS = {
  "qwen2-vl": "#3d7aed",
  internvl2: "#e8853d",
  jpeg: "#e8853d",
  neural: "#9b6dff",
  baseline: "#34c77b",
};

const VLM_COLORS = ["#3d7aed", "#e8853d", "#34c77b", "#9b6dff"];

const HEATMAP_SCALE = [
  { min: 0, max: 0.2, bg: "#3b1219", text: "#f87171" },
  { min: 0.2, max: 0.4, bg: "#3b2012", text: "#fb923c" },
  { min: 0.4, max: 0.6, bg: "#3b3312", text: "#facc15" },
  { min: 0.6, max: 0.8, bg: "#1a3328", text: "#4ade80" },
  { min: 0.8, max: 1.01, bg: "#0f3d2c", text: "#34d399" },
];

function getHeatmapColor(value) {
  if (value == null) return { bg: "#161d27", text: "#556677" };
  for (const s of HEATMAP_SCALE) {
    if (value >= s.min && value < s.max) return s;
  }
  return HEATMAP_SCALE[HEATMAP_SCALE.length - 1];
}

// ============================================================================
// Page principale
// ============================================================================

export default function AnalysisPage() {
  const [searchParams] = useSearchParams();

  // Filtres depuis l'URL (passés par SelectionPage)
  const urlCategory = searchParams.get("category") || null;
  const urlVlm = searchParams.get("vlm_name") || null;
  const urlCompType = searchParams.get("compression_type") || null;

  // État local des filtres
  const [metric, setMetric] = useState("wer");
  const [category, setCategory] = useState(urlCategory);
  const [vlmFilter, setVlmFilter] = useState(urlVlm);
  const [filterOptions, setFilterOptions] = useState(null);

  // Données des graphiques
  const [degradData, setDegradData] = useState(null);
  const [heatmapData, setHeatmapData] = useState(null);
  const [isoData, setIsoData] = useState(null);
  const [vlmCompData, setVlmCompData] = useState(null);

  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState("degradation");

  // Charger les filtres
  useEffect(() => {
    fetchFilters()
      .then(setFilterOptions)
      .catch(() => {});
  }, []);

  // Charger les données quand les filtres changent
  useEffect(() => {
    setLoading(true);

    const params = {};
    if (category) params.category = category;
    if (vlmFilter) params.vlm_name = vlmFilter;

    Promise.all([
      fetchDegradation({ metric_name: metric, ...params }).catch(() => null),
      fetchHeatmap({ metric_name: metric === "wer" ? "bleu" : metric, vlm_name: vlmFilter }).catch(() => null),
      fetchIsoBitrate({ metric_name: metric, ...params }).catch(() => null),
      fetchVlmComparison({ metric_name: metric, category }).catch(() => null),
    ]).then(([degrad, heatmap, iso, vlmComp]) => {
      setDegradData(degrad);
      setHeatmapData(heatmap);
      setIsoData(iso);
      setVlmCompData(vlmComp);
      setLoading(false);
    });
  }, [metric, category, vlmFilter]);

  // ============================================================================
  // Rendu
  // ============================================================================

  return (
    <div className="an-page">
      {/* Header */}
      <header className="an-header">
        <div className="an-header-left">
          <Link to="/" className="an-back-link">← Sélection</Link>
          <div>
            <h1>Analyse Agrégée</h1>
            <p className="an-subtitle">Courbes, heatmaps et comparaisons</p>
          </div>
        </div>

        {/* Export */}
        <div className="an-export-btns">
          <a
            href={getExportAggregatedUrl({ vlm_name: vlmFilter, category })}
            className="an-export-btn"
            target="_blank"
            rel="noopener"
          >
            Export CSV
          </a>
          <a
            href={getExportReportUrl({ vlm_name: vlmFilter, category })}
            className="an-export-btn an-export-pdf"
            target="_blank"
            rel="noopener"
          >
            Rapport PDF
          </a>
        </div>
      </header>

      {/* Barre de filtres */}
      <div className="an-filters">
        <div className="an-filter-group">
          <label>Métrique</label>
          <div className="an-filter-btns">
            {["cer", "wer", "bleu"].map((m) => (
              <button
                key={m}
                className={`an-fbtn ${metric === m ? "active" : ""}`}
                onClick={() => setMetric(m)}
              >
                {m.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        <div className="an-filter-group">
          <label>Catégorie</label>
          <div className="an-filter-btns">
            <button
              className={`an-fbtn ${!category ? "active" : ""}`}
              onClick={() => setCategory(null)}
            >
              Toutes
            </button>
            {filterOptions?.categories?.map((cat) => (
              <button
                key={cat}
                className={`an-fbtn ${category === cat ? "active" : ""}`}
                onClick={() => setCategory(cat)}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>

        <div className="an-filter-group">
          <label>VLM</label>
          <div className="an-filter-btns">
            <button
              className={`an-fbtn ${!vlmFilter ? "active" : ""}`}
              onClick={() => setVlmFilter(null)}
            >
              Tous
            </button>
            {filterOptions?.vlm_names?.map((vlm) => (
              <button
                key={vlm}
                className={`an-fbtn ${vlmFilter === vlm ? "active" : ""}`}
                onClick={() => setVlmFilter(vlm)}
              >
                {formatVlm(vlm)}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Onglets */}
      <div className="an-tabs">
        {[
          { id: "degradation", label: "Courbes de Dégradation" },
          { id: "iso-bitrate", label: "JPEG vs Neural" },
          { id: "heatmap", label: "Heatmap" },
          { id: "vlm-compare", label: "Comparaison VLMs" },
        ].map((tab) => (
          <button
            key={tab.id}
            className={`an-tab ${activeTab === tab.id ? "active" : ""}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Contenu */}
      <div className="an-content">
        {loading ? (
          <div className="an-loading">
            <div className="an-spinner" />
            <p>Chargement des données...</p>
          </div>
        ) : (
          <>
            {activeTab === "degradation" && (
              <DegradationChart data={degradData} metric={metric} />
            )}
            {activeTab === "iso-bitrate" && (
              <IsoBitrateChart data={isoData} metric={metric} />
            )}
            {activeTab === "heatmap" && (
              <HeatmapChart data={heatmapData} />
            )}
            {activeTab === "vlm-compare" && (
              <VlmComparisonChart data={vlmCompData} metric={metric} />
            )}
          </>
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Graphique 1 — Courbes de dégradation
// ============================================================================

function DegradationChart({ data, metric }) {
  if (!data || !data.series || Object.keys(data.series).length === 0) {
    return <EmptyChart message="Pas de données de dégradation disponibles." />;
  }

  // Construire les données pour Recharts
  // On veut un point par condition, une ligne par VLM
  const conditionOrder = buildConditionOrder(data.series);
  const vlmNames = Object.keys(data.series);

  const chartData = conditionOrder.map((cond) => {
    const point = { condition: cond };
    for (const vlm of vlmNames) {
      const entry = data.series[vlm]?.find((e) => e.condition_label === cond);
      if (entry) {
        point[vlm] = entry.mean;
        point[`${vlm}_std`] = entry.std;
        point[`${vlm}_n`] = entry.n_images;
      }
    }
    return point;
  });

  const isInverted = metric === "cer" || metric === "wer";
  const yLabel = metric.toUpperCase() + (isInverted ? " (↓ mieux)" : " (↑ mieux)");

  return (
    <div className="an-chart-container">
      <div className="an-chart-title">
        <h2>Courbe de Dégradation — {metric.toUpperCase()} vs Compression</h2>
        <p>À partir de quel niveau de compression le VLM décroche ?</p>
      </div>

      <div className="an-chart-wrapper">
        <ResponsiveContainer width="100%" height={420}>
          <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" />
            <XAxis
              dataKey="condition"
              tick={{ fill: "#8899aa", fontSize: 11 }}
              angle={-30}
              textAnchor="end"
              height={70}
              stroke="#2a3a50"
            />
            <YAxis
              tick={{ fill: "#8899aa", fontSize: 11 }}
              stroke="#2a3a50"
              label={{ value: yLabel, angle: -90, position: "insideLeft", fill: "#8899aa", fontSize: 12 }}
              domain={isInverted ? [0, "auto"] : [0, 1]}
            />
            <Tooltip
              contentStyle={{ background: "#161d27", border: "1px solid #2a3a50", borderRadius: 8, color: "#e8ecf1" }}
              formatter={(value, name) => [
                typeof value === "number" ? value.toFixed(4) : value,
                formatVlm(name),
              ]}
            />
            <Legend
              formatter={(value) => formatVlm(value)}
              wrapperStyle={{ color: "#8899aa", fontSize: 12, paddingTop: 8 }}
            />
            {vlmNames.map((vlm, i) => (
              <Line
                key={vlm}
                type="monotone"
                dataKey={vlm}
                stroke={VLM_COLORS[i % VLM_COLORS.length]}
                strokeWidth={2.5}
                dot={{ r: 5, fill: VLM_COLORS[i % VLM_COLORS.length] }}
                activeDot={{ r: 7 }}
                connectNulls
              />
            ))}
            <ReferenceLine
              x="baseline"
              stroke="#34c77b"
              strokeDasharray="5 5"
              label={{ value: "Baseline", fill: "#34c77b", fontSize: 10, position: "top" }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Tableau numérique */}
      <DataTable
        headers={["Condition", ...vlmNames.map(formatVlm)]}
        rows={chartData.map((row) => [
          row.condition,
          ...vlmNames.map((vlm) =>
            row[vlm] != null ? row[vlm].toFixed(4) : "—"
          ),
        ])}
      />
    </div>
  );
}

// ============================================================================
// Graphique 2 — JPEG vs Neural à iso-bitrate
// ============================================================================

function IsoBitrateChart({ data, metric }) {
  if (!data || !data.series || Object.keys(data.series).length === 0) {
    return <EmptyChart message="Pas de données iso-bitrate disponibles." />;
  }

  const vlmNames = Object.keys(data.series);
  const isInverted = metric === "cer" || metric === "wer";

  // Construire les données pour le bar chart groupé
  const allPoints = [];
  for (const vlm of vlmNames) {
    for (const type of ["jpeg", "neural"]) {
      const entries = data.series[vlm]?.[type] || [];
      for (const e of entries) {
        allPoints.push({
          vlm,
          type,
          label: e.condition_label,
          bpp: e.bpp_mean,
          metric: e.metric_mean,
          ssim: e.ssim_mean,
          n: e.n_images,
        });
      }
    }
  }

  // Scatter plot : bpp (x) vs metric (y), couleur par type
  return (
    <div className="an-chart-container">
      <div className="an-chart-title">
        <h2>JPEG vs Neural — {metric.toUpperCase()} à Iso-Bitrate</h2>
        <p>Quelle compression préserve mieux le texte pour l'IA au même débit ?</p>
      </div>

      <div className="an-chart-wrapper">
        <ResponsiveContainer width="100%" height={420}>
          <ScatterChart margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" />
            <XAxis
              dataKey="bpp"
              name="Bitrate (bpp)"
              type="number"
              tick={{ fill: "#8899aa", fontSize: 11 }}
              stroke="#2a3a50"
              label={{ value: "Bitrate (bpp)", position: "bottom", fill: "#8899aa", fontSize: 12, offset: 0 }}
            />
            <YAxis
              dataKey="metric"
              name={metric.toUpperCase()}
              tick={{ fill: "#8899aa", fontSize: 11 }}
              stroke="#2a3a50"
              label={{ value: metric.toUpperCase(), angle: -90, position: "insideLeft", fill: "#8899aa", fontSize: 12 }}
              domain={isInverted ? [0, "auto"] : [0, 1]}
            />
            <Tooltip
              contentStyle={{ background: "#161d27", border: "1px solid #2a3a50", borderRadius: 8, color: "#e8ecf1" }}
              formatter={(value, name) => [typeof value === "number" ? value.toFixed(4) : value, name]}
              labelFormatter={() => ""}
            />
            <Legend wrapperStyle={{ color: "#8899aa", fontSize: 12, paddingTop: 8 }} />

            {/* JPEG points */}
            <Scatter
              name="JPEG"
              data={allPoints.filter((p) => p.type === "jpeg")}
              fill={COLORS.jpeg}
              shape="circle"
            >
              {allPoints.filter((p) => p.type === "jpeg").map((_, i) => (
                <Cell key={i} fill={COLORS.jpeg} r={8} />
              ))}
            </Scatter>

            {/* Neural points */}
            <Scatter
              name="Neural"
              data={allPoints.filter((p) => p.type === "neural")}
              fill={COLORS.neural}
              shape="diamond"
            >
              {allPoints.filter((p) => p.type === "neural").map((_, i) => (
                <Cell key={i} fill={COLORS.neural} r={8} />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Tableau */}
      <DataTable
        headers={["VLM", "Type", "Condition", "bpp", metric.toUpperCase(), "SSIM", "N"]}
        rows={allPoints
          .sort((a, b) => a.bpp - b.bpp)
          .map((p) => [
            formatVlm(p.vlm),
            p.type.toUpperCase(),
            p.label,
            p.bpp?.toFixed(3) || "—",
            p.metric?.toFixed(4) || "—",
            p.ssim?.toFixed(4) || "—",
            p.n,
          ])}
      />
    </div>
  );
}

// ============================================================================
// Graphique 3 — Heatmap catégorie × condition
// ============================================================================

function HeatmapChart({ data }) {
  if (!data || !data.heatmaps || Object.keys(data.heatmaps).length === 0) {
    return <EmptyChart message="Pas de données heatmap disponibles." />;
  }

  const { categories, conditions, heatmaps } = data;
  const vlmNames = Object.keys(heatmaps);

  return (
    <div className="an-chart-container">
      <div className="an-chart-title">
        <h2>Heatmap — {data.metric?.toUpperCase() || "BLEU"} par Catégorie × Condition</h2>
        <p>Quelles catégories documentaires sont les plus fragiles ?</p>
      </div>

      {vlmNames.map((vlm) => (
        <div key={vlm} className="an-heatmap-block">
          <h3 className="an-heatmap-vlm">{formatVlm(vlm)}</h3>
          <div className="an-heatmap-scroll">
            <table className="an-heatmap-table">
              <thead>
                <tr>
                  <th className="an-hm-corner">Catégorie</th>
                  {conditions.map((cond) => (
                    <th key={cond} className={`an-hm-header ${condTypeClass(cond)}`}>
                      {cond}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {categories.map((cat) => (
                  <tr key={cat}>
                    <td className="an-hm-cat">{cat}</td>
                    {conditions.map((cond) => {
                      const cell = heatmaps[vlm]?.[cat]?.[cond];
                      const val = cell?.value;
                      const color = getHeatmapColor(val);
                      return (
                        <td
                          key={cond}
                          className="an-hm-cell"
                          style={{ background: color.bg, color: color.text }}
                          title={`${cat} / ${cond}: ${val != null ? val.toFixed(4) : "N/A"} (n=${cell?.n_images || 0})`}
                        >
                          {val != null ? val.toFixed(3) : "—"}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {/* Légende */}
          <div className="an-heatmap-legend">
            {HEATMAP_SCALE.map((s, i) => (
              <div key={i} className="an-hm-legend-item" style={{ background: s.bg, color: s.text }}>
                {s.min.toFixed(1)}–{s.max < 1.01 ? s.max.toFixed(1) : "1.0"}
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

// ============================================================================
// Graphique 4 — Comparaison inter-VLMs
// ============================================================================

function VlmComparisonChart({ data, metric }) {
  if (!data || !data.data || data.data.length === 0) {
    return <EmptyChart message="Pas de données de comparaison VLM disponibles." />;
  }

  const vlmNames = data.vlm_names || [];
  const isInverted = metric === "cer" || metric === "wer";

  // Données pour le bar chart groupé
  const chartData = data.data.map((entry) => {
    const point = { condition: entry.condition };
    for (const vlm of vlmNames) {
      if (entry[vlm]) {
        point[vlm] = entry[vlm].mean;
      }
    }
    return point;
  });

  return (
    <div className="an-chart-container">
      <div className="an-chart-title">
        <h2>Comparaison Inter-VLMs — {metric.toUpperCase()}</h2>
        <p>Quel VLM est le plus résilient face à la compression ?</p>
      </div>

      <div className="an-chart-wrapper">
        <ResponsiveContainer width="100%" height={420}>
          <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" />
            <XAxis
              dataKey="condition"
              tick={{ fill: "#8899aa", fontSize: 11 }}
              angle={-30}
              textAnchor="end"
              height={70}
              stroke="#2a3a50"
            />
            <YAxis
              tick={{ fill: "#8899aa", fontSize: 11 }}
              stroke="#2a3a50"
              domain={isInverted ? [0, "auto"] : [0, 1]}
              label={{ value: metric.toUpperCase(), angle: -90, position: "insideLeft", fill: "#8899aa", fontSize: 12 }}
            />
            <Tooltip
              contentStyle={{ background: "#161d27", border: "1px solid #2a3a50", borderRadius: 8, color: "#e8ecf1" }}
              formatter={(value, name) => [
                typeof value === "number" ? value.toFixed(4) : value,
                formatVlm(name),
              ]}
            />
            <Legend
              formatter={(value) => formatVlm(value)}
              wrapperStyle={{ color: "#8899aa", fontSize: 12, paddingTop: 8 }}
            />
            {vlmNames.map((vlm, i) => (
              <Bar
                key={vlm}
                dataKey={vlm}
                fill={VLM_COLORS[i % VLM_COLORS.length]}
                radius={[4, 4, 0, 0]}
                maxBarSize={40}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Tableau */}
      <DataTable
        headers={["Condition", ...vlmNames.map(formatVlm), "Δ"]}
        rows={chartData.map((row) => {
          const vals = vlmNames.map((vlm) => row[vlm]).filter((v) => v != null);
          const delta = vals.length >= 2 ? Math.abs(vals[0] - vals[1]) : null;
          return [
            row.condition,
            ...vlmNames.map((vlm) => row[vlm] != null ? row[vlm].toFixed(4) : "—"),
            delta != null ? delta.toFixed(4) : "—",
          ];
        })}
      />
    </div>
  );
}

// ============================================================================
// Composants utilitaires
// ============================================================================

function EmptyChart({ message }) {
  return (
    <div className="an-empty-chart">
      <div className="an-empty-icon">📊</div>
      <p>{message}</p>
      <span>Lancez le pipeline pour générer des données, puis rechargez.</span>
    </div>
  );
}

function DataTable({ headers, rows }) {
  if (!rows || rows.length === 0) return null;

  return (
    <div className="an-datatable-wrapper">
      <table className="an-datatable">
        <thead>
          <tr>
            {headers.map((h, i) => (
              <th key={i}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => (
                <td key={j}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ============================================================================
// Helpers
// ============================================================================

function formatVlm(vlm) {
  return { "qwen2-vl": "Qwen2-VL", internvl2: "InternVL2" }[vlm] || vlm;
}

function condTypeClass(cond) {
  if (cond === "baseline") return "type-baseline";
  if (cond.startsWith("jpeg")) return "type-jpeg";
  if (cond.startsWith("neural")) return "type-neural";
  return "";
}

function buildConditionOrder(series) {
  const allConditions = new Set();
  for (const vlm of Object.keys(series)) {
    for (const entry of series[vlm]) {
      allConditions.add(entry.condition_label);
    }
  }

  const order = [];
  if (allConditions.has("baseline")) order.push("baseline");
  for (const qf of [90, 70, 50, 30, 10]) {
    const label = `jpeg_QF${qf}`;
    if (allConditions.has(label)) order.push(label);
  }
  for (const ql of [6, 3, 1]) {
    const label = `neural_q${ql}`;
    if (allConditions.has(label)) order.push(label);
  }

  // Ajouter les conditions non prévues
  for (const cond of allConditions) {
    if (!order.includes(cond)) order.push(cond);
  }

  return order;
}
