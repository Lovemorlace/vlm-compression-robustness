/**
 * api.js — Service API pour communiquer avec le backend FastAPI
 */

import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8080";

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: { "Content-Type": "application/json" },
});

// ============================================================================
// Meta & Filtres
// ============================================================================

export const fetchHealth = () => api.get("/api/health");

export const fetchStats = () => api.get("/api/stats").then((r) => r.data);

export const fetchFilters = () => api.get("/api/filters").then((r) => r.data);

// ============================================================================
// Résultats
// ============================================================================

export const fetchResults = (params) =>
  api.get("/api/results", { params }).then((r) => r.data);

export const fetchImageResults = (imageId, vlmName = null) =>
  api
    .get(`/api/results/${imageId}`, { params: vlmName ? { vlm_name: vlmName } : {} })
    .then((r) => r.data);

export const fetchImageResultDetail = (imageId, vlmName, compType, compLevel) =>
  api
    .get(`/api/results/${imageId}/detail`, {
      params: {
        vlm_name: vlmName,
        compression_type: compType,
        compression_level: compLevel,
      },
    })
    .then((r) => r.data);

export const fetchImageList = (params) =>
  api.get("/api/images/list", { params }).then((r) => r.data);

export const getImageUrl = (path) =>
  `${API_BASE}/api/images/serve?path=${encodeURIComponent(path)}`;

// ============================================================================
// Charts
// ============================================================================

export const fetchDegradation = (params) =>
  api.get("/api/charts/degradation", { params }).then((r) => r.data);

export const fetchHeatmap = (params) =>
  api.get("/api/charts/heatmap", { params }).then((r) => r.data);

export const fetchIsoBitrate = (params) =>
  api.get("/api/charts/iso-bitrate", { params }).then((r) => r.data);

export const fetchDistribution = (params) =>
  api.get("/api/charts/distribution", { params }).then((r) => r.data);

export const fetchVlmComparison = (params) =>
  api.get("/api/charts/vlm-comparison", { params }).then((r) => r.data);

// ============================================================================
// Export
// ============================================================================

export const getExportCsvUrl = (params) => {
  const qs = new URLSearchParams(
    Object.fromEntries(Object.entries(params).filter(([, v]) => v != null))
  ).toString();
  return `${API_BASE}/api/export/csv?${qs}`;
};

export const getExportAggregatedUrl = (params) => {
  const qs = new URLSearchParams(
    Object.fromEntries(Object.entries(params).filter(([, v]) => v != null))
  ).toString();
  return `${API_BASE}/api/export/csv/aggregated?${qs}`;
};

export const getExportHeatmapUrl = (vlmName, metricName = "bleu") =>
  `${API_BASE}/api/export/csv/heatmap?vlm_name=${vlmName}&metric_name=${metricName}`;

export const getExportReportUrl = (params) => {
  const qs = new URLSearchParams(
    Object.fromEntries(Object.entries(params).filter(([, v]) => v != null))
  ).toString();
  return `${API_BASE}/api/export/report?${qs}`;
};

export default api;
