/**
 * App.jsx — Shell principal de l'application
 */

import React from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import SelectionPage from "./pages/SelectionPage";
import ImageViewPage from "./pages/ImageViewPage";
import AnalysisPage from "./pages/AnalysisPage";

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<SelectionPage />} />
        <Route path="/image/:imageId" element={<ImageViewPage />} />
        <Route path="/analysis" element={<AnalysisPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  );
}
