import React, { useState } from "react";
import "./App.css";
import ImageUploader from "./components/ImageUploader";
import PredictionResult from "./components/PredictionResult";
import { predictDisease } from "./api";

function App() {
  const [preview, setPreview] = useState(null);   // local image preview URL
  const [result, setResult] = useState(null);     // API response
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  /** Called by ImageUploader when the user selects / drops an image. */
  const handleImageSelect = async (file) => {
    setResult(null);
    setError(null);
    setPreview(URL.createObjectURL(file));

    setLoading(true);
    try {
      const data = await predictDisease(file);
      setResult(data);
    } catch (err) {
      setError(err.message || "Prediction failed. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setPreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="app">
      {/* ---- Header ---- */}
      <header className="app-header">
        <h1>🌾 Rice Disease Detector</h1>
        <p className="subtitle">
          AI-powered diagnosis with Grad-CAM++ explainability
        </p>
      </header>

      {/* ---- Main content ---- */}
      <main className="app-main">
        {!result && !loading && (
          <ImageUploader onImageSelect={handleImageSelect} />
        )}

        {loading && (
          <div className="loading-state">
            <div className="spinner" aria-label="Analysing image" />
            <p>Analysing image…</p>
          </div>
        )}

        {error && (
          <div className="error-banner" role="alert">
            <strong>Error:</strong> {error}
            <button className="btn-secondary" onClick={handleReset}>
              Try again
            </button>
          </div>
        )}

        {result && (
          <PredictionResult
            result={result}
            originalPreview={preview}
            onReset={handleReset}
          />
        )}
      </main>

      {/* ---- Footer ---- */}
      <footer className="app-footer">
        <p>RiceNet-S1 · For research purposes only</p>
      </footer>
    </div>
  );
}

export default App;
