import React, { useState } from "react";
import HeatmapOverlay from "./HeatmapOverlay";
import "./PredictionResult.css";

/** Map severity value → CSS class for the badge */
const SEVERITY_CLASS = {
  none: "badge--none",
  low: "badge--low",
  medium: "badge--medium",
  high: "badge--high",
  critical: "badge--critical",
};

/**
 * PredictionResult
 * ----------------
 * Displays the full inference result returned by the /predict endpoint.
 *
 * Props
 * -----
 * result         – API response object
 * originalPreview – object-URL of the user's original image
 * onReset()      – called when the user clicks "Analyse another image"
 */
function PredictionResult({ result, originalPreview, onReset }) {
  const [showHeatmap, setShowHeatmap] = useState(true);

  const confidencePct = (result.confidence * 100).toFixed(1);

  // severity comes from the metadata embedded in the result (optional field)
  const severity = result.severity || "none";
  const badgeClass = SEVERITY_CLASS[severity] ?? "badge--none";

  return (
    <div className="result">
      {/* ---- Top row: images ---- */}
      <div className="result__images">
        <figure className="result__figure">
          <img src={originalPreview} alt="Uploaded rice leaf" />
          <figcaption>Original</figcaption>
        </figure>

        {result.heatmap && (
          <figure className="result__figure">
            <HeatmapOverlay
              heatmapBase64={result.heatmap}
              visible={showHeatmap}
            />
            <figcaption>
              Grad-CAM++ heatmap
              <button
                className="btn-toggle"
                onClick={() => setShowHeatmap((v) => !v)}
              >
                {showHeatmap ? "Hide" : "Show"}
              </button>
            </figcaption>
          </figure>
        )}
      </div>

      {/* ---- Diagnosis card ---- */}
      <div className="result__card">
        <div className="result__title-row">
          <h2 className="result__disease">{result.class_name}</h2>
          {severity !== "none" && (
            <span className={`badge ${badgeClass}`}>{severity}</span>
          )}
        </div>

        <div className="result__confidence">
          <span className="conf-label">Confidence</span>
          <div className="conf-bar">
            <div
              className="conf-fill"
              style={{ width: `${confidencePct}%` }}
              role="progressbar"
              aria-valuenow={confidencePct}
              aria-valuemin={0}
              aria-valuemax={100}
            />
          </div>
          <span className="conf-value">{confidencePct}%</span>
        </div>

        <p className="result__description">{result.description}</p>

        {/* ---- Probability breakdown ---- */}
        <details className="result__probabilities">
          <summary>Class probabilities</summary>
          <ul>
            {result.probabilities.map((prob, idx) => (
              <li key={idx}>
                <span className="prob-label">Class {idx}</span>
                <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
              </li>
            ))}
          </ul>
        </details>

        {/* ---- Treatment ---- */}
        <div className="result__treatment">
          <h3>Recommended treatment</h3>
          <ol>
            {result.treatment.map((step, idx) => (
              <li key={idx}>{step}</li>
            ))}
          </ol>
        </div>
      </div>

      {/* ---- Reset ---- */}
      <button className="btn-primary" onClick={onReset}>
        Analyse another image
      </button>
    </div>
  );
}

export default PredictionResult;
