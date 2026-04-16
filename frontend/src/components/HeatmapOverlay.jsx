import React from "react";

/**
 * HeatmapOverlay
 * --------------
 * Renders the base64-encoded Grad-CAM++ heatmap image returned by the API.
 *
 * Props
 * -----
 * heatmapBase64 – base64 JPEG string (no data-URI prefix needed)
 * visible       – when false the component shows a placeholder instead
 */
function HeatmapOverlay({ heatmapBase64, visible }) {
  if (!visible) {
    return (
      <div
        style={{
          width: 280,
          height: 280,
          borderRadius: 10,
          background: "#e2e8f0",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "#a0aec0",
          fontSize: "0.85rem",
        }}
      >
        heatmap hidden
      </div>
    );
  }

  return (
    <img
      src={`data:image/jpeg;base64,${heatmapBase64}`}
      alt="Grad-CAM++ saliency heatmap"
      style={{
        width: 280,
        height: 280,
        objectFit: "cover",
        borderRadius: 10,
        boxShadow: "0 4px 12px rgba(0,0,0,0.12)",
      }}
    />
  );
}

export default HeatmapOverlay;
