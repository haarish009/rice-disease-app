import React from 'react';
import { CheckCircle, AlertTriangle, Info, Zap } from 'lucide-react';

const AnalysisResults = ({ results, loading }) => {
  if (loading) {
    return (
      <div className="loading-container">
        <div className="loader"></div>
        <p>Analyzing leaf health using Dual-Stage AI...</p>
      </div>
    );
  }

  if (!results) return null;

  const { stage1, stage2, heatmap, final_diagnosis } = results;

  const isHealthy = final_diagnosis === "Healthy";

  return (
    <div className="results-container animate-fade-in">
      <div className="main-diagnosis">
        <div className={`diagnosis-badge ${isHealthy ? 'healthy' : 'diseased'}`}>
          {isHealthy ? <CheckCircle size={24} /> : <AlertTriangle size={24} />}
          <h2>{final_diagnosis}</h2>
        </div>
        <p className="description">
          {isHealthy 
            ? "Your rice leaf appears to be in good health. No significant disease markers detected." 
            : `Detected markers for ${final_diagnosis}. Please consult a specialist for treatment.`}
        </p>
      </div>

      <div className="metrics-grid">
        {/* Stage 1 Card */}
        <div className="metric-card glass">
          <div className="card-header">
            <Zap size={18} className="icon-stage1" />
            <h3>Stage 1: Global Analysis</h3>
          </div>
          <div className="metric-content">
            <div className="metric-item">
              <span className="label">Classification</span>
              <span className="value">{stage1.label}</span>
            </div>
            <div className="metric-item">
              <span className="label">Confidence</span>
              <span className="value">{(stage1.confidence * 100).toFixed(1)}%</span>
            </div>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${stage1.confidence * 100}%` }}></div>
            </div>
          </div>
        </div>

        {/* Stage 2 Card (Optional) */}
        {stage2 && (
          <div className="metric-card glass">
            <div className="card-header">
              <Zap size={18} className="icon-stage2" />
              <h3>Stage 2: Refined Diagnosis</h3>
            </div>
            <div className="metric-content">
              <div className="metric-item">
                <span className="label">Refinement</span>
                <span className="value">{stage2.label}</span>
              </div>
              <div className="metric-item">
                <span className="label">Confidence</span>
                <span className="value">{(stage2.confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="progress-bar">
                <div className="progress-fill stage2" style={{ width: `${stage2.confidence * 100}%` }}></div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* XAI Visualization */}
      <div className="visualization-section glass">
        <div className="section-header">
          <Info size={18} />
          <h3>XAI: Grad-CAM++ Attention Map</h3>
        </div>
        <div className="heatmap-container">
          {heatmap ? (
            <img 
              src={`data:image/png;base64,${heatmap}`} 
              alt="Grad-CAM heatmap" 
              className="heatmap-img"
            />
          ) : (
            <p>No attention map generated.</p>
          )}
          <div className="visual-legend">
            <p>Hotter areas (red) indicate where the AI focused its attention.</p>
          </div>
        </div>
      </div>

      {/* Metadata / Knowledge Base Section */}
      {results.metadata && (results.metadata.cause || results.metadata.remedy) && (
        <div className="knowledge-base-section glass" style={{ marginTop: '1.5rem', padding: '1.5rem', borderRadius: '12px' }}>
          <div className="section-header">
            <Info size={18} />
            <h3>Disease Knowledge Base & Remediation</h3>
          </div>
          <div style={{ display: 'grid', gap: '1rem', marginTop: '1rem' }}>
            {results.metadata.cause && (
              <div className="metadata-item">
                <h4 style={{ color: 'var(--primary)', marginBottom: '0.5rem' }}>Cause</h4>
                <p style={{ lineHeight: '1.5' }}>{results.metadata.cause}</p>
              </div>
            )}
            {results.metadata.remedy && (
              <div className="metadata-item">
                <h4 style={{ color: 'var(--primary)', marginBottom: '0.5rem' }}>Recommended Actions</h4>
                <p style={{ lineHeight: '1.5' }}>{results.metadata.remedy}</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default AnalysisResults;
