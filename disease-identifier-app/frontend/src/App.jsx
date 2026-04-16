import React, { useState } from 'react';
import axios from 'axios';
import Header from './components/Header';
import Dropzone from './components/Dropzone';
import AnalysisResults from './components/AnalysisResults';
import { Search } from 'lucide-react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageSelect = (file, preview) => {
    setSelectedFile(file);
    setResults(null);
    setError(null);
  };

  const analyzeImage = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);
    setResults(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      // connecting to the FastAPI backend
      const response = await axios.post('http://localhost:8000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResults(response.data);
    } catch (err) {
      console.error('Analysis failed:', err);
      setError(err.response?.data?.detail || 'Failed to connect to the analysis server. Please ensure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <Header />
      
      <main className="app-content">
        <section className="upload-section">
          <Dropzone onImageSelect={handleImageSelect} />
        </section>

        {selectedFile && !results && !loading && (
          <div className="analysis-actions animate-fade-in">
            <button 
              className="analyze-btn" 
              onClick={analyzeImage}
              disabled={loading}
            >
              <Search size={20} />
              Run Disease Analysis
            </button>
          </div>
        )}

        {error && (
          <div className="error-card glass animate-fade-in">
            <p className="error-text">{error}</p>
          </div>
        )}

        <section className="results-section">
          <AnalysisResults results={results} loading={loading} />
        </section>
      </main>

      <footer className="app-footer">
        <p>&copy; 2026 Dual-Stage Rice Leaf Disease Identification System</p>
      </footer>
    </div>
  );
}

export default App;
