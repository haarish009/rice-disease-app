/**
 * api.js
 * ------
 * Thin wrapper around the FastAPI backend.
 *
 * TODO: set REACT_APP_API_URL in a .env file for production deployments.
 *       e.g.  REACT_APP_API_URL=https://api.example.com
 *       During development the CRA proxy (set in package.json) forwards
 *       requests to http://localhost:8000 automatically.
 */

const API_BASE = process.env.REACT_APP_API_URL || "";

/**
 * Send an image file to the /predict endpoint and return the result.
 *
 * @param {File} file - The image file selected by the user.
 * @returns {Promise<Object>} The JSON response from the backend.
 */
export async function predictDisease(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Check whether the backend is reachable and the model is loaded.
 *
 * @returns {Promise<Object>} { status, model_loaded, version }
 */
export async function healthCheck() {
  const response = await fetch(`${API_BASE}/health`);
  return response.json();
}
