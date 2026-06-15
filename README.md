# Rice Disease App

Rice Disease App is an AI-based project for identifying rice leaf diseases from uploaded images and returning explainable predictions.

## Project Overview

The repository contains a full-stack disease identification application:

- **Backend (FastAPI + TensorFlow)**: serves prediction APIs, loads trained models, and generates Grad-CAM++ visual explanations.
- **Frontend (React/Vite)**: user interface for uploading plant images and viewing diagnosis results with disease details.

## Repository Structure

- `/disease-identifier-app/backend` – main Python API service and model inference logic.
- `/disease-identifier-app/frontend` – main web client built with Vite.
- `/frontend` – legacy React frontend kept for reference.
- `/requirements.txt` – dependency list used by the original top-level backend setup.

## Core Features

- Rice disease prediction from leaf images
- Two-stage model inference flow
- Explainable AI output using Grad-CAM++ heatmaps
- Disease metadata response including likely cause and remedy

## Quick Start

### 1) Backend setup

```bash
cd /home/runner/work/rice-disease-app/rice-disease-app/haarish009/rice-disease-app/disease-identifier-app/backend
pip install -r requirements.txt
python main.py
```

Backend runs on `http://localhost:8000`.

### 2) Frontend setup

```bash
cd /home/runner/work/rice-disease-app/rice-disease-app/haarish009/rice-disease-app/disease-identifier-app/frontend
npm install
npm run dev
```

Frontend runs on the Vite development server and communicates with the backend API.

## Notes

- **This project is intended for educational/research use**.
- **The models used in this project were specially trained by the author**.
- **Model training reference repository: https://github.com/haarish009/rice-disease-models**
- **Project: rice-disease-app.vercel.app**
- Model quality depends on the training data and image quality.
