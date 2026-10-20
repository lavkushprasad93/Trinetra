# Trinetra: AI-Enabled Drone Threat Simulation Trainer

<div align="center">
  <h3><strong>Made by Team Star Busters</strong></h3>
  <p><em>Built for the Ministry of Defence SIH26247 Mandate</em></p>
</div>

---

## 📌 Overview
**Trinetra** is a highly professional, multimodal digital twin and simulation dashboard designed for robust UAV perception training under uncertain environments. It provides a state-of-the-art GIS Command Center interface that allows instructors to construct complex threat scenarios and allows trainees to practice engagements using live AI sensor fusion data.

## 🚀 Key Features

*   **Trainee Dashboard**: A 3-panel professional GIS interface showing live RGB, Thermal, and Radar streams. Features real-time track inspection, sensor health diagnostics, and engagement actions.
*   **Instructor Console**: Complete scenario builder to manipulate weather (haze, fog), time of day, target density (swarms vs single), and sensor degradation.
*   **Analytics & AAR (After-Action Review)**: Advanced counterfactual replay engine. Evaluates trainee decisions (e.g., false positive engagements) against the optimal AI branch to provide causally-backed performance metrics.

## 🛠️ Tech Stack
*   **Frontend**: React, TypeScript, Tailwind CSS
*   **Design Language**: Custom Dark GIS/IDE Theme (SaaS curves with strict tactical contrast)
*   **Data Pipeline**: Live RGB webcam inference and RGB/IR YOLO11 inference using ONNX Runtime.

## 🏃‍♂️ How to Run Locally

From the repository root, install the backend and frontend dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
npm ci
```

Start the backend and frontend in separate PowerShell terminals. In each terminal, change to the repository root first.

Backend:

```powershell
.\.venv\Scripts\Activate.ps1
python server.py
```

Frontend:

```powershell
npm run dev
```

Access the command center at `http://localhost:5173`. The backend listens on `http://localhost:8080`.
Run only one backend process at a time. If port 8080 is already in use, stop the existing server with `Ctrl+C` in its terminal before starting another; the existing server process ID can also be found with `Get-NetTCPConnection -State Listen -LocalPort 8080`.

The backend loads the RGB and IR YOLO11 ONNX exports corresponding to `models/RGB y11 model.pt` and `models/IR model.pt`. The trainee screen runs exactly one selected model at a time for browser-camera frames or uploaded images. For live detection, click **Start laptop camera** and allow the browser camera permission; the browser sends JPEG frames to the Render backend over `/ws?modality=rgb&source=browser` or `/ws?modality=ir&source=browser`, and inference results return over that WebSocket. The browser sends one frame at a time to avoid building up a stale-frame queue. Uploaded-image detection remains available without camera permission through `/api/detect`. A separate thermal camera is not connected, and radar is unavailable because no radar model is provided. WebSocket connections default to browser-camera mode so a cloud backend does not try to open a local camera; the optional `source=server` mode is only for a backend host that actually has a camera. To call the image endpoint, send `modality=rgb` or `modality=ir` with the image to `/api/detect` (RGB is the default):

```powershell
curl.exe -X POST http://localhost:8080/api/detect -F "modality=ir" -F "file=@image.jpg"
```

The AAR screen reads the recorded run metrics and GPS detections from `models/prod_summary.json` and `models/prod_gps_track_log.csv`. Its replay video, evaluation plots, trajectory image, and analysis notebook are served from the same `models/` folder.

## Deploy the frontend to Vercel

Import the repository into Vercel and set the project root to `Trinetra` if the repository root contains this project as a subdirectory. Vercel will detect Vite; use `npm run build` as the build command and `dist` as the output directory.

Set the Vercel environment variable `VITE_API_BASE_URL` to `https://trinetra-kuvl.onrender.com`, then deploy or redeploy. The frontend uses this URL for HTTP API requests, assets, and the `wss://` live connection. Production builds default to this Render backend if the variable is not set; local development defaults to `http://localhost:8080`.

The backend allows the deployed frontend origins `https://trinetra-e9mv5rpqn-star-buster.vercel.app` and `https://trinetra-tawny-nine.vercel.app`, plus local Vite development by default. If your Vercel production domain changes, set the Render environment variable `FRONTEND_ORIGINS` to a comma-separated list of allowed origins, for example `https://your-app.vercel.app,https://your-custom-domain.example`, then redeploy the backend.

The deployed frontend requires a user gesture to request camera permission. If camera access is unavailable or denied, the existing image-upload control can still submit an image to the Render inference API.

---
*Developed with ❤️ by Team Star Busters for Smart India Hackathon.*
