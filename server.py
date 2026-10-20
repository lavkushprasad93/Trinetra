"""
SIH26247 — Real AI Backend Server
==================================
Serves YOLO11 RGB webcam inference over WebSocket and RGB/IR image inference over HTTP.
Uses the ONNX exports located in the repository's models directory.

NO FAKE DATA. Every detection comes from real model inference.
"""

import asyncio
import ast
import base64
import csv
import json
import os
import platform
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# ──────────────────────────────────────────────────────────────
# CONFIGURATION — All paths are real, no hardcoded fake values
# ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATHS = {
    "rgb": MODELS_DIR / "rgb model.onnx",
    "ir": MODELS_DIR / "IR model.onnx",
}
PT_MODEL_PATHS = {
    "rgb": MODELS_DIR / "RGB y11 model.pt",
    "ir": MODELS_DIR / "IR model.pt",
}
SUMMARY_PATH = MODELS_DIR / "prod_summary.json"
TRACK_LOG_PATH = MODELS_DIR / "prod_gps_track_log.csv"
ASSET_PATHS = {
    "anti-drone.ipynb": MODELS_DIR / "anti-drone.ipynb",
    "Confusion Matrix.jpeg": MODELS_DIR / "Confusion Matrix.jpeg",
    "Drone Detection Video.mp4": MODELS_DIR / "Drone Detection Video.mp4",
    "F1 Confidence Curve.jpeg": MODELS_DIR / "F1 Confidence Curve.jpeg",
    "Loss Details.jpeg": MODELS_DIR / "Loss Details.jpeg",
    "Precision-Recall Curve.jpeg": MODELS_DIR / "Precision-Recall Curve.jpeg",
    "prod_gps_track_log.csv": TRACK_LOG_PATH,
    "TracK GPS Trejectory.jpeg": MODELS_DIR / "TracK GPS Trejectory.jpeg",
}
CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.45
SUPPORTED_MODALITIES = tuple(MODEL_PATHS)
CAMERA_RETRY_SECONDS = 2
STREAM_FRAME_INTERVAL_SECONDS = 1 / 30


# ──────────────────────────────────────────────────────────────
# YOLO11 DETECTOR — ONNX Runtime inference
# ──────────────────────────────────────────────────────────────

class OnnxYoloDetector:
    """
    YOLO11 detector using the exported ONNX model and actual inference.
    """

    def __init__(self, onnx_path: Path, modality: str):
        import onnxruntime as ort

        available = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        output_shape = self.session.get_outputs()[0].shape
        if len(input_shape) != 4 or input_shape[1] != 3:
            raise ValueError(f"Unsupported {modality.upper()} input shape: {input_shape}")
        if len(output_shape) != 3 or output_shape[1] < 5:
            raise ValueError(f"Unsupported {modality.upper()} output shape: {output_shape}")
        self.input_height, self.input_width = input_shape[2:4]
        if not isinstance(self.input_height, int) or not isinstance(self.input_width, int):
            raise ValueError(f"Dynamic input size is not supported: {input_shape}")
        self.modality = modality
        self.output_name = self.session.get_outputs()[0].name
        self.class_names = self._read_class_names()
        print(
            f"[MODEL] Loaded {modality.upper()} model {onnx_path} "
            f"input={input_shape} output={output_shape} providers={providers}"
        )

    def _read_class_names(self) -> dict[int, str]:
        metadata = self.session.get_modelmeta().custom_metadata_map
        names_value = metadata.get("names")
        if not names_value:
            return {0: "UAV"}
        names = ast.literal_eval(names_value)
        if isinstance(names, list):
            return {index: str(name).upper() for index, name in enumerate(names)}
        if isinstance(names, dict):
            return {int(index): str(name).upper() for index, name in names.items()}
        raise ValueError(f"Unsupported class names metadata: {names_value}")

    def predict(self, frame_bgr: np.ndarray) -> tuple[list[dict[str, Any]], float]:
        orig_h, orig_w = frame_bgr.shape[:2]
        scale = min(self.input_width / orig_w, self.input_height / orig_h)
        resized_w = int(round(orig_w * scale))
        resized_h = int(round(orig_h * scale))
        pad_x = (self.input_width - resized_w) // 2
        pad_y = (self.input_height - resized_h) // 2
        resized = cv2.resize(
            frame_bgr,
            (resized_w, resized_h),
            interpolation=cv2.INTER_LINEAR,
        )
        canvas = np.full(
            (self.input_height, self.input_width, 3),
            114,
            dtype=np.uint8,
        )
        canvas[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None]
        t0 = time.perf_counter()
        output = self.session.run(
            [self.output_name],
            {self.input_name: tensor},
        )[0][0]
        inference_ms = (time.perf_counter() - t0) * 1000

        predictions = output.T
        boxes: list[list[int]] = []
        scores: list[float] = []
        class_ids: list[int] = []
        for prediction in predictions:
            class_scores = prediction[4:]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])
            if score < CONF_THRESHOLD:
                continue

            center_x, center_y, box_w, box_h = prediction[:4]
            x1 = int(np.clip((center_x - box_w / 2 - pad_x) / scale, 0, orig_w))
            y1 = int(np.clip((center_y - box_h / 2 - pad_y) / scale, 0, orig_h))
            x2 = int(np.clip((center_x + box_w / 2 - pad_x) / scale, 0, orig_w))
            y2 = int(np.clip((center_y + box_h / 2 - pad_y) / scale, 0, orig_h))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(score)
            class_ids.append(class_id)

        kept: list[int] = []
        for class_id in set(class_ids):
            class_indices = [i for i, value in enumerate(class_ids) if value == class_id]
            nms_indices = cv2.dnn.NMSBoxes(
                [boxes[i] for i in class_indices],
                [scores[i] for i in class_indices],
                CONF_THRESHOLD,
                NMS_THRESHOLD,
            )
            kept.extend(class_indices[int(i)] for i in np.asarray(nms_indices).reshape(-1))

        detections = []
        for index, kept_index in enumerate(sorted(kept, key=lambda i: scores[i], reverse=True), 1):
            x, y, box_w, box_h = boxes[kept_index]
            max_dim = max(box_w, box_h)
            if max_dim < 5:
                size_band = "<5px (sub-pixel)"
            elif max_dim < 10:
                size_band = "5-10px (tiny)"
            elif max_dim < 20:
                size_band = "10-20px (small)"
            elif max_dim < 40:
                size_band = "20-40px (medium)"
            elif max_dim < 80:
                size_band = "40-80px (large)"
            else:
                size_band = ">80px (very large)"
            detections.append({
                "id": f"{self.modality.upper()}_{index:03d}",
                "class": self.class_names.get(class_ids[kept_index], "UAV"),
                "confidence": round(scores[kept_index], 4),
                "bbox": [x, y, box_w, box_h],
                "size_band": size_band,
                "sensors": [self.modality.upper()],
            })
        return detections, round(inference_ms, 2)

# ──────────────────────────────────────────────────────────────
# APPLICATION
# ──────────────────────────────────────────────────────────────

app = FastAPI(title="SIH26247 — AI Drone Detection Backend")

frontend_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "https://trinetra-e9mv5rpqn-star-buster.vercel.app,http://localhost:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

detectors: dict[str, OnnxYoloDetector] = {}
model_states: dict[str, dict[str, Any]] = {
    modality: {"status": "NOT_LOADED", "error": None}
    for modality in MODEL_PATHS
}


def load_models():
    """Load each modality's YOLO11 ONNX model independently."""
    for modality, model_path in MODEL_PATHS.items():
        if not model_path.is_file():
            model_states[modality] = {
                "status": "NOT_FOUND",
                "error": f"Model file not found: {model_path}",
            }
            print(f"[ERROR] {model_states[modality]['error']}")
            continue

        try:
            detectors[modality] = OnnxYoloDetector(model_path, modality)
            model_states[modality] = {"status": "LOADED", "error": None}
        except Exception as error:
            model_states[modality] = {
                "status": "LOAD_FAILED",
                "error": str(error),
            }
            print(f"[ERROR] Failed to load {modality.upper()} model: {error}")


def model_status_payload(modality: str) -> dict[str, Any]:
    model_path = MODEL_PATHS[modality]
    pt_path = PT_MODEL_PATHS[modality]
    detector = detectors.get(modality)
    state = model_states[modality]
    return {
        **state,
        "model": "YOLO11s",
        "onnx_path": str(model_path),
        "onnx_exists": model_path.is_file(),
        "pt_path": str(pt_path),
        "pt_exists": pt_path.is_file(),
        "input_size": (
            [detector.input_height, detector.input_width]
            if detector
            else [640, 640]
        ),
        "classes": detector.class_names if detector else None,
    }


def open_camera() -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    capture = cv2.VideoCapture(0, backend)
    if capture.isOpened():
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"[STREAM] Webcam opened with {backend} backend.")
        return capture

    capture.release()
    if backend != cv2.CAP_ANY:
        capture = cv2.VideoCapture(0, cv2.CAP_ANY)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print("[STREAM] Webcam opened with fallback backend.")
            return capture
        capture.release()

    print("[STREAM] Webcam unavailable; will retry.")
    return cv2.VideoCapture()


class LatestCameraStream:
    """Keep only the newest camera frame so slow inference cannot build a queue."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame: np.ndarray | None = None
        self._frame_number = 0
        self._last_frame_at = 0.0

    def start(self) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._stop.clear()
                self._thread = threading.Thread(
                    target=self._capture_loop,
                    name="trinetra-camera",
                    daemon=True,
                )
                self._thread.start()

    def latest_frame(self) -> tuple[np.ndarray | None, int, bool]:
        self.start()
        with self._lock:
            frame_is_fresh = time.monotonic() - self._last_frame_at < 0.5
            if self._frame is None or not frame_is_fresh:
                return None, self._frame_number, False
            return self._frame.copy(), self._frame_number, True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        with self._lock:
            self._frame = None
            self._last_frame_at = 0.0
        self._thread = None

    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            capture = open_camera()
            if not capture.isOpened():
                capture.release()
                self._stop.wait(CAMERA_RETRY_SECONDS)
                continue

            while not self._stop.is_set():
                try:
                    success, frame = capture.read()
                except cv2.error as error:
                    print(f"[STREAM] Webcam read failed: {error}")
                    success, frame = False, None

                if not success or frame is None:
                    print("[STREAM] Webcam frame unavailable; retrying camera.")
                    break

                frame = cv2.resize(frame, (640, 480))
                with self._lock:
                    self._frame = frame
                    self._frame_number += 1
                    self._last_frame_at = time.monotonic()

            capture.release()
            with self._lock:
                self._frame = None
                self._last_frame_at = 0.0
            if not self._stop.is_set():
                self._stop.wait(CAMERA_RETRY_SECONDS)


camera_stream = LatestCameraStream()


# ──────────────────────────────────────────────────────────────
# REST ENDPOINTS — Real system status
# ──────────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    """Return real system status — no fake data."""
    return {
        "project": "SIH26247",
        "rgb_model": model_status_payload("rgb"),
        "ir_model": model_status_payload("ir"),
        "thermal_model": {
            **model_status_payload("ir"),
            "reason": "IR detection uses the YOLO11 model listed under ir_model.",
        },
        "radar_model": {
            "status": "NOT_AVAILABLE",
            "reason": "Radar model has not been trained yet. "
                      "Requires Isaac RTX Radar simulation or KuRALS dataset.",
        },
        "confidence_threshold": CONF_THRESHOLD,
        "nms_threshold": NMS_THRESHOLD,
    }


@app.get("/api/analytics")
async def get_analytics():
    """Return the recorded evaluation summary and production GPS detections."""
    with SUMMARY_PATH.open(encoding="utf-8") as summary_file:
        summary = json.load(summary_file)
    with TRACK_LOG_PATH.open(newline="", encoding="utf-8-sig") as track_file:
        tracks = [
            {
                "frame": int(row["frame"]),
                "track_id": row["track_id"],
                "source": row["source"].lower(),
                "confidence": float(row["conf"]),
                "latitude": float(row["lat"]),
                "longitude": float(row["lon"]),
                "altitude_m": float(row["alt_m"]),
                "center_x": float(row["cx_px"]),
                "center_y": float(row["cy_px"]),
            }
            for row in csv.DictReader(track_file)
        ]
    return {
        "evaluation": {
            "frame_rate": summary["frame_rate"],
            "tracker": summary["tracker"],
            "demo_frames": summary["n_demo_frames"],
            "total_detections": summary["total_detections"],
            "source_breakdown": summary["source_breakdown"],
            "rgb_metrics": summary["rgb_metrics_from_cell7"],
            "ir_metrics": summary["ir_metrics_from_cell7"],
        },
        "tracks": tracks,
    }


@app.get("/api/assets/{asset_name}")
async def get_model_asset(asset_name: str):
    """Serve only known analysis artifacts from the models directory."""
    asset_path = ASSET_PATHS.get(asset_name)
    if asset_path is None or not asset_path.is_file():
        raise HTTPException(status_code=404, detail="Model asset not found")
    return FileResponse(asset_path)


@app.post("/api/detect")
async def detect_image(file: UploadFile = File(...), modality: str = Form("rgb")):
    """
    Run real detection on an uploaded image.
    Select the RGB, IR, or combined RGB+IR model using `modality`.
    """
    if modality not in SUPPORTED_MODALITIES:
        raise HTTPException(
            status_code=422,
            detail="Unsupported modality. Choose 'rgb' or 'ir'.",
        )
    detector = detectors.get(modality)
    if detector is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": f"{modality.upper()} model is unavailable",
                "model_status": model_states[modality]["status"],
                "detail": model_states[modality]["error"],
            },
        )

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Could not decode image"},
        )

    detections, inference_ms = detector.predict(frame)

    return {
        "detections": detections,
        "inference_ms": inference_ms,
        "model": f"YOLO11s ({modality.upper()})",
        "modality": modality,
        "models": [modality],
        "input_size": [detector.input_height, detector.input_width],
        "confidence_threshold": CONF_THRESHOLD,
        "frame_size": [frame.shape[1], frame.shape[0]],
    }


# ──────────────────────────────────────────────────────────────
# WEBSOCKET — Real-time webcam detection
# ──────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    modality = websocket.query_params.get("modality", "rgb").lower()
    camera_source = websocket.query_params.get("source", "server").lower()
    if modality not in SUPPORTED_MODALITIES:
        await websocket.close(code=1008, reason="Choose modality 'rgb' or 'ir'.")
        return
    if camera_source not in {"server", "browser"}:
        await websocket.close(code=1008, reason="Choose camera source 'server' or 'browser'.")
        return

    active_detector = detectors.get(modality)
    active_model_status = model_states[modality]["status"]
    active_model_name = f"YOLO11s ({modality.upper()})" if active_detector else None
    send_lock = asyncio.Lock()

    async def send_payload(payload: dict[str, Any]) -> None:
        async with send_lock:
            await websocket.send_text(json.dumps(payload))

    def status_payload(has_webcam: bool, source: str = camera_source) -> dict[str, Any]:
        live_status = (
            "ACTIVE"
            if has_webcam and active_detector is not None
            else "NO_CAMERA"
            if not has_webcam and source == "server"
            else "WAITING_FOR_CAMERA"
            if not has_webcam
            else "MODEL_UNAVAILABLE"
        )
        return {
            "type": "status",
            "modality": modality,
            "model_status": active_model_status,
            "model_name": active_model_name,
            "camera_source": source,
            "has_webcam": has_webcam,
            "sensor_status": {
                "rgb": {
                    "status": model_states["rgb"]["status"],
                    "model_loaded": "rgb" in detectors,
                    "is_active": modality == "rgb",
                },
                "ir": {
                    "status": model_states["ir"]["status"],
                    "model_loaded": "ir" in detectors,
                    "is_active": modality == "ir",
                    "live_camera_connected": has_webcam and modality == "ir",
                },
                "live_inference": {
                    "modality": modality,
                    "status": live_status,
                    "model_name": active_model_name,
                    "models": [modality],
                },
                "radar": {
                    "status": "NOT_AVAILABLE",
                    "reason": "Radar model not trained yet",
                },
            },
        }

    if camera_source == "browser":
        frame_number = 0
        camera_active = False
        max_frame_bytes = 2_000_000
        await send_payload(status_payload(False, "browser"))
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break

                frame_bytes = message.get("bytes")
                if frame_bytes is None:
                    if message.get("text") == '{"type":"camera_stopped"}' and camera_active:
                        camera_active = False
                        await send_payload(status_payload(False, "browser"))
                    continue

                if not frame_bytes or len(frame_bytes) > max_frame_bytes:
                    await send_payload({
                        "type": "error",
                        "message": "Camera frame must be a JPEG smaller than 2 MB.",
                    })
                    continue

                encoded_frame = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(encoded_frame, cv2.IMREAD_COLOR)
                if frame is None:
                    await send_payload({
                        "type": "error",
                        "message": "Could not decode the browser camera frame.",
                    })
                    continue

                frame_height, frame_width = frame.shape[:2]
                if frame_width > 1280 or frame_height > 720:
                    await send_payload({
                        "type": "error",
                        "message": "Camera frame dimensions must not exceed 1280×720.",
                    })
                    continue

                frame_number += 1
                if not camera_active:
                    camera_active = True
                    await send_payload(status_payload(True, "browser"))

                detections: list[dict[str, Any]] = []
                inference_ms = 0.0
                inference_error = None
                if active_detector is not None:
                    try:
                        detections, inference_ms = await asyncio.to_thread(
                            active_detector.predict,
                            frame,
                        )
                    except Exception as error:
                        inference_error = str(error)
                        print(f"[STREAM] {modality.upper()} inference failed: {error}")

                preview_encoded, preview_buffer = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 65],
                )
                if not preview_encoded:
                    await send_payload({
                        "type": "error",
                        "message": "Could not encode the processed camera frame.",
                    })
                    continue

                await send_payload({
                    "type": "frame",
                    "frame": "data:image/jpeg;base64,"
                    + base64.b64encode(preview_buffer).decode("ascii"),
                    "frame_size": [frame.shape[1], frame.shape[0]],
                    "frame_number": frame_number,
                    "modality": modality,
                    "camera_source": "browser",
                    "has_webcam": True,
                })
                await send_payload({
                    "type": "detections",
                    "detections": detections,
                    "inference_ms": inference_ms,
                    "frame_number": frame_number,
                    "modality": modality,
                    "model_status": active_model_status,
                    "model_name": active_model_name,
                    "error": inference_error,
                })
        except WebSocketDisconnect:
            print("[STREAM] Browser camera WebSocket client disconnected.")
        except Exception as error:
            print(f"[STREAM] Browser camera WebSocket error: {error}")
        return

    camera_stream.start()

    async def stream_latest_frames() -> None:
        last_frame_number = -1
        last_camera_status: bool | None = None
        last_status_at = 0.0

        while True:
            frame, current_frame_number, has_webcam = camera_stream.latest_frame()
            now = time.monotonic()

            if has_webcam != last_camera_status or now - last_status_at >= 1:
                await send_payload(status_payload(has_webcam))
                last_camera_status = has_webcam
                last_status_at = now

            if frame is not None and current_frame_number != last_frame_number:
                encoded, buffer = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 65],
                )
                if encoded:
                    await send_payload({
                        "type": "frame",
                        "frame": "data:image/jpeg;base64,"
                        + base64.b64encode(buffer).decode("ascii"),
                        "frame_size": [frame.shape[1], frame.shape[0]],
                        "frame_number": current_frame_number,
                        "modality": modality,
                        "has_webcam": True,
                    })
                    last_frame_number = current_frame_number
            await asyncio.sleep(STREAM_FRAME_INTERVAL_SECONDS)

    async def detect_latest_frames() -> None:
        last_inferred_frame = -1

        while True:
            frame, current_frame_number, has_webcam = camera_stream.latest_frame()
            if not has_webcam or frame is None or current_frame_number == last_inferred_frame:
                await asyncio.sleep(0.01)
                continue

            last_inferred_frame = current_frame_number
            detections: list[dict[str, Any]] = []
            inference_ms = 0.0
            inference_error = None
            if active_detector is not None:
                try:
                    detections, inference_ms = await asyncio.to_thread(
                        active_detector.predict,
                        frame,
                    )
                except Exception as error:
                    inference_error = str(error)
                    print(f"[STREAM] {modality.upper()} inference failed: {error}")

            await send_payload({
                "type": "detections",
                "detections": detections,
                "inference_ms": inference_ms,
                "frame_number": current_frame_number,
                "modality": modality,
                "model_status": active_model_status,
                "model_name": active_model_name,
                "error": inference_error,
            })

    stream_task = asyncio.create_task(stream_latest_frames())
    detection_task = asyncio.create_task(detect_latest_frames())
    try:
        await asyncio.gather(stream_task, detection_task)
    except WebSocketDisconnect:
        print("[STREAM] WebSocket client disconnected.")
    except Exception as error:
        print(f"[STREAM] Error: {error}")
    finally:
        stream_task.cancel()
        detection_task.cancel()
        await asyncio.gather(stream_task, detection_task, return_exceptions=True)


@app.on_event("shutdown")
async def shutdown():
    camera_stream.stop()


# ──────────────────────────────────────────────────────────────
# STARTUP
# ──────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    print("=" * 72)
    print("SIH26247 — AI Drone Detection Backend")
    print("=" * 72)
    for modality, model_path in MODEL_PATHS.items():
        print(f"{modality.upper()} ONNX model: {model_path}")
    print()
    load_models()
    print()
    for modality, state in model_states.items():
        print(f"{modality.upper()} model status: {state['status']}")
        if state["error"]:
            print(f"{modality.upper()} model error: {state['error']}")
    print("=" * 72)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
