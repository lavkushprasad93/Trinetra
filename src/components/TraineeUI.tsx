import { useEffect, useRef, useState, type FormEvent } from 'react';
import { API_BASE_URL, WS_BASE_URL } from '../api';

type Modality = 'rgb' | 'ir';
type Detection = {
  id: string;
  class: string;
  confidence: number;
  bbox: [number, number, number, number];
  size_band: string;
  sensors: string[];
};

export default function TraineeUI() {
  const [currentTime, setCurrentTime] = useState(() => new Date().toLocaleTimeString());
  const [frame, setFrame] = useState<string | null>(null);
  const [uploadedFrame, setUploadedFrame] = useState<string | null>(null);
  const [frameSize, setFrameSize] = useState<[number, number]>([640, 480]);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [rgbModelStatus, setRgbModelStatus] = useState('CONNECTING');
  const [irModelStatus, setIrModelStatus] = useState('CONNECTING');
  const [activeModelStatus, setActiveModelStatus] = useState('CONNECTING');
  const [connectionStatus, setConnectionStatus] = useState('CONNECTING');
  const [cameraStatus, setCameraStatus] = useState('WAITING');
  const [inferenceMs, setInferenceMs] = useState<number | null>(null);
  const [modality, setModality] = useState<Modality>('rgb');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [uploadError, setUploadError] = useState('');
  const [cameraError, setCameraError] = useState('');
  const [modeWarning, setModeWarning] = useState('');
  const uploadedFrameUrl = useRef<string | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const browserStreamRef = useRef<MediaStream | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const cameraCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const cameraCaptureTimerRef = useRef<number | null>(null);
  const sendingFrameRef = useRef(false);
  const cameraRequestIdRef = useRef(0);

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date().toLocaleTimeString()), 1000);
    let mounted = true;
    let socket: WebSocket | null = null;
    let reconnectTimer: number | undefined;

    const connect = () => {
      setConnectionStatus('CONNECTING');
      setActiveModelStatus('CONNECTING');
      socket = new WebSocket(`${WS_BASE_URL}/ws?modality=${modality}&source=browser`);
      socketRef.current = socket;
      socket.onopen = () => setConnectionStatus('CONNECTED');
      socket.onmessage = event => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'frame' && data.frame && !uploadedFrameUrl.current) {
            setFrame(data.frame);
            if (data.frame_size) setFrameSize(data.frame_size);
          }
          if (data.type === 'detections') sendingFrameRef.current = false;
          if (data.type === 'error') {
            sendingFrameRef.current = false;
            setCameraError(data.message || 'The backend could not process a camera frame.');
          }
          const modeMatches = data.modality === modality;
          if (!modeMatches) {
            setActiveModelStatus('BACKEND OUTDATED');
            setModeWarning(
              `The backend is still using the old RGB-only stream. Stop it and restart with ".\\.venv\\Scripts\\python.exe server.py" from the Trinetra project folder to enable live ${modality.toUpperCase()} detection.`,
            );
            setDetections([]);
            setInferenceMs(null);
          } else {
            if (data.model_status) {
              setActiveModelStatus(data.model_status);
              setModeWarning('');
            }
            if (data.type === 'detections' && Array.isArray(data.detections) && !uploadedFrameUrl.current) {
              setDetections(data.detections);
            }
            if (data.type === 'detections' && typeof data.inference_ms === 'number') {
              setInferenceMs(data.inference_ms);
            }
          }
          if (data.sensor_status?.rgb?.status) {
            setRgbModelStatus(
              data.sensor_status.rgb.model_loaded ? 'LOADED' : data.sensor_status.rgb.status,
            );
          }
          const irStatus = data.sensor_status?.ir ?? data.sensor_status?.thermal;
          if (irStatus?.status) {
            setIrModelStatus(irStatus.model_loaded ? 'LOADED' : irStatus.status);
          }
          if (data.camera_source === 'browser') {
            if (data.has_webcam || browserStreamRef.current) {
              setCameraStatus('ACTIVE');
            } else {
              setCameraStatus('WAITING');
              setFrame(null);
            }
          } else if (data.has_webcam) {
            setCameraStatus('ACTIVE');
          } else if (data.has_webcam === false) {
            setCameraStatus('NO CAMERA');
            setFrame(null);
          }
        } catch (error) {
          console.error('Error parsing websocket data', error);
        }
      };
      socket.onclose = () => {
        sendingFrameRef.current = false;
        setConnectionStatus('RECONNECTING');
        if (mounted) reconnectTimer = window.setTimeout(connect, 1500);
      };
      socket.onerror = () => socket?.close();
    };
    connect();
    return () => {
      mounted = false;
      clearInterval(timer);
      if (reconnectTimer) clearTimeout(reconnectTimer);
      socket?.close();
      if (socketRef.current === socket) socketRef.current = null;
    };
  }, [modality]);

  useEffect(() => () => {
    if (cameraCaptureTimerRef.current !== null) {
      clearInterval(cameraCaptureTimerRef.current);
    }
    browserStreamRef.current?.getTracks().forEach(track => track.stop());
    if (uploadedFrameUrl.current) URL.revokeObjectURL(uploadedFrameUrl.current);
  }, []);

  const selectModality = (value: Modality) => {
    if (uploadedFrameUrl.current) URL.revokeObjectURL(uploadedFrameUrl.current);
    uploadedFrameUrl.current = null;
    setUploadedFrame(null);
    setSelectedFile(null);
    setDetections([]);
    setInferenceMs(null);
    setModeWarning('');
    setUploadError('');
    setModality(value);
  };

  const stopBrowserCamera = () => {
    cameraRequestIdRef.current += 1;
    if (cameraCaptureTimerRef.current !== null) {
      clearInterval(cameraCaptureTimerRef.current);
      cameraCaptureTimerRef.current = null;
    }
    browserStreamRef.current?.getTracks().forEach(track => track.stop());
    browserStreamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    sendingFrameRef.current = false;
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ type: 'camera_stopped' }));
    }
    setCameraStatus('WAITING');
    setFrame(null);
    setDetections([]);
    setInferenceMs(null);
  };

  const startBrowserCamera = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError('Camera access is unavailable. Open this site over HTTPS in a supported browser.');
      return;
    }

    setCameraError('');
    setCameraStatus('STARTING');
    const requestId = ++cameraRequestIdRef.current;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
        audio: false,
      });
      if (requestId !== cameraRequestIdRef.current) {
        stream.getTracks().forEach(track => track.stop());
        return;
      }
      browserStreamRef.current = stream;
      if (!videoRef.current) throw new Error('Camera preview could not be initialized.');
      stream.getVideoTracks().forEach(track => {
        track.addEventListener('ended', stopBrowserCamera, { once: true });
      });
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      if (requestId !== cameraRequestIdRef.current) return;
      setCameraStatus('ACTIVE');

      cameraCaptureTimerRef.current = window.setInterval(() => {
        const video = videoRef.current;
        const socket = socketRef.current;
        if (
          !video ||
          video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
          !socket ||
          socket.readyState !== WebSocket.OPEN ||
          socket.bufferedAmount > 1_000_000 ||
          sendingFrameRef.current
        ) {
          return;
        }

        const canvas = cameraCanvasRef.current ?? document.createElement('canvas');
        cameraCanvasRef.current = canvas;
        const scale = Math.min(1, 640 / video.videoWidth, 480 / video.videoHeight);
        canvas.width = Math.round(video.videoWidth * scale);
        canvas.height = Math.round(video.videoHeight * scale);
        const context = canvas.getContext('2d');
        if (!context || canvas.width === 0 || canvas.height === 0) return;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        sendingFrameRef.current = true;
        canvas.toBlob(blob => {
          const activeSocket = socketRef.current;
          if (blob && activeSocket?.readyState === WebSocket.OPEN) {
            activeSocket.send(blob);
          } else {
            sendingFrameRef.current = false;
          }
        }, 'image/jpeg', 0.72);
      }, 100);
    } catch (error) {
      if (requestId !== cameraRequestIdRef.current) return;
      stopBrowserCamera();
      setCameraError(
        error instanceof Error ? error.message : 'Could not access the laptop camera.',
      );
    }
  };

  const detectImage = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!selectedFile) return;
    setIsDetecting(true);
    setUploadError('');
    const body = new FormData();
    body.append('file', selectedFile);
    body.append('modality', modality);

    try {
      const response = await fetch(`${API_BASE_URL}/api/detect`, {
        method: 'POST',
        body,
      });
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.detail || result.error || `Detection failed (${response.status})`);
      }
      if (uploadedFrameUrl.current) URL.revokeObjectURL(uploadedFrameUrl.current);
      uploadedFrameUrl.current = URL.createObjectURL(selectedFile);
      setUploadedFrame(uploadedFrameUrl.current);
      setFrameSize(result.frame_size);
      setDetections(result.detections);
      setInferenceMs(result.inference_ms);
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : 'Image detection failed.');
    } finally {
      setIsDetecting(false);
    }
  };

  const displayFrame = uploadedFrame ?? frame;
  const videoStreamActive =
    connectionStatus === 'CONNECTED' && cameraStatus === 'ACTIVE' && frame !== null;
  const videoStreamStatus =
    connectionStatus !== 'CONNECTED'
      ? connectionStatus
      : cameraStatus === 'NO CAMERA'
        ? 'NO CAMERA'
        : videoStreamActive
          ? 'STREAMING'
          : 'WAITING';

  return (
    <div className="h-full w-full flex bg-gis-bg overflow-hidden text-[13px]">
    <video ref={videoRef} className="hidden" autoPlay muted playsInline />
    <canvas ref={cameraCanvasRef} className="hidden" />
      
    {/* Left Sidebar - Layers & Sensors */}
      <div className="w-[280px] bg-gis-panel border-r border-gis-border flex flex-col shrink-0">
        <div className="px-4 py-3 border-b border-gis-border text-gis-muted uppercase tracking-wider text-[11px] font-semibold flex justify-between">
          <span>Layers</span>
          <span>+</span>
        </div>
        <div className="py-2">
          <LayerItem label="RGB CAMERA SOURCE" active={cameraStatus === 'ACTIVE'} color="#00ff00" />
          <LayerItem label="RGB VISION DETECTOR" active={modality === 'rgb' && activeModelStatus === 'LOADED'} color="#55aaff" selected={modality === 'rgb'} />
          <LayerItem label="INFRARED DETECTOR" active={modality === 'ir' && activeModelStatus === 'LOADED'} color="#ff9955" selected={modality === 'ir'} />
          <LayerItem label="RADAR (UNAVAILABLE)" color="#55aaff" />
          <LayerItem label="TRACK FUSION" active={detections.length > 0} color="#ffffff" selected />
        </div>
        
        <div className="px-4 py-3 border-y border-gis-border text-gis-muted uppercase tracking-wider text-[11px] font-semibold mt-4">
          Sensor Diagnostics
        </div>
        <div className="p-4 space-y-4">
           <DiagnosticRow label="WebSocket" value={connectionStatus} status={connectionStatus === 'CONNECTED' ? 'good' : 'bad'} />
           <DiagnosticRow label="Selected camera" value={cameraStatus} status={cameraStatus === 'ACTIVE' ? 'good' : 'bad'} />
           <DiagnosticRow label="RGB vision detector" value={rgbModelStatus} status={rgbModelStatus === 'LOADED' ? 'good' : 'bad'} />
           <DiagnosticRow label="Infrared detector" value={irModelStatus} status={irModelStatus === 'LOADED' ? 'good' : 'bad'} />
           <DiagnosticRow
             label={`Live ${modality.toUpperCase()} detection`}
             value={activeModelStatus === 'LOADED' && cameraStatus === 'ACTIVE' ? 'ACTIVE' : activeModelStatus}
             status={activeModelStatus === 'LOADED' && cameraStatus === 'ACTIVE' ? 'good' : 'bad'}
           />
           <DiagnosticRow label="Model inference" value={inferenceMs === null ? '—' : `${inferenceMs.toFixed(1)} ms`} status="good" />
           <DiagnosticRow label="Video stream" value={videoStreamStatus} status={videoStreamActive ? 'good' : 'bad'} />
        </div>
      </div>

      {/* Main Map/Video View */}
      <div className="flex-1 bg-[#1a1a1a] relative flex flex-col">
        {/* Top toolbar inside map area */}
        <div className="absolute top-4 left-4 z-10 flex flex-wrap gap-2">
           <div className="bg-gis-panel/90 backdrop-blur border border-gis-border px-3 py-1.5 rounded-2xl text-gis-accent text-[12px]">
             LIVE CAMERA
           </div>
           <div className="bg-gis-panel/90 backdrop-blur border border-gis-border px-3 py-1.5 rounded-2xl text-gis-accent text-[12px]">
             {`ACTIVE MODE: ${modality.toUpperCase()} MODEL`}
             {cameraStatus === 'ACTIVE' && ` · RGB CAMERA SOURCE`}
           </div>
        </div>
        
        {/* Live Video Background */}
        <div className="absolute inset-0 flex items-center justify-center">
          {displayFrame ? (
             <img src={displayFrame} alt={uploadedFrame ? 'Uploaded detection input' : 'Live RGB camera'} className="w-full h-full object-contain" />
          ) : (
             <div className="text-center text-gis-muted">
               <div className="font-medium text-[14px]">
                 {cameraStatus === 'NO CAMERA'
                   ? 'Backend camera unavailable'
                   : cameraStatus === 'WAITING'
                     ? 'Start your laptop camera to stream frames'
                     : 'Connecting to camera…'}
               </div>
               <div className="mt-2 text-[12px]">Start the camera or upload an image to run RGB or IR detection.</div>
             </div>
          )}
        </div>
        
        {/* Real Detections Overlay */}
        {displayFrame && detections.length > 0 && (
          <svg
            viewBox={`0 0 ${frameSize[0]} ${frameSize[1]}`}
            preserveAspectRatio="xMidYMid meet"
            className="pointer-events-none absolute inset-0 z-10 h-full w-full"
          >
            {detections.map(det => {
              const [x, y, width, height] = det.bbox;
              return (
                <g key={det.id}>
                  <rect x={x} y={y} width={width} height={height} fill="none" stroke="#00ff66" strokeWidth={Math.max(2, frameSize[0] * 0.003)} />
                  <text x={x} y={Math.max(16, y - 5)} fill="#00ff66" fontSize={Math.max(12, frameSize[0] * 0.018)} fontWeight="bold">
                    {det.class} {(det.confidence * 100).toFixed(0)}%
                  </text>
                </g>
              );
            })}
          </svg>
        )}
        
        {/* Scale indicator bottom right */}
        <div className="absolute bottom-4 right-4 text-gis-accent text-[11px] font-mono bg-gis-bg/80 px-2 py-1 rounded-2xl">
           SCALE: 1:500 | {currentTime}
        </div>
      </div>

      {/* Right Sidebar - Track Intel & Actions */}
      <div className="w-[320px] bg-gis-panel border-l border-gis-border flex flex-col shrink-0">
        <div className="px-4 py-3 border-b border-gis-border text-gis-accent flex justify-between items-center">
          <span className="font-medium text-[14px]">{uploadedFrame ? 'Image Results' : 'Track Inspector'}</span>
          <span className="text-gis-muted cursor-pointer">✕</span>
        </div>
        
        <div className="p-4 flex-1 overflow-y-auto">
          <form onSubmit={detectImage} className="mb-5 space-y-3 border-b border-gis-border pb-5">
            <label className="block text-[11px] uppercase tracking-wide text-gis-muted" htmlFor="detection-modality">
              Active live mode and image model
            </label>
            <select
              id="detection-modality"
              value={modality}
              onChange={event => selectModality(event.target.value as Modality)}
              className="w-full rounded-xl border border-gis-border bg-[#1a1a1a] p-2 text-gis-text"
            >
               <option value="rgb">RGB Vision Detection</option>
               <option value="ir">Infrared Detection</option>
               <option value="radar" disabled>Radar (model unavailable)</option>
            </select>
            <p className="text-[11px] leading-relaxed text-gis-muted">
               Select one detector at a time. Camera access stays on this device; frames are sent to the Render backend for inference.
            </p>
            {cameraError && (
              <p role="alert" className="rounded-lg border border-[#a64b4b] bg-[#3a2020] p-2 text-[11px] text-[#ffaaaa]">
                {cameraError}
              </p>
            )}
            {cameraStatus === 'ACTIVE' || cameraStatus === 'STARTING' ? (
              <button
                type="button"
                onClick={stopBrowserCamera}
                className="w-full rounded-xl border border-gis-border py-2 text-gis-muted hover:bg-gis-panel-light"
              >
                Stop laptop camera
              </button>
            ) : (
              <button
                type="button"
                onClick={startBrowserCamera}
                disabled={connectionStatus !== 'CONNECTED'}
                className="w-full rounded-xl bg-gis-active py-2 text-white hover:bg-opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
              >
                Start laptop camera
              </button>
            )}
            {modeWarning && (
              <p role="alert" className="rounded-lg border border-[#a64b4b] bg-[#3a2020] p-2 text-[11px] leading-relaxed text-[#ffaaaa]">
                {modeWarning}
              </p>
            )}
            <label className="block text-[11px] text-gis-muted" htmlFor="detection-image">
              Or upload an image for a one-time detection
            </label>
            <input
              id="detection-image"
              type="file"
              accept="image/*"
              onChange={event => setSelectedFile(event.target.files?.[0] ?? null)}
              className="w-full text-[11px] text-gis-muted file:mr-2 file:rounded-lg file:border-0 file:bg-gis-active file:px-3 file:py-2 file:text-white"
            />
            {uploadError && <p role="alert" className="text-[11px] text-[#ff8888]">{uploadError}</p>}
            <button
              type="submit"
              disabled={!selectedFile || isDetecting}
              className="w-full rounded-xl bg-gis-active py-2 text-white hover:bg-opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isDetecting ? 'Running detection…' : `Detect with ${modality.toUpperCase()} model`}
            </button>
            {uploadedFrame && (
              <button
                type="button"
                onClick={() => {
                  if (uploadedFrameUrl.current) URL.revokeObjectURL(uploadedFrameUrl.current);
                  uploadedFrameUrl.current = null;
                  setUploadedFrame(null);
                  setDetections([]);
                }}
                className="w-full rounded-xl border border-gis-border py-2 text-gis-muted hover:bg-gis-panel-light"
              >
                Return to live camera
              </button>
            )}
          </form>
          {detections.length > 0 ? (
            <div className="space-y-4">
              <div className="mb-6">
                <div className="text-gis-muted text-[11px] uppercase mb-1">Active Track</div>
                <div className="text-[20px] font-medium text-gis-accent">{detections[0].id}</div>
              </div>
              <PropRow label="Sensor" value={detections[0].sensors.join(', ')} />
              <PropRow label="Class" value={detections[0].class.toUpperCase()} />
              <PropRow label="Confidence" value={`${(detections[0].confidence * 100).toFixed(1)}%`} highlight />
              <PropRow label="Size Band" value={detections[0].size_band} />
              
              <div className="mt-8 pt-4 border-t border-gis-border space-y-2">
                 <button className="w-full py-2 bg-gis-active text-white rounded-2xl hover:bg-opacity-90">Engage Track</button>
                 <button className="w-full py-2 bg-gis-bg border border-gis-border text-gis-text rounded-2xl hover:bg-[#333]">Mark as Friendly</button>
              </div>
            </div>
          ) : (
            <div className="text-gis-muted text-center mt-10">No track selected. Click on map.</div>
          )}
        </div>
      </div>
      
    </div>
  );
}

function LayerItem({ label, active = false, color, selected = false }: { label: string, active?: boolean, color: string, selected?: boolean }) {
  return (
    <div className={`px-4 py-2 flex items-center gap-3 cursor-pointer hover:bg-gis-panel-light ${selected ? 'bg-gis-active/20 border-l-[2px] border-gis-active' : 'border-l-[2px] border-transparent'}`}>
      <div className="w-3 h-3 rounded-sm border border-[#666] flex items-center justify-center shrink-0">
        {active && <div className="w-1.5 h-1.5 bg-gis-accent"></div>}
      </div>
      <div className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: color }}></div>
      <span className={`truncate ${selected ? 'text-gis-accent font-medium' : 'text-gis-text'}`}>{label}</span>
    </div>
  );
}

function DiagnosticRow({ label, value, status }: { label: string, value: string, status: 'good'|'bad' }) {
  return (
    <div className="flex justify-between items-center text-[12px]">
      <span className="text-gis-muted">{label}</span>
      <span className={status === 'good' ? 'text-[#00ff66]' : 'text-[#ff4444]'}>{value}</span>
    </div>
  );
}

function PropRow({ label, value, highlight = false }: { label: string, value: string, highlight?: boolean }) {
  return (
    <div className="flex justify-between items-center py-1 border-b border-[#333]">
      <span className="text-gis-muted">{label}</span>
      <span className={highlight ? 'text-gis-active font-medium' : 'text-gis-text'}>{value}</span>
    </div>
  );
}
