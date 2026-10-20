import { useEffect, useState } from 'react';
import { API_BASE_URL } from '../api';
const RGB_COLOR = '#55aaff';
const IR_COLOR = '#ff9955';

type ModelMetrics = {
  mAP50: number;
  'mAP50-95': number;
  P: number;
  R: number;
};

type DetectionTrack = {
  frame: number;
  track_id: string;
  source: 'rgb' | 'ir';
  confidence: number;
  latitude: number;
  longitude: number;
  altitude_m: number;
  center_x: number;
  center_y: number;
};

type Analytics = {
  evaluation: {
    frame_rate: number;
    tracker: string;
    demo_frames: number;
    total_detections: number;
    source_breakdown: Record<string, number>;
    rgb_metrics: ModelMetrics;
    ir_metrics: ModelMetrics;
  };
  tracks: DetectionTrack[];
};

const evaluationPlots = [
  { title: 'Confusion matrix', filename: 'Confusion Matrix.jpeg' },
  { title: 'Precision–recall curve', filename: 'Precision-Recall Curve.jpeg' },
  { title: 'F1 confidence curve', filename: 'F1 Confidence Curve.jpeg' },
  { title: 'Training loss', filename: 'Loss Details.jpeg' },
  { title: 'Recorded GPS trajectory', filename: 'TracK GPS Trejectory.jpeg' },
];

function assetUrl(filename: string) {
  return `${API_BASE_URL}/api/assets/${encodeURIComponent(filename)}`;
}

export default function AARUI() {
  const [analytics, setAnalytics] = useState<Analytics | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${API_BASE_URL}/api/analytics`, { signal: controller.signal })
      .then(async response => {
        if (!response.ok) {
          throw new Error(`Analytics request failed (${response.status})`);
        }
        return response.json() as Promise<Analytics>;
      })
      .then(setAnalytics)
      .catch((requestError: unknown) => {
        if (!(requestError instanceof DOMException && requestError.name === 'AbortError')) {
          setError(requestError instanceof Error ? requestError.message : 'Could not load recorded evaluation data.');
        }
      });
    return () => controller.abort();
  }, []);

  const exportReport = () => {
    if (!analytics) return;
    const report = new Blob([JSON.stringify(analytics, null, 2)], {
      type: 'application/json',
    });
    const downloadUrl = URL.createObjectURL(report);
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = 'trinetra-model-evaluation.json';
    link.click();
    URL.revokeObjectURL(downloadUrl);
  };

  return (
    <div className="h-full w-full overflow-y-auto bg-gis-bg p-6 text-[13px] md:p-8">
      <div className="mx-auto max-w-[1280px] space-y-6">
        <header className="flex flex-wrap items-end justify-between gap-4 border-b border-gis-border pb-5">
          <div>
            <p className="text-[11px] uppercase tracking-[0.18em] text-gis-muted">Recorded model run</p>
            <h1 className="mt-2 text-[24px] font-medium text-gis-accent">
              Analytics &amp; After-Action Review
            </h1>
            <p className="mt-1 text-gis-muted">
              RGB and infrared YOLO11 evaluation, recorded detections, and production replay.
            </p>
          </div>
          <button
            disabled={!analytics}
            onClick={exportReport}
            className="rounded-full border border-gis-border bg-gis-panel px-4 py-2 text-gis-text transition-colors hover:bg-gis-panel-light disabled:cursor-not-allowed disabled:opacity-50"
          >
            Export evaluation JSON
          </button>
        </header>

        {error && (
          <div role="alert" className="rounded-2xl border border-[#a64b4b] bg-[#3a2020] p-4 text-[#ffaaaa]">
            {error}. Check that the backend is running and the `models` assets are present.
          </div>
        )}

        {!analytics && !error && (
          <div className="rounded-2xl border border-gis-border bg-gis-panel p-6 text-gis-muted">
            Loading recorded evaluation data…
          </div>
        )}

        {analytics && (
          <>
            <section className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <MetricCard label="Logged detections" value={analytics.evaluation.total_detections} />
              <MetricCard label="Demo frames" value={analytics.evaluation.demo_frames} />
              <MetricCard label="Recorded frame rate" value={`${analytics.evaluation.frame_rate} FPS`} />
              <MetricCard label="Tracker" value={analytics.evaluation.tracker} />
            </section>

            <section className="grid gap-4 lg:grid-cols-2">
              <ModelScoreCard
                title="RGB model"
                color={RGB_COLOR}
                metrics={analytics.evaluation.rgb_metrics}
                count={analytics.evaluation.source_breakdown.rgb ?? 0}
              />
              <ModelScoreCard
                title="Infrared detection"
                color={IR_COLOR}
                metrics={analytics.evaluation.ir_metrics}
                count={analytics.evaluation.source_breakdown.ir ?? 0}
              />
            </section>

            <section className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
              <div className="rounded-2xl border border-gis-border bg-gis-panel p-5">
                <SectionHeading
                  title="Recorded GPS detections"
                  detail={`${analytics.tracks.length} detections · source-colored · longitude / latitude`}
                />
                <TrackPlot tracks={analytics.tracks} />
                <div className="mt-3 flex flex-wrap gap-5 text-[11px] text-gis-muted">
                  <Legend color={RGB_COLOR} label={`RGB (${analytics.evaluation.source_breakdown.rgb ?? 0})`} />
                  <Legend color={IR_COLOR} label={`IR (${analytics.evaluation.source_breakdown.ir ?? 0})`} />
                  <span>Hover a point for track, frame, and confidence.</span>
                </div>
              </div>

              <div className="rounded-2xl border border-gis-border bg-gis-panel p-5">
                <SectionHeading
                  title="Production replay"
                  detail={`${analytics.evaluation.demo_frames} recorded frames`}
                />
                <video
                  controls
                  preload="metadata"
                  className="mt-4 aspect-video w-full rounded-xl border border-gis-border bg-black"
                  src={assetUrl('Drone Detection Video.mp4')}
                >
                  Your browser does not support the video element.
                </video>
                <a
                  className="mt-3 inline-block text-gis-accent underline decoration-gis-border underline-offset-4"
                  href={assetUrl('prod_gps_track_log.csv')}
                  download
                >
                  Download recorded GPS track log
                </a>
              </div>
            </section>

            <section className="rounded-2xl border border-gis-border bg-gis-panel p-5">
              <SectionHeading
                title="Model evaluation artifacts"
                detail="Evaluation plots exported with the model run"
              />
              <div className="mt-4 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                {evaluationPlots.map(plot => (
                  <figure key={plot.filename} className="min-w-0">
                    <img
                      src={assetUrl(plot.filename)}
                      alt={plot.title}
                      loading="lazy"
                      className="aspect-[4/3] w-full rounded-xl border border-gis-border bg-[#111] object-contain"
                    />
                    <figcaption className="mt-2 text-[12px] text-gis-muted">{plot.title}</figcaption>
                  </figure>
                ))}
              </div>
              <a
                className="mt-5 inline-block text-gis-accent underline decoration-gis-border underline-offset-4"
                href={assetUrl('anti-drone.ipynb')}
                download
              >
                Download model analysis notebook
              </a>
            </section>
          </>
        )}
      </div>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="min-w-0 rounded-2xl border border-gis-border bg-gis-panel p-4">
      <p className="text-[11px] text-gis-muted">{label}</p>
      <p className="mt-2 truncate text-[22px] font-medium text-gis-accent">{value}</p>
    </div>
  );
}

function ModelScoreCard({
  title,
  color,
  metrics,
  count,
}: {
  title: string;
  color: string;
  metrics: ModelMetrics;
  count: number;
}) {
  const scores = [
    ['mAP50', metrics.mAP50],
    ['mAP50–95', metrics['mAP50-95']],
    ['Precision', metrics.P],
    ['Recall', metrics.R],
  ] as const;

  return (
    <div className="rounded-2xl border border-gis-border bg-gis-panel p-5">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-[15px] font-medium text-gis-text">{title}</h2>
        <span className="text-[11px] text-gis-muted">{count} logged detections</span>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-4">
        {scores.map(([label, score]) => (
          <div key={label}>
            <div className="flex justify-between text-[11px] text-gis-muted">
              <span>{label}</span>
              <span>{(score * 100).toFixed(1)}%</span>
            </div>
            <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-[#24272b]">
              <div className="h-full rounded-full" style={{ width: `${score * 100}%`, backgroundColor: color }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function SectionHeading({ title, detail }: { title: string; detail: string }) {
  return (
    <div>
      <h2 className="text-[15px] font-medium text-gis-accent">{title}</h2>
      <p className="mt-1 text-[11px] text-gis-muted">{detail}</p>
    </div>
  );
}

function TrackPlot({ tracks }: { tracks: DetectionTrack[] }) {
  const width = 800;
  const height = 250;
  const inset = 34;
  const longitudes = tracks.map(track => track.longitude);
  const latitudes = tracks.map(track => track.latitude);
  const minLongitude = Math.min(...longitudes);
  const maxLongitude = Math.max(...longitudes);
  const minLatitude = Math.min(...latitudes);
  const maxLatitude = Math.max(...latitudes);
  const longitudeRange = maxLongitude - minLongitude || 1;
  const latitudeRange = maxLatitude - minLatitude || 1;

  return (
    <div className="mt-3 overflow-x-auto">
      <svg
        role="img"
        aria-label="Recorded GPS detections by longitude and latitude"
        viewBox={`0 0 ${width} ${height}`}
        className="min-w-[560px] w-full"
      >
        <line x1={inset} y1={height - inset} x2={width - inset} y2={height - inset} stroke="#454a50" />
        <line x1={inset} y1={inset} x2={inset} y2={height - inset} stroke="#454a50" />
        <text x={inset} y={height - 10} fill="#969da5" fontSize="10">
          {minLongitude.toFixed(5)}
        </text>
        <text x={width - inset} y={height - 10} fill="#969da5" fontSize="10" textAnchor="end">
          {maxLongitude.toFixed(5)}
        </text>
        <text x={inset + 4} y={inset + 10} fill="#969da5" fontSize="10">
          {maxLatitude.toFixed(5)}
        </text>
        <text x={inset + 4} y={height - inset - 4} fill="#969da5" fontSize="10">
          {minLatitude.toFixed(5)}
        </text>
        {tracks.map((track, index) => {
          const x = maxLongitude === minLongitude
            ? width / 2
            : inset + ((track.longitude - minLongitude) / longitudeRange) * (width - inset * 2);
          const y = maxLatitude === minLatitude
            ? height / 2
            : height - inset - ((track.latitude - minLatitude) / latitudeRange) * (height - inset * 2);
          const color = track.source === 'ir' ? IR_COLOR : RGB_COLOR;
          return (
            <circle key={`${track.frame}-${track.track_id}-${track.source}-${index}`} cx={x} cy={y} r="4" fill={color} fillOpacity="0.8">
              <title>
                {track.source.toUpperCase()} · track {track.track_id} · frame {track.frame} · {(track.confidence * 100).toFixed(1)}%
              </title>
            </circle>
          );
        })}
      </svg>
    </div>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-2">
      <span className="h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
      {label}
    </span>
  );
}
