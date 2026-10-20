import { useEffect, useState } from 'react';
import { API_BASE_URL } from '../api';

type Scenario = {
  environment: string;
  weather: string;
  timeOfDay: string;
  targetDensity: string;
};

type ModelStatus = {
  status: string;
};

type BackendStatus = {
  rgb_model: ModelStatus;
  ir_model: ModelStatus;
  radar_model: ModelStatus;
};

const DEFAULT_SCENARIO: Scenario = {
  environment: 'Urban Center (High Clutter)',
  weather: 'Clear',
  timeOfDay: 'Daylight',
  targetDensity: 'Single Target',
};

export default function InstructorUI() {
  const [scenario, setScenario] = useState(DEFAULT_SCENARIO);
  const [modelStatus, setModelStatus] = useState<BackendStatus | null>(null);
  const [statusError, setStatusError] = useState('');

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${API_BASE_URL}/api/status`, { signal: controller.signal })
      .then(async response => {
        if (!response.ok) throw new Error(`Model status request failed (${response.status})`);
        return response.json() as Promise<BackendStatus>;
      })
      .then(setModelStatus)
      .catch((error: unknown) => {
        if (!(error instanceof DOMException && error.name === 'AbortError')) {
          setStatusError(error instanceof Error ? error.message : 'Could not load model status.');
        }
      });
    return () => controller.abort();
  }, []);

  const updateScenario = (key: keyof Scenario, value: string) => {
    setScenario(current => ({ ...current, [key]: value }));
  };

  const exportScenario = () => {
    const file = new Blob([JSON.stringify(scenario, null, 2)], { type: 'application/json' });
    const downloadUrl = URL.createObjectURL(file);
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = 'trinetra-scenario.json';
    link.click();
    URL.revokeObjectURL(downloadUrl);
  };

  return (
    <div className="h-full w-full overflow-y-auto bg-gis-bg p-6 text-[13px] md:p-8">
      <div className="mx-auto max-w-[1100px] space-y-6">
        <header>
          <h1 className="text-[24px] font-medium text-gis-accent">Instructor Setup</h1>
          <p className="mt-1 text-gis-muted">
            Configure and export scenario metadata. These settings do not modify detector inputs.
          </p>
        </header>

        <div className="grid gap-6 lg:grid-cols-12">
          <section className="space-y-6 rounded-2xl border border-gis-border bg-gis-panel p-6 lg:col-span-8">
            <div>
              <h2 className="border-b border-gis-border pb-3 text-[16px] font-medium text-gis-accent">
                Scenario Builder
              </h2>
              <div className="mt-5 grid gap-4 sm:grid-cols-2">
                <ControlSelect
                  label="Environment Model"
                  value={scenario.environment}
                  options={['Urban Center (High Clutter)', 'Industrial Complex', 'Open Desert']}
                  onChange={value => updateScenario('environment', value)}
                />
                <ControlSelect
                  label="Weather"
                  value={scenario.weather}
                  options={['Clear', 'Haze', 'Heavy Fog']}
                  onChange={value => updateScenario('weather', value)}
                />
                <ControlSelect
                  label="Time of Day"
                  value={scenario.timeOfDay}
                  options={['Daylight', 'Dusk', 'Night']}
                  onChange={value => updateScenario('timeOfDay', value)}
                />
                <ControlSelect
                  label="Target Density"
                  value={scenario.targetDensity}
                  options={['Single Target', 'Small Swarm (3-5)', 'Large Swarm (10+)']}
                  onChange={value => updateScenario('targetDensity', value)}
                />
              </div>
            </div>

            <div className="border-t border-gis-border pt-5">
              <h3 className="mb-3 text-gis-accent">Loaded model status</h3>
              {statusError && <p role="alert" className="mb-3 text-[#ff8888]">{statusError}</p>}
              <div className="grid gap-3 sm:grid-cols-3">
                <SensorCard label="RGB YOLO11" status={modelStatus?.rgb_model.status ?? 'CONNECTING'} />
                <SensorCard label="IR YOLO11" status={modelStatus?.ir_model.status ?? 'CONNECTING'} />
                <SensorCard label="Radar" status={modelStatus?.radar_model.status ?? 'CONNECTING'} />
              </div>
            </div>

            <div className="flex flex-wrap justify-end gap-3 border-t border-gis-border pt-5">
              <button
                onClick={() => setScenario(DEFAULT_SCENARIO)}
                className="rounded-xl border border-gis-border bg-gis-bg px-4 py-2 text-gis-text hover:bg-gis-panel-light"
              >
                Reset Defaults
              </button>
              <button
                onClick={exportScenario}
                className="rounded-xl bg-gis-active px-5 py-2 font-medium text-white hover:bg-opacity-90"
              >
                Export Scenario JSON
              </button>
            </div>
          </section>

          <aside className="space-y-6 lg:col-span-4">
            <section className="rounded-2xl border border-gis-border bg-gis-panel p-5">
              <h2 className="text-gis-accent">Recorded GPS trajectory</h2>
              <img
                src={`${API_BASE_URL}/api/assets/${encodeURIComponent('TracK GPS Trejectory.jpeg')}`}
                alt="Recorded GPS track trajectory"
                className="mt-3 aspect-[4/3] w-full rounded-xl border border-gis-border bg-[#111] object-contain"
              />
              <p className="mt-2 text-[11px] text-gis-muted">
                Preview from the recorded production run, not a live map.
              </p>
            </section>
            <section className="rounded-2xl border border-gis-border bg-gis-panel p-5">
              <h2 className="text-gis-accent">Scenario notes</h2>
              <div className="mt-3 space-y-2 rounded-xl border border-gis-border bg-[#1a1a1a] p-4 text-[11px] text-gis-muted">
                <p>Selections are saved locally as a JSON file.</p>
                <p>RGB and IR models are available for live webcam and uploaded-image inference.</p>
                <p>Environmental simulation and radar inference are not connected.</p>
              </div>
            </section>
          </aside>
        </div>
      </div>
    </div>
  );
}

function ControlSelect({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
}) {
  return (
    <label className="space-y-1.5">
      <span className="block text-gis-muted">{label}</span>
      <select
        value={value}
        onChange={event => onChange(event.target.value)}
        className="w-full rounded-xl border border-[#333] bg-[#1a1a1a] p-2 text-gis-text focus:border-gis-active focus:outline-none"
      >
        {options.map(option => <option key={option}>{option}</option>)}
      </select>
    </label>
  );
}

function SensorCard({ label, status }: { label: string; status: string }) {
  const isReady = status === 'LOADED';
  return (
    <div className="flex min-w-0 flex-col gap-1 rounded-xl border border-gis-border bg-[#1a1a1a] p-3">
      <span className="text-gis-muted">{label}</span>
      <span className={isReady ? 'text-[#00ff66]' : 'text-[#ffaa00]'}>{status}</span>
    </div>
  );
}
