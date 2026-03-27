import { useMemo, useState } from "react";

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [rawSpectrum, setRawSpectrum] = useState(null);

  const backendUrl = "http://127.0.0.1:8000/api/invert";

  const runInversion = async (file) => {
    if (!file) {
      setError("Please choose a spectrum CSV file first.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(backendUrl, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "Backend inference failed.");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(
        err?.message ||
          "Could not reach the Python backend. Make sure the local API server is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const parseLocalSpectrum = async (file) => {
    if (!file) {
      setRawSpectrum(null);
      return;
    }

    try {
      const text = await file.text();
      const lines = text.split(/\r?\n/).filter((line) => line.trim());
      const rows = [];

      for (const line of lines) {
        const parts = line.split(/[;,\t ]+/).filter(Boolean);
        if (parts.length < 2) continue;
        const x = Number(parts[0]);
        const y = Number(parts[1]);
        if (Number.isFinite(x) && Number.isFinite(y)) rows.push([x, y]);
      }

      if (!rows.length) {
        setRawSpectrum(null);
        return;
      }

      const omegaScaled = rows.map((r) => r[0]);
      const xsTHz = omegaScaled.map((v) => (v * 1e14) / (2 * Math.PI) / 1e12);
      const ys = rows.map((r) => r[1]);
      const yMax = Math.max(...ys, 1e-12);
      const ysNorm = ys.map((v) => Math.max(v, 0) / yMax);
      setRawSpectrum({ xs: xsTHz, ys: ysNorm, rawYMax: yMax });
    } catch {
      setRawSpectrum(null);
    }
  };

  const onFileChange = (file) => {
    if (!file) return;
    setSelectedFile(file);
    setResult(null);
    setError("");
    parseLocalSpectrum(file);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    event.stopPropagation();
    setDragActive(false);
    const file = event.dataTransfer.files?.[0];
    onFileChange(file);
  };

  const handleDrag = (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (event.type === "dragenter" || event.type === "dragover") {
      setDragActive(true);
    } else if (event.type === "dragleave") {
      setDragActive(false);
    }
  };

  const summary = useMemo(() => {
    if (!result) return [];
    return [
      ["Model", result.model_name],
      ["Band", `${result.band_min_thz}–${result.band_max_thz} THz`],
      ["Input points", String(result.raw_points)],
      ["Resampled", String(result.resampled_points)],
      ["Profile max", result.profile_max.toFixed(4)],
      ["Profile min", result.profile_min.toFixed(4)],
    ];
  }, [result]);

  const rawXs = rawSpectrum?.xs || [];
  const rawYs = rawSpectrum?.ys || [];
  const profileXs = result?.z_um || [];
  const profileYs = result?.profile_norm || [];

  const modelInfo = [
    ["Backend endpoint", "127.0.0.1:8000/api/invert"],
    ["Input unit", "10^14 rad/s → THz"],
    ["Inference band", "50–230 THz"],
    ["Display spectrum", "0–230 THz"],
    ["Selected file", selectedFile ? selectedFile.name : "None"],
    ["Backend status", loading ? "Running" : result ? "Connected" : "Idle"],
  ];

  return (
    <div style={styles.page}>
      <div style={styles.app}>
        <section style={styles.topGrid}>
          <div style={{ ...styles.panel, ...styles.titlePanel }}>
            <div style={styles.badge}>Local spectrum → PyTorch inversion</div>
            <h1 style={styles.title}>Spectrum-to-beam inversion on your own machine</h1>
            <p style={styles.subtitle}>
              Upload a local spectrum CSV, send it to your Python backend, load the saved model,
              and visualize the reconstructed electron-bunch profile directly in the browser.
            </p>
          </div>

          <div style={styles.panel}>
            <div style={styles.panelTitle}>Drag or choose spectrum file</div>
            <div
              style={{
                ...styles.uploadBox,
                ...(dragActive ? styles.uploadBoxActive : {}),
              }}
              onDragEnter={handleDrag}
              onDragOver={handleDrag}
              onDragLeave={handleDrag}
              onDrop={handleDrop}
            >
              <div style={styles.uploadTitle}>Drop your CSV here</div>
              <div style={styles.uploadText}>Two numeric columns expected</div>
              <input
                type="file"
                accept=".csv,text/csv"
                onChange={(e) => onFileChange(e.target.files?.[0])}
                style={{ marginTop: 10 }}
              />
            </div>
            <div style={styles.actionRow}>
              <button
                style={styles.primaryButton}
                onClick={() => runInversion(selectedFile)}
                disabled={loading}
              >
                {loading ? "Running..." : "Run inversion"}
              </button>
              <button
                style={styles.secondaryButtonLight}
                onClick={() => {
                  setSelectedFile(null);
                  setRawSpectrum(null);
                  setResult(null);
                  setError("");
                }}
              >
                Clear
              </button>
            </div>
          </div>

          <div style={styles.panel}>
            <div style={styles.panelTitle}>Model parameters & data</div>
            <div style={styles.infoList}>
              {modelInfo.map(([label, value]) => (
                <div key={label} style={styles.infoRow}>
                  <span style={styles.infoLabel}>{label}</span>
                  <span style={styles.infoValue}>{value}</span>
                </div>
              ))}
            </div>
          </div>

          <div style={styles.panel}>
            <div style={styles.panelTitle}>Summary</div>
            <div style={styles.summaryGrid}>
              {summary.length ? (
                summary.map(([label, value]) => (
                  <div key={label} style={styles.summaryItem}>
                    <div style={styles.summaryLabel}>{label}</div>
                    <div style={styles.summaryValue}>{value}</div>
                  </div>
                ))
              ) : (
                <div style={styles.emptyHint}>Run the backend inference to fill this panel.</div>
              )}
            </div>
            {error ? <div style={styles.errorBox}>{error}</div> : null}
          </div>
        </section>

        <section style={styles.bottomGrid}>
          <div style={styles.chartCard}>
            <div style={styles.cardTitle}>Measured spectrum (original F)</div>
            <div style={styles.cardSub}>
              The uploaded first column is interpreted as <b>10^14 rad/s</b> and converted to <b>THz</b>.
              The blue shaded region marks <b>0–50 THz</b>, which is truncated before inference.
            </div>
            <div style={styles.chartWrap}>
              {rawXs.length ? (
                <AxisChart
                  xs={rawXs}
                  ys={rawYs}
                  xLabel="Frequency (THz)"
                  yLabel="Normalized intensity"
                  shadeToX={50}
                  lineColor="#1d4ed8"
                  xDomain={[0, 230]}
                  xTicks={[0, 50, 100, 150, 200, 230]}
                  yTicks={[0, 0.25, 0.5, 0.75, 1.0]}
                />
              ) : (
                <div style={styles.chartEmpty}>Upload a local CSV to preview the original spectrum.</div>
              )}
            </div>
          </div>

          <div style={styles.chartCard}>
            <div style={styles.cardTitle}>Reconstructed electron-bunch profile</div>
            <div style={styles.cardSub}>Direct model output after local inference.</div>
            <div style={styles.chartWrap}>
              {profileXs.length ? (
                <AxisChart
                  xs={profileXs}
                  ys={profileYs}
                  xLabel="z (μm)"
                  yLabel="Normalized profile"
                  fillArea
                  lineColor="#0f4ddf"
                  yTicks={[0, 0.25, 0.5, 0.75, 1.0]}
                />
              ) : (
                <div style={styles.chartEmpty}>Profile will appear here after inversion.</div>
              )}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

function AxisChart({
  xs,
  ys,
  xLabel,
  yLabel,
  shadeToX,
  fillArea = false,
  lineColor = "#2563eb",
  xDomain,
  xTicks,
  yTicks,
}) {
  const width = 760;
  const height = 360;
  const left = 72;
  const right = 24;
  const top = 22;
  const bottom = 52;
  const plotW = width - left - right;
  const plotH = height - top - bottom;

  const xMin = xDomain ? xDomain[0] : Math.min(...xs);
  const xMax = xDomain ? xDomain[1] : Math.max(...xs);
  const yMin = 0;
  const yMax = Math.max(...ys, 1e-8);

  const mapX = (x) => left + ((x - xMin) / Math.max(xMax - xMin, 1e-8)) * plotW;
  const mapY = (y) => top + plotH - ((y - yMin) / Math.max(yMax - yMin, 1e-8)) * plotH;

  const points = xs.map((x, i) => `${mapX(x)},${mapY(ys[i])}`).join(" ");
  const areaPath = `M ${mapX(xs[0])},${top + plotH} L ${points.replace(/ /g, " L ")} L ${mapX(
    xs[xs.length - 1]
  )},${top + plotH} Z`;

  const finalXTicks = xTicks || makeNiceXTicks(xMin, xMax);
  const finalYTicks = yTicks || makeTicks(0, yMax, 4);

  let shadeRect = null;
  if (typeof shadeToX === "number") {
    const clamped = Math.max(xMin, Math.min(shadeToX, xMax));
    const x0 = mapX(xMin);
    const x1 = mapX(clamped);
    shadeRect = (
      <>
        <rect
          x={x0}
          y={top}
          width={Math.max(0, x1 - x0)}
          height={plotH}
          fill="rgba(37, 99, 235, 0.16)"
        />
        <text x={(x0 + x1) / 2} y={top + 22} textAnchor="middle" fontSize="12" fill="#1d4ed8">
          truncated for inference
        </text>
      </>
    );
  }

  return (
    <svg viewBox={`0 0 ${width} ${height}`} style={styles.svg}>
      <rect x="0" y="0" width={width} height={height} fill="#f8fafc" rx="18" />
      {shadeRect}

      {finalXTicks.map((tick) => {
        const x = mapX(tick);
        return (
          <g key={`x-${tick}`}>
            <line x1={x} y1={top} x2={x} y2={top + plotH} stroke="#e2e8f0" strokeDasharray="3 4" />
            <text x={x} y={top + plotH + 22} textAnchor="middle" fontSize="12" fill="#475569">
              {formatTick(tick)}
            </text>
          </g>
        );
      })}

      {finalYTicks.map((tick) => {
        const y = mapY(tick);
        return (
          <g key={`y-${tick}`}>
            <line x1={left} y1={y} x2={left + plotW} y2={y} stroke="#e2e8f0" strokeDasharray="3 4" />
            <text x={left - 10} y={y + 4} textAnchor="end" fontSize="12" fill="#475569">
              {formatTick(tick)}
            </text>
          </g>
        );
      })}

      <line x1={left} y1={top + plotH} x2={left + plotW} y2={top + plotH} stroke="#94a3b8" />
      <line x1={left} y1={top} x2={left} y2={top + plotH} stroke="#94a3b8" />

      {fillArea ? <path d={areaPath} fill="rgba(37, 99, 235, 0.20)" /> : null}
      <polyline fill="none" stroke={lineColor} strokeWidth="3" points={points} />

      <text x={left + plotW / 2} y={height - 10} textAnchor="middle" fontSize="14" fill="#334155">
        {xLabel}
      </text>
      <text
        x={18}
        y={top + plotH / 2}
        textAnchor="middle"
        fontSize="14"
        fill="#334155"
        transform={`rotate(-90 18 ${top + plotH / 2})`}
      >
        {yLabel}
      </text>
    </svg>
  );
}

function makeNiceXTicks(min, max) {
  const span = max - min;
  if (span <= 60) return makeRoundedTicks(min, max, 10);
  if (span <= 120) return makeRoundedTicks(min, max, 20);
  return makeRoundedTicks(min, max, 50);
}

function makeRoundedTicks(min, max, step) {
  const ticks = [];
  const start = Math.ceil(min / step) * step;
  const end = Math.floor(max / step) * step;
  if (min === 0) ticks.push(0);
  for (let v = start; v <= end; v += step) {
    if (ticks.length === 0 || Math.abs(ticks[ticks.length - 1] - v) > 1e-9) ticks.push(v);
  }
  if (ticks[ticks.length - 1] !== max) ticks.push(max);
  return ticks;
}

function makeTicks(min, max, count) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (Math.abs(max - min) < 1e-12) return [min];
  const ticks = [];
  for (let i = 0; i <= count; i += 1) {
    ticks.push(min + ((max - min) * i) / count);
  }
  return ticks;
}

function formatTick(v) {
  if (Math.abs(v) >= 100 || Number.isInteger(v)) return String(Math.round(v));
  if (Math.abs(v) >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

const styles = {
  page: {
    height: "100vh",
    overflow: "hidden",
    background: "#eef2f7",
    color: "#0f172a",
    fontFamily:
      'Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
  },
  app: {
    height: "100vh",
    display: "grid",
    gridTemplateRows: "auto 1fr",
    gap: 14,
    padding: 14,
    boxSizing: "border-box",
  },
  topGrid: {
    display: "grid",
    gridTemplateColumns: "1.25fr 1fr 1fr 1fr",
    gap: 14,
    minHeight: 0,
  },
  panel: {
    background: "white",
    borderRadius: 22,
    padding: 16,
    boxShadow: "0 1px 2px rgba(15,23,42,0.06)",
    minWidth: 0,
    overflow: "hidden",
  },
  titlePanel: {
    background: "linear-gradient(135deg, #0f172a, #1e3a8a)",
    color: "white",
  },
  badge: {
    display: "inline-block",
    background: "rgba(255,255,255,0.12)",
    padding: "5px 10px",
    borderRadius: 999,
    fontSize: 12,
    marginBottom: 10,
  },
  title: {
    margin: 0,
    fontSize: 30,
    lineHeight: 1.08,
  },
  subtitle: {
    margin: "10px 0 0 0",
    color: "#dbeafe",
    fontSize: 14,
    lineHeight: 1.55,
  },
  panelTitle: {
    fontSize: 18,
    fontWeight: 700,
    marginBottom: 10,
  },
  uploadBox: {
    border: "1.5px dashed #cbd5e1",
    background: "#f8fafc",
    borderRadius: 16,
    padding: 14,
  },
  uploadBoxActive: {
    background: "#eff6ff",
    borderColor: "#60a5fa",
  },
  uploadTitle: {
    fontWeight: 700,
    fontSize: 15,
  },
  uploadText: {
    fontSize: 13,
    color: "#64748b",
    marginTop: 4,
  },
  actionRow: {
    display: "flex",
    gap: 10,
    marginTop: 12,
  },
  primaryButton: {
    padding: "11px 16px",
    border: "none",
    background: "#0f172a",
    color: "white",
    borderRadius: 12,
    cursor: "pointer",
    fontWeight: 700,
  },
  secondaryButtonLight: {
    padding: "11px 16px",
    border: "1px solid #cbd5e1",
    background: "white",
    color: "#0f172a",
    borderRadius: 12,
    cursor: "pointer",
    fontWeight: 700,
  },
  infoList: {
    display: "grid",
    gap: 8,
  },
  infoRow: {
    display: "flex",
    justifyContent: "space-between",
    gap: 12,
    padding: "7px 0",
    borderBottom: "1px solid #eef2f7",
    fontSize: 14,
  },
  infoLabel: {
    color: "#64748b",
  },
  infoValue: {
    fontWeight: 700,
    textAlign: "right",
    wordBreak: "break-word",
  },
  summaryGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
    gap: 10,
  },
  summaryItem: {
    border: "1px solid #e2e8f0",
    borderRadius: 14,
    padding: 10,
    background: "#f8fafc",
    minWidth: 0,
  },
  summaryLabel: {
    fontSize: 12,
    color: "#64748b",
  },
  summaryValue: {
    marginTop: 5,
    fontSize: 14,
    fontWeight: 700,
    wordBreak: "break-word",
  },
  emptyHint: {
    color: "#64748b",
    fontSize: 14,
    lineHeight: 1.5,
    padding: "8px 2px",
  },
  errorBox: {
    marginTop: 12,
    border: "1px solid #fecaca",
    background: "#fef2f2",
    color: "#991b1b",
    borderRadius: 14,
    padding: 12,
    fontSize: 13,
    lineHeight: 1.5,
  },
  bottomGrid: {
    minHeight: 0,
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: 14,
  },
  chartCard: {
    minHeight: 0,
    background: "white",
    borderRadius: 22,
    padding: 16,
    display: "grid",
    gridTemplateRows: "auto auto 1fr",
    boxShadow: "0 1px 2px rgba(15,23,42,0.06)",
  },
  cardTitle: {
    fontSize: 20,
    fontWeight: 700,
  },
  cardSub: {
    color: "#64748b",
    fontSize: 14,
    marginTop: 4,
    lineHeight: 1.45,
  },
  chartWrap: {
    minHeight: 0,
    display: "flex",
    alignItems: "stretch",
    marginTop: 10,
  },
  svg: {
    width: "100%",
    height: "100%",
    display: "block",
  },
  chartEmpty: {
    width: "100%",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "#f8fafc",
    borderRadius: 18,
    color: "#64748b",
    fontSize: 14,
  },
};