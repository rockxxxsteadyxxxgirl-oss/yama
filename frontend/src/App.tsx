import L from "leaflet";
import iconRetinaUrl from "leaflet/dist/images/marker-icon-2x.png";
import iconUrl from "leaflet/dist/images/marker-icon.png";
import shadowUrl from "leaflet/dist/images/marker-shadow.png";
import "leaflet/dist/leaflet.css";
import { useEffect, useMemo, useRef, useState } from "react";

L.Icon.Default.mergeOptions({
  iconRetinaUrl,
  iconUrl,
  shadowUrl,
});

type BodyPosition = {
  name: string;
  altitude_deg: number;
  azimuth_deg: number;
};

type SunMoon = {
  sun: BodyPosition;
  moon: BodyPosition;
  moon_phase_deg: number;
  moon_illumination: number;
  sunrise: string | null;
  sunset: string | null;
  moonrise: string | null;
  moonset: string | null;
};

type StarPoint = {
  name: string;
  magnitude: number;
  altitude_deg: number;
  azimuth_deg: number;
};

type HorizonProfile = {
  bearings_deg: number[];
  elevations_deg: number[];
  source: string;
};

type FieldOfView = {
  horizontal_deg: number;
  vertical_deg: number;
};

type ConstellationSegment = {
  start_alt_deg: number;
  start_az_deg: number;
  end_alt_deg: number;
  end_az_deg: number;
};

type Constellation = {
  id: string;
  name: string;
  abbr: string;
  label_alt_deg: number;
  label_az_deg: number;
  segments: ConstellationSegment[];
};

type Snapshot = {
  timestamp_utc: string;
  location: [number, number];
  fov: FieldOfView;
  sunmoon: SunMoon;
  stars: StarPoint[];
  horizon: HorizonProfile;
  constellations: Constellation[];
};

type Preset = {
  name: string;
  lat: number;
  lng: number;
  dtLocal: string;
  focal: number;
  sensor: "full" | "aps-c" | "mft";
  heading: number;
  tilt: number;
};

const apiBase = import.meta.env.VITE_API_BASE_URL ?? "";
const defaultLat = Number(import.meta.env.VITE_DEFAULT_LAT ?? 35.6586);
const defaultLng = Number(import.meta.env.VITE_DEFAULT_LNG ?? 139.7454);
const PRESET_KEY = "yamanoha-presets";
const gradientCard = "rounded-2xl border border-white/10 bg-white/5 backdrop-blur-sm shadow-glow";

// IAU略称 -> 日本語名（主要88星座）
const JP_CONSTELLATION_NAMES: Record<string, string> = {
  AND: "アンドロメダ座",
  ANT: "ポンプ座",
  APS: "ふうちょう座",
  AQL: "わし座",
  AQR: "みずがめ座",
  ARA: "さいだん座",
  ARI: "おひつじ座",
  AUR: "ぎょしゃ座",
  BOO: "うしかい座",
  CAE: "ちょうこくぐ座",
  CAM: "きりん座",
  CAP: "やぎ座",
  CAR: "りゅうこつ座",
  CAS: "カシオペヤ座",
  CEN: "ケンタウルス座",
  CEP: "ケフェウス座",
  CET: "くじら座",
  CHA: "カメレオン座",
  CIR: "らしんばん座",
  CMA: "おおいぬ座",
  CMI: "こいぬ座",
  CNC: "かに座",
  COL: "はと座",
  COM: "かみのけ座",
  CRA: "みなみのかんむり座",
  CRB: "かんむり座",
  CRV: "からす座",
  CRT: "コップ座",
  CRU: "みなみじゅうじ座",
  CYG: "はくちょう座",
  DEL: "いるか座",
  DOR: "かじき座",
  DRA: "りゅう座",
  EQU: "こうま座",
  ERI: "エリダヌス座",
  FOR: "ろ座",
  GEM: "ふたご座",
  GRU: "つる座",
  HER: "ヘルクレス座",
  HOR: "とけい座",
  HYA: "うみへび座",
  HYI: "みずへび座",
  IND: "インディアン座",
  LAC: "とかげ座",
  LEO: "しし座",
  LEP: "うさぎ座",
  LIB: "てんびん座",
  LUP: "おおかみ座",
  LYN: "やまねこ座",
  LYR: "こと座",
  MEN: "テーブルさん座",
  MIC: "けんびきょう座",
  MON: "いっかくじゅう座",
  MUS: "はちどり座",
  NOR: "じょうぎ座",
  OCT: "はちぶんぎ座",
  OPH: "へびつかい座",
  ORI: "オリオン座",
  PAV: "くじゃく座",
  PEG: "ペガスス座",
  PER: "ペルセウス座",
  PHE: "フェニックス座",
  PIC: "がか座",
  PSC: "うお座",
  PSA: "みなみのうお座",
  PUP: "とも座",
  PYX: "コンパス座",
  RET: "レチクル座",
  SCL: "ちょうこくしつ座",
  SCO: "さそり座",
  SCT: "たて座",
  SER: "へび座",
  SEX: "ろくぶんぎ座",
  SGE: "や座",
  SGR: "いて座",
  TAU: "おうし座",
  TEL: "ぼうえんきょう座",
  TRA: "みなみのさんかく座",
  TRI: "さんかく座",
  TUC: "きょしちょう座",
  UMA: "おおぐま座",
  UMI: "こぐま座",
  VEL: "ほ座",
  VIR: "おとめ座",
  VOL: "とびうお座",
  VUL: "こぎつね座",
};

function constellationName(abbr: string, fallback: string) {
  const key = (abbr || "").trim().toUpperCase();
  return JP_CONSTELLATION_NAMES[key] ?? JP_CONSTELLATION_NAMES[fallback?.trim().toUpperCase() ?? ""] ?? abbr;
}

function toIso(dtLocal: string): string {
  const d = new Date(dtLocal);
  return d.toISOString();
}

function nowLocalInput(): string {
  const d = new Date();
  const off = d.getTimezoneOffset();
  const local = new Date(d.getTime() - off * 60000);
  return local.toISOString().slice(0, 16);
}

function shiftLocalIso(input: string, deltaMinutes: number): string {
  const d = new Date(input);
  if (Number.isNaN(d.getTime())) return input;
  d.setMinutes(d.getMinutes() + deltaMinutes);
  const off = d.getTimezoneOffset();
  const local = new Date(d.getTime() - off * 60000);
  return local.toISOString().slice(0, 16);
}

function formatDeg(v: number, digits = 1) {
  return `${v.toFixed(digits)}°`;
}

function formatTime(v: string | null) {
  if (!v) return "-";
  const d = new Date(v);
  return d.toLocaleString();
}

function MoonPhaseLabel({ phase }: { phase: number }) {
  if (phase < 45) return <>新月側</>;
  if (phase < 135) return <>上弦付近</>;
  if (phase < 225) return <>満月側</>;
  if (phase < 315) return <>下弦付近</>;
  return <>新月側</>;
}

function wrapAngle(angle: number) {
  return ((angle + 540) % 360) - 180;
}

function clamp01(v: number) {
  return Math.min(1, Math.max(0, v));
}

function projectAltAz(alt: number, az: number, centerAz: number, centerAlt: number, fov: FieldOfView) {
  const dAz = wrapAngle(az - centerAz);
  const dAlt = alt - centerAlt;
  if (Math.abs(dAz) > fov.horizontal_deg / 2 || Math.abs(dAlt) > fov.vertical_deg / 2) return null;
  const x = 0.5 + dAz / fov.horizontal_deg;
  const y = 0.5 - dAlt / fov.vertical_deg;
  if (x < 0 || x > 1 || y < 0 || y > 1) return null;
  return { x, y, dAz, dAlt };
}

function computeHorizonSlice(
  horizon: HorizonProfile,
  heading: number,
  tilt: number,
  fov: FieldOfView,
  width: number,
  height: number,
) {
  const rows = horizon.bearings_deg.map((bearing, idx) => {
    const dAz = wrapAngle(bearing - heading);
    if (Math.abs(dAz) > fov.horizontal_deg / 2 + 8) return null;
    const relAlt = horizon.elevations_deg[idx] - tilt;
    const x = clamp01(0.5 + dAz / fov.horizontal_deg);
    const y = clamp01(0.5 - relAlt / fov.vertical_deg);
    return { x: x * width, y: y * height, dAz, alt: horizon.elevations_deg[idx] };
  });
  return rows.filter((r): r is NonNullable<typeof r> => Boolean(r)).sort((a, b) => a.dAz - b.dAz);
}

function CompositionPreview({
  snapshot,
  heading,
  tilt,
  showConstellations,
  arMode,
  videoRef,
}: {
  snapshot: Snapshot;
  heading: number;
  tilt: number;
  showConstellations: boolean;
  arMode: boolean;
  videoRef: React.RefObject<HTMLVideoElement | null>;
}) {
  const width = 640;
  const height = 360;
  const fov = snapshot.fov;

  const projectedStars = snapshot.stars
    .map((star) => {
      const projected = projectAltAz(star.altitude_deg, star.azimuth_deg, heading, tilt, fov);
      if (!projected) return null;
      return { ...projected, name: star.name, magnitude: star.magnitude };
    })
    .filter((s): s is NonNullable<typeof s> => Boolean(s))
    .sort((a, b) => a.magnitude - b.magnitude)
    .slice(0, 40);

  const sunProj = projectAltAz(snapshot.sunmoon.sun.altitude_deg, snapshot.sunmoon.sun.azimuth_deg, heading, tilt, fov);
  const moonProj = projectAltAz(
    snapshot.sunmoon.moon.altitude_deg,
    snapshot.sunmoon.moon.azimuth_deg,
    heading,
    tilt,
    fov,
  );

  const horizonPoints = computeHorizonSlice(snapshot.horizon, heading, tilt, fov, width, height);

  const projectedConstellations = useMemo(() => {
    if (!showConstellations) return { segments: [], labels: [] };
    const segments: { x1: number; y1: number; x2: number; y2: number; name: string; abbr: string }[] = [];
    const labels: { x: number; y: number; name: string; abbr: string }[] = [];
    snapshot.constellations.forEach((c) => {
      const abbr = (c.abbr || c.id || "").split(" ").pop() ?? "";
      const name = constellationName(abbr, c.name);
      c.segments.forEach((seg) => {
        const p1 = projectAltAz(seg.start_alt_deg, seg.start_az_deg, heading, tilt, fov);
        const p2 = projectAltAz(seg.end_alt_deg, seg.end_az_deg, heading, tilt, fov);
        if (p1 && p2) {
          segments.push({ x1: p1.x * width, y1: p1.y * height, x2: p2.x * width, y2: p2.y * height, name, abbr: c.abbr });
        }
      });
      const labelPos = projectAltAz(c.label_alt_deg, c.label_az_deg, heading, tilt, fov);
      if (labelPos) {
        labels.push({ x: labelPos.x * width, y: labelPos.y * height, name, abbr: c.abbr });
      }
    });
    return { segments, labels };
  }, [snapshot.constellations, heading, tilt, fov, showConstellations]);

  const cardinalLabels = useMemo(() => {
    const cards = [
      { az: 0, label: "北" },
      { az: 90, label: "東" },
      { az: 180, label: "南" },
      { az: 270, label: "西" },
    ];
    return cards
      .map((c) => {
        const dAz = wrapAngle(c.az - heading);
        if (Math.abs(dAz) > fov.horizontal_deg / 2 + 1) return null;
        const x = clamp01(0.5 + dAz / fov.horizontal_deg) * width;
        return { x, label: c.label };
      })
      .filter((v): v is { x: number; label: string } => Boolean(v));
  }, [heading, fov.horizontal_deg, width]);

  return (
    <div className="rounded-2xl border border-white/10 bg-gradient-to-br from-slate-900/70 via-slate-900/40 to-sky-900/30 p-4 relative overflow-hidden">
      {arMode && (
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="absolute inset-0 h-full w-full object-cover opacity-50"
        />
      )}
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-sky-100/80">
        <div>方位 {heading.toFixed(0)}° (0=北, 90=東)</div>
        <div>仰角 {tilt.toFixed(0)}°</div>
        <div>
          FOV H {snapshot.fov.horizontal_deg.toFixed(1)}° / V {snapshot.fov.vertical_deg.toFixed(1)}°
        </div>
      </div>
      <div className="mt-3 overflow-hidden rounded-xl border border-white/10 bg-slate-900/60 shadow-inner">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full">
          <defs>
            <linearGradient id="skyGradient" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor="#0a1c33" />
              <stop offset="60%" stopColor="#0c1a2b" />
              <stop offset="100%" stopColor="#08101f" />
            </linearGradient>
            <linearGradient id="horizonGround" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor="rgba(56,189,248,0.25)" />
              <stop offset="100%" stopColor="rgba(2,6,23,0.9)" />
            </linearGradient>
          </defs>
          <rect x={0} y={0} width={width} height={height} fill="url(#skyGradient)" />
          <rect x={0} y={height / 2} width={width} height={height / 2} fill="rgba(7,13,26,0.6)" />

          {horizonPoints.length > 1 && (
            <>
              <polyline
                points={horizonPoints.map((p) => `${p.x},${p.y}`).join(" ")}
                fill="none"
                stroke="rgba(125,211,252,0.9)"
                strokeWidth={2}
              />
              <polygon
                points={[`${horizonPoints[0].x},${height}`, ...horizonPoints.map((p) => `${p.x},${p.y}`), `${horizonPoints.at(-1)!.x},${height}`].join(" ")}
                fill="url(#horizonGround)"
                opacity={0.55}
              />
            </>
          )}

          {projectedConstellations.segments.map((seg, idx) => (
            <line
              key={`c-${idx}`}
              x1={seg.x1}
              y1={seg.y1}
              x2={seg.x2}
              y2={seg.y2}
              stroke="rgba(125,211,252,0.55)"
              strokeWidth={1}
            />
          ))}

          {projectedConstellations.labels.slice(0, 20).map((label, idx) => (
            <text
              key={`cl-${idx}`}
              x={label.x + 4}
              y={label.y - 4}
              fontSize={12}
              fontWeight={700}
              fill="#ffffff"
            >
              {label.name || label.abbr}
            </text>
          ))}

          <rect
            x={16}
            y={16}
            width={width - 32}
            height={height - 32}
            stroke="rgba(255,255,255,0.15)"
            strokeWidth={1}
            fill="none"
            strokeDasharray="4 6"
          />

          {moonProj && (
            <g transform={`translate(${moonProj.x * width}, ${moonProj.y * height})`}>
              <circle r={9} fill="#f5f3c4" opacity={0.95} />
              <text x={12} y={4} fontSize={12} fontWeight={700} fill="#ffffff">
                月 {(snapshot.sunmoon.moon_illumination * 100).toFixed(0)}%
              </text>
            </g>
          )}
          {sunProj && (
            <g transform={`translate(${sunProj.x * width}, ${sunProj.y * height})`}>
              <circle r={9} fill="#fbbf24" opacity={0.95} />
              <text x={12} y={4} fontSize={12} fontWeight={700} fill="#ffffff">
                太陽
              </text>
            </g>
          )}

          {projectedStars.map((s, idx) => {
            const size = Math.max(2, 6 - s.magnitude);
            const showLabel = idx < 12;
            return (
              <g key={`${s.name}-${idx}`} transform={`translate(${s.x * width}, ${s.y * height})`}>
                <circle r={size} fill="white" opacity={0.95} />
                {showLabel && (
                  <text
                    x={size + 4}
                    y={4}
                    fontSize={11}
                    fontWeight={700}
                    fill="#ffffff"
                  >
                    恒星 等級 {s.magnitude.toFixed(1)}
                  </text>
                )}
              </g>
            );
          })}

          <line
            x1={width / 2}
            y1={0}
            x2={width / 2}
            y2={height}
            stroke="rgba(255,255,255,0.08)"
            strokeWidth={1}
          />
          <line
            x1={0}
            y1={height / 2}
            x2={width}
            y2={height / 2}
            stroke="rgba(255,255,255,0.08)"
            strokeWidth={1}
          />
          {/* 方位ラベル */}
          {cardinalLabels.map((c, idx) => (
            <text
              key={`card-${idx}`}
              x={c.x}
              y={16}
              textAnchor="middle"
              fontSize={12}
              fontWeight={700}
              fill="#ffffff"
            >
              {c.label}
            </text>
          ))}
        </svg>
      </div>
      <p className="mt-3 text-xs text-sky-100/70">
        ・稜線は周囲の標高差から近似しています（{snapshot.horizon.source}） /
        ・星は赤道座標を Alt/Az に投影し、レンズFOV内のみ表示しています。星座線は IAU 現代星座データを使用。
      </p>
    </div>
  );
}

function HorizonChart({ horizon }: { horizon: HorizonProfile | null }) {
  if (!horizon) return <p className="text-sky-100/70">稜線プロファイルを取得中...</p>;
  const width = 680;
  const height = 200;
  const padding = 20;
  const xs = horizon.bearings_deg;
  const ys = horizon.elevations_deg;
  const maxY = Math.max(10, ...ys.map((y) => Math.max(0, y)));

  const points = xs
    .map((x, i) => {
      const px = padding + (x / 360) * (width - padding * 2);
      const py = padding + ((maxY - ys[i]) / (maxY + 5)) * (height - padding * 2);
      return `${px},${py}`;
    })
    .join(" ");

  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`} className="rounded-lg bg-white/5">
      <defs>
        <linearGradient id="horizonFill" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="rgba(56,189,248,0.35)" />
          <stop offset="100%" stopColor="rgba(15,23,42,0.1)" />
        </linearGradient>
      </defs>
      <rect x={0} y={0} width={width} height={height} fill="url(#horizonFill)" />
      <polyline
        fill="none"
        stroke="rgba(125,211,252,0.9)"
        strokeWidth={2}
        points={points}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <text x={padding} y={height - 6} fill="#cfeafe" fontSize="10">
        方位ごとの最高仰角（OpenTopodataで近似、取得失敗時はフラット）
      </text>
    </svg>
  );
}

function StarsList({ stars }: { stars: StarPoint[] }) {
  if (!stars.length) return <p className="text-sky-100/70">地平線より上にある恒星がありません。</p>;
  const visible = [...stars].sort((a, b) => a.magnitude - b.magnitude).slice(0, 12);
  return (
    <div className="grid gap-3 md:grid-cols-2">
      {visible.map((s) => (
        <div key={s.name} className="rounded-lg border border-white/5 bg-white/5 px-3 py-2">
          <div className="flex items-center justify-between">
            <p className="text-sm font-semibold text-white">{s.name}</p>
            <span className="text-xs text-sky-100/70">mag {s.magnitude.toFixed(1)}</span>
          </div>
          <p className="text-xs text-sky-100/80">
            Alt {formatDeg(s.altitude_deg, 1)} / Az {formatDeg(s.azimuth_deg, 1)}
          </p>
        </div>
      ))}
    </div>
  );
}

function App() {
  const [lat, setLat] = useState(defaultLat);
  const [lng, setLng] = useState(defaultLng);
  const [dtLocal, setDtLocal] = useState(nowLocalInput());
  const [sensor, setSensor] = useState<"full" | "aps-c" | "mft">("full");
  const [focal, setFocal] = useState(24);
  const [heading, setHeading] = useState(180);
  const [tilt, setTilt] = useState(20);
  const [showConstellations, setShowConstellations] = useState(true);
  const [sampleRadiusKm, setSampleRadiusKm] = useState(30);
  const [arMode, setArMode] = useState(false);
  const [arError, setArError] = useState<string | null>(null);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [presetName, setPresetName] = useState("お気に入り構図");
  const [locating, setLocating] = useState(false);

  const mapRef = useRef<HTMLDivElement | null>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const markerRef = useRef<L.Marker | null>(null);
  const areaCircleRef = useRef<L.Circle | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const arStreamRef = useRef<MediaStream | null>(null);
  const orientationHandlerRef = useRef<((e: DeviceOrientationEvent) => void) | null>(null);

  useEffect(() => {
    if (!mapRef.current || mapInstanceRef.current) return;
    const map = L.map(mapRef.current).setView([lat, lng], 7);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OpenStreetMap contributors",
      maxZoom: 16,
    }).addTo(map);
    const marker = L.marker([lat, lng]).addTo(map);
    const circle = L.circle([lat, lng], {
      radius: sampleRadiusKm * 1000,
      color: "rgba(56,189,248,0.95)",
      weight: 1.2,
      fillColor: "rgba(56,189,248,0.38)",
      fillOpacity: 0.5,
    }).addTo(map);
    map.on("click", (e: L.LeafletMouseEvent) => {
      const { lat: newLat, lng: newLng } = e.latlng;
      setLat(Number(newLat.toFixed(5)));
      setLng(Number(newLng.toFixed(5)));
      marker.setLatLng([newLat, newLng]);
    });
    mapInstanceRef.current = map;
    markerRef.current = marker;
    areaCircleRef.current = circle;
  }, []);

  useEffect(() => {
    if (mapInstanceRef.current && markerRef.current) {
      markerRef.current.setLatLng([lat, lng]);
      mapInstanceRef.current.setView([lat, lng]);
      if (areaCircleRef.current) {
        areaCircleRef.current.setLatLng([lat, lng]);
        areaCircleRef.current.setRadius(sampleRadiusKm * 1000);
      }
    }
  }, [lat, lng, sampleRadiusKm]);

  useEffect(() => {
    const run = async () => {
      setLoading(true);
      setError(null);
      try {
        const params = new URLSearchParams({
          lat: lat.toString(),
          lng: lng.toString(),
          dt: toIso(dtLocal),
          focal_length_mm: focal.toString(),
          sensor_format: sensor,
        });
        const base = apiBase || "";
        const res = await fetch(`${base.replace(/\/$/, "")}/api/v1/astro/snapshot?${params.toString()}`);
        if (!res.ok) throw new Error(`API ${res.status}`);
        const json: Snapshot = await res.json();
        setSnapshot(json);
      } catch (err) {
        setError(err instanceof Error ? err.message : "unknown error");
      } finally {
        setLoading(false);
      }
    };
    void run();
  }, [lat, lng, dtLocal, focal, sensor]);

  useEffect(() => {
    const saved = localStorage.getItem(PRESET_KEY);
    if (saved) {
      try {
        const parsed: Preset[] = JSON.parse(saved);
        setPresets(parsed);
      } catch (e) {
        console.error("failed to parse presets", e);
      }
    }
  }, []);

  useEffect(() => {
    localStorage.setItem(PRESET_KEY, JSON.stringify(presets));
  }, [presets]);

  const visibleStars = useMemo(() => (snapshot ? snapshot.stars.filter((s) => s.altitude_deg > 0) : []), [snapshot]);

  const savePreset = () => {
    const name = presetName.trim();
    if (!name) return;
    const newPreset: Preset = { name, lat, lng, dtLocal, focal, sensor, heading, tilt };
    setPresets((prev) => {
      const filtered = prev.filter((p) => p.name !== name);
      return [newPreset, ...filtered];
    });
  };

  const loadPreset = (p: Preset) => {
    setLat(p.lat);
    setLng(p.lng);
    setDtLocal(p.dtLocal);
    setFocal(p.focal);
    setSensor(p.sensor);
    setHeading(p.heading);
    setTilt(p.tilt);
  };

  const deletePreset = (name: string) => {
    setPresets((prev) => prev.filter((p) => p.name !== name));
  };

  // ARモード（カメラ＋方位センサー）
  useEffect(() => {
    if (!arMode) {
      // stop
      if (orientationHandlerRef.current) {
        window.removeEventListener("deviceorientation", orientationHandlerRef.current as any);
        orientationHandlerRef.current = null;
      }
      if (arStreamRef.current) {
        arStreamRef.current.getTracks().forEach((t) => t.stop());
        arStreamRef.current = null;
      }
      setArError(null);
      return;
    }

    setArError(null);

    const handleOrientation = (e: DeviceOrientationEvent) => {
      const compass = (e as any).webkitCompassHeading;
      const alpha = e.alpha;
      let newHeading = heading;
      if (typeof compass === "number") {
        newHeading = compass; // iOS
      } else if (typeof alpha === "number") {
        newHeading = 360 - alpha; // convert to compass-like
      }
      const newTilt = typeof e.beta === "number" ? e.beta : tilt;
      setHeading((prev) => Number.isFinite(newHeading) ? newHeading : prev);
      setTilt((prev) => Number.isFinite(newTilt) ? Math.max(-10, Math.min(85, newTilt)) : prev);
    };
    orientationHandlerRef.current = handleOrientation;

    const enable = async () => {
      try {
        // iOS permission
        const g = (window as any).DeviceOrientationEvent;
        if (g && typeof g.requestPermission === "function") {
          const perm = await g.requestPermission();
          if (perm !== "granted") throw new Error("方位センサーの許可が必要です");
        }
        window.addEventListener("deviceorientation", handleOrientation as any, true);

        if (navigator.mediaDevices?.getUserMedia) {
          const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
          arStreamRef.current = stream;
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            await videoRef.current.play().catch(() => {});
          }
        }
      } catch (err) {
        setArMode(false);
        setArError(err instanceof Error ? err.message : "ARモードを開始できませんでした");
      }
    };

    void enable();

    return () => {
      if (orientationHandlerRef.current) {
        window.removeEventListener("deviceorientation", orientationHandlerRef.current as any);
        orientationHandlerRef.current = null;
      }
      if (arStreamRef.current) {
        arStreamRef.current.getTracks().forEach((t) => t.stop());
        arStreamRef.current = null;
      }
    };
  }, [arMode]);

  return (
    <div className="min-h-screen bg-slate-950/95 px-6 py-8 text-white md:px-10">
      <header className={`${gradientCard} px-6 py-6 md:px-10`}>
        <p className="text-sm uppercase tracking-[0.18em] text-sky-200/80">Yamanoha Simulator (Web Demo)</p>
        <div className="mt-2 flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
          <div>
            <h1 className="text-3xl font-semibold md:text-4xl">撮影前の星空×稜線シミュレーション</h1>
            <p className="mt-2 max-w-3xl text-sm text-sky-100/80 md:text-base">
              任意の撮影地・日時・焦点距離を入れると、その場所から見える山の稜線と夜空（星の位置）を投影します。
              カメラの向きと仰角を指定して、事前に構図をイメージできます。
            </p>
          </div>
          <div className="flex items-center gap-3 rounded-full bg-white/10 px-4 py-2 text-sm text-sky-100">
            <span className="inline-block h-2 w-2 rounded-full bg-emerald-400 shadow-glow" />
            API: {apiBase}
          </div>
        </div>
      </header>

      <main className="mt-6 grid gap-6 lg:grid-cols-[2fr_1fr]">
        <section className={`${gradientCard} p-6 space-y-4`}>
          <div className="grid gap-4 md:grid-cols-2">
            <div>
              <label className="text-xs uppercase tracking-[0.12em] text-sky-200/70">緯度 (deg)</label>
              <input
                type="number"
                value={lat}
                step="0.0001"
                onChange={(e) => setLat(Number(e.target.value))}
                className="mt-1 w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-white"
              />
            </div>
            <div>
              <label className="text-xs uppercase tracking-[0.12em] text-sky-200/70">経度 (deg)</label>
              <input
                type="number"
                value={lng}
                step="0.0001"
                onChange={(e) => setLng(Number(e.target.value))}
                className="mt-1 w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-white"
              />
            </div>
            <div>
              <label className="text-xs uppercase tracking-[0.12em] text-sky-200/70">日時 (ローカル)</label>
              <input
                type="datetime-local"
                value={dtLocal}
                onChange={(e) => setDtLocal(e.target.value)}
                className="mt-1 w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-white"
              />
              <div className="mt-2 flex flex-wrap gap-2 text-xs text-sky-100/80">
                <button
                  onClick={() => setDtLocal(nowLocalInput())}
                  className="rounded-md border border-white/15 bg-white/5 px-2 py-1 hover:border-sky-300"
                >
                  現在時刻
                </button>
                {[{ m: -1440, label: "-1日" }, { m: -60, label: "-1時間" }, { m: -10, label: "-10分" }, { m: -1, label: "-1分" }, { m: 1, label: "+1分" }, { m: 10, label: "+10分" }, { m: 60, label: "+1時間" }, { m: 1440, label: "+1日" }].map((b) => (
                  <button
                    key={b.label}
                    onClick={() => setDtLocal((prev) => shiftLocalIso(prev, b.m))}
                    className="rounded-md border border-white/15 bg-white/5 px-2 py-1 hover:border-sky-300"
                  >
                    {b.label}
                  </button>
                ))}
              </div>
            </div>
            <div className="grid grid-cols-[2fr_1fr] gap-3">
              <div>
                <label className="text-xs uppercase tracking-[0.12em] text-sky-200/70">焦点距離 (mm)</label>
                <input
                  type="range"
                  min={8}
                  max={200}
                  value={focal}
                  onChange={(e) => setFocal(Number(e.target.value))}
                  className="mt-2 w-full"
                />
                <p className="text-xs text-sky-100/70">{focal} mm</p>
              </div>
              <div>
                <label className="text-xs uppercase tracking-[0.12em] text-sky-200/70">センサー</label>
                <select
                  value={sensor}
                  onChange={(e) => setSensor(e.target.value as any)}
                  className="mt-1 w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-white"
                >
                  <option value="full">フルサイズ</option>
                  <option value="aps-c">APS-C</option>
                  <option value="mft">マイクロフォーサーズ</option>
                </select>
              </div>
            </div>
            <div>
              <label className="text-xs uppercase tracking-[0.12em] text-sky-200/70">カメラの向き (0=北)</label>
              <input
                type="range"
                min={0}
                max={359}
                value={heading}
                onChange={(e) => setHeading(Number(e.target.value))}
                className="mt-2 w-full"
              />
              <p className="text-xs text-sky-100/70">{heading.toFixed(0)}°</p>
            </div>
            <div>
              <label className="text-xs uppercase tracking-[0.12em] text-sky-200/70">カメラの仰角</label>
              <input
                type="range"
                min={-10}
                max={85}
                value={tilt}
                onChange={(e) => setTilt(Number(e.target.value))}
                className="mt-2 w-full"
              />
              <p className="text-xs text-sky-100/70">{tilt.toFixed(0)}°</p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-4 rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-sky-50">
            {loading && <span>計算中...</span>}
            {error && <span className="text-rose-200">{error}</span>}
            {!loading && !error && snapshot && (
              <>
                <div>UTC: {snapshot.timestamp_utc}</div>
                <div>
                  FOV H {snapshot.fov.horizontal_deg.toFixed(1)}° / V {snapshot.fov.vertical_deg.toFixed(1)}°
                </div>
                <div>稜線ソース: {snapshot.horizon.source}</div>
                <label className="flex items-center gap-2 text-xs text-sky-100/80">
                  <input
                    type="checkbox"
                    checked={showConstellations}
                    onChange={(e) => setShowConstellations(e.target.checked)}
                    className="accent-sky-400"
                  />
                  星座線・星座名を表示
                </label>
                <div className="flex items-center gap-2 text-xs">
                  <span className="text-sky-100/80">AR</span>
                  <button
                    onClick={() => setArMode((v) => !v)}
                    className={`rounded-md px-3 py-1 ${arMode ? "bg-emerald-500 text-slate-950" : "bg-white/10 text-white"}`}
                  >
                    {arMode ? "ON" : "OFF"}
                  </button>
                  {arError && <span className="text-rose-200">{arError}</span>}
                </div>
              </>
            )}
          </div>

          {snapshot && (
            <CompositionPreview
              snapshot={snapshot}
              heading={heading}
              tilt={tilt}
              showConstellations={showConstellations}
              arMode={arMode}
              videoRef={videoRef}
            />
          )}

          <div className="grid gap-4 md:grid-cols-2">
            <div className="rounded-xl bg-white/5 p-4">
              <p className="text-sm uppercase tracking-[0.12em] text-sky-200/70">太陽 / 月</p>
              {snapshot ? (
                <div className="mt-2 space-y-2 text-sm text-sky-100/80">
                  <div className="flex items-center justify-between">
                    <span>太陽 Alt/Az</span>
                    <span>
                      {formatDeg(snapshot.sunmoon.sun.altitude_deg)} / {formatDeg(snapshot.sunmoon.sun.azimuth_deg)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>月 Alt/Az</span>
                    <span>
                      {formatDeg(snapshot.sunmoon.moon.altitude_deg)} / {formatDeg(snapshot.sunmoon.moon.azimuth_deg)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>月齢・照度</span>
                    <span>
                      {snapshot.sunmoon.moon_phase_deg.toFixed(1)}° / {(snapshot.sunmoon.moon_illumination * 100).toFixed(0)}%{" "}
                      <MoonPhaseLabel phase={snapshot.sunmoon.moon_phase_deg} />
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <p className="text-sky-200/70">日の出 / 日の入</p>
                      <p>{formatTime(snapshot.sunmoon.sunrise)}</p>
                      <p>{formatTime(snapshot.sunmoon.sunset)}</p>
                    </div>
                    <div>
                      <p className="text-sky-200/70">月の出 / 月の入</p>
                      <p>{formatTime(snapshot.sunmoon.moonrise)}</p>
                      <p>{formatTime(snapshot.sunmoon.moonset)}</p>
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-sky-100/70">計算中...</p>
              )}
            </div>

            <div className="rounded-xl bg-white/5 p-4">
              <p className="text-sm uppercase tracking-[0.12em] text-sky-200/70">稜線プロファイル</p>
              <HorizonChart horizon={snapshot?.horizon ?? null} />
            </div>
          </div>

          <div className="rounded-xl bg-white/5 p-4">
            <p className="text-sm uppercase tracking-[0.12em] text-sky-200/70">いま見えている明るい恒星</p>
            {snapshot && <StarsList stars={visibleStars} />}
          </div>
        </section>

        <section className={`${gradientCard} p-5 space-y-4`}>
          <div className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-sky-100/80">
            <p>地図をタップして撮影地点を選ぶと、その方位の稜線シルエットと星空を描画します。</p>
            <p className="text-xs text-sky-200/70">日時を未来に変えると星の位置もリアルタイムで切り替わります。</p>
            <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
              <button
                onClick={() => {
                  if (!navigator.geolocation) return;
                  setLocating(true);
                  navigator.geolocation.getCurrentPosition(
                    (pos) => {
                      setLocating(false);
                      setLat(Number(pos.coords.latitude.toFixed(5)));
                      setLng(Number(pos.coords.longitude.toFixed(5)));
                    },
                    () => setLocating(false),
                    { enableHighAccuracy: true, timeout: 10000 },
                  );
                }}
                className="rounded-md border border-white/20 bg-white/10 px-3 py-1 text-white hover:border-sky-300"
              >
                {locating ? "現在地取得中..." : "現在地を取得"}
              </button>
              <span className="text-sky-200/70">薄い円は地平線サンプリング範囲（~30km）です。</span>
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
              <span className="text-sky-200/70">サンプリング範囲</span>
              <button
                onClick={() => setSampleRadiusKm((v) => Math.max(5, v - 5))}
                className="rounded-md border border-white/20 bg-white/10 px-2 py-1 text-white hover:border-sky-300"
              >
                -5km
              </button>
              <button
                onClick={() => setSampleRadiusKm((v) => Math.min(50, v + 5))}
                className="rounded-md border border-white/20 bg-white/10 px-2 py-1 text-white hover:border-sky-300"
              >
                +5km
              </button>
              <span className="text-white font-semibold">{sampleRadiusKm} km</span>
            </div>
          </div>
          <div className="h-[340px] overflow-hidden rounded-xl border border-white/10">
            <div ref={mapRef} className="h-full w-full" />
          </div>

          <div className="rounded-xl bg-white/5 px-4 py-3 text-sm text-sky-100/80 space-y-3">
            <div>
              <p className="text-xs uppercase tracking-[0.12em] text-sky-200/70">現在のセット</p>
              <p>
                {lat.toFixed(4)}, {lng.toFixed(4)} / {dtLocal.replace("T", " ")} / {focal}mm ({sensor})
              </p>
              {snapshot && (
                <p className="text-xs text-sky-200/70 mt-1">
                  FOV: H {snapshot.fov.horizontal_deg.toFixed(1)}° / V {snapshot.fov.vertical_deg.toFixed(1)}°
                </p>
              )}
            </div>

            <div className="rounded-lg border border-white/10 bg-white/5 p-3">
              <p className="text-xs uppercase tracking-[0.12em] text-sky-200/70">プリセットを保存</p>
              <div className="mt-2 flex flex-col gap-2">
                <input
                  type="text"
                  value={presetName}
                  onChange={(e) => setPresetName(e.target.value)}
                  className="w-full rounded-lg border border-white/10 bg-slate-900/50 px-3 py-2 text-sm text-white"
                  placeholder="例: 冬のオリオン構図"
                />
                <button
                  onClick={savePreset}
                  className="rounded-lg bg-sky-500 px-3 py-2 text-sm font-semibold text-slate-950 hover:bg-sky-400"
                >
                  保存する
                </button>
              </div>
            </div>

            {presets.length > 0 && (
              <div className="rounded-lg border border-white/10 bg-white/5 p-3">
                <p className="text-xs uppercase tracking-[0.12em] text-sky-200/70">保存済みプリセット</p>
                <div className="mt-2 space-y-2 text-sm">
                  {presets.map((p) => (
                    <div key={p.name} className="flex items-center justify-between gap-2 rounded-md bg-white/5 px-2 py-2">
                      <div>
                        <p className="font-semibold text-white">{p.name}</p>
                        <p className="text-[11px] text-sky-200/80">
                          {p.lat.toFixed(2)}, {p.lng.toFixed(2)} / {p.focal}mm / 方位 {p.heading.toFixed(0)}° / 仰角 {p.tilt.toFixed(0)}°
                        </p>
                      </div>
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => loadPreset(p)}
                          className="rounded-md border border-white/20 px-2 py-1 text-xs text-white hover:border-sky-300"
                        >
                          適用
                        </button>
                        <button
                          onClick={() => deletePreset(p.name)}
                          className="rounded-md border border-rose-300/40 px-2 py-1 text-xs text-rose-200 hover:border-rose-200"
                        >
                          削除
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
