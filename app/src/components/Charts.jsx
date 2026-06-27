// Graphiques SVG maison — pas de dépendance, esthétique hairline.

function path(values, w, h, pad = 2) {
  if (!values || values.length < 2) return "";
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const sx = (i) => pad + (i / (values.length - 1)) * (w - pad * 2);
  const sy = (v) => h - pad - ((v - min) / span) * (h - pad * 2);
  return values.map((v, i) => `${i ? "L" : "M"}${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join(" ");
}

export function Sparkline({ values, width = 180, height = 44, health }) {
  const d = path(values, width, height);
  return (
    <svg className={"spark" + (health ? " " + health : "")}
         width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      <defs>
        <linearGradient id="sparkfill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="rgba(94,139,255,0.22)" />
          <stop offset="100%" stopColor="rgba(94,139,255,0)" />
        </linearGradient>
      </defs>
      {d && <path className="fill" d={`${d} L ${width - 2},${height} L 2,${height} Z`} />}
      {d && <path className="line" d={d} />}
    </svg>
  );
}

export function BigChart({ values, label, width = 560, height = 180, color,
                           health }) {
  const d = path(values, width, height, 8);
  const min = values?.length ? Math.min(...values) : 0;
  const max = values?.length ? Math.max(...values) : 0;
  return (
    <svg className={"bigchart" + (health ? " " + health : "")}
         width="100%" height={height} viewBox={`0 0 ${width} ${height}`}
         preserveAspectRatio="none">
      {[0.25, 0.5, 0.75].map((f) => (
        <line key={f} className="gridline" x1="0" x2={width} y1={height * f} y2={height * f} />
      ))}
      {d && <path className="line" d={d}
                  style={color && !health ? { stroke: color } : undefined} />}
      <text className="axis" x="8" y="14">{max.toFixed(2)}</text>
      <text className="axis" x="8" y={height - 6}>{min.toFixed(2)}</text>
      {label && <text className="axis" x={width - 8} y="14" textAnchor="end">{label}</text>}
    </svg>
  );
}
