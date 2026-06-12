/* Marque Judas — éclipse : un disque de nuit, un croissant de lumière.
   Monochrome, hairline, aucun glow — sobre par construction.
   Copie identique dans app/ et viz/ (apps Electron séparées). */

export default function Logo({ size = 22 }) {
  return (
    <svg className="logo" width={size} height={size} viewBox="0 0 24 24"
         fill="none" aria-hidden="true">
      <defs>
        <linearGradient id="j-light" x1="0" y1="1" x2="1" y2="0">
          <stop offset="0" stopColor="#a7c2ee" />
          <stop offset="1" stopColor="#e9ebf1" />
        </linearGradient>
        <mask id="j-eclipse">
          <rect width="24" height="24" fill="black" />
          <circle cx="12" cy="12" r="8.5" fill="white" />
          <circle cx="10.1" cy="13.9" r="8.5" fill="black" />
        </mask>
      </defs>
      <circle cx="12" cy="12" r="8.5" stroke="#cdd6eb" strokeOpacity="0.22"
              strokeWidth="0.8" />
      <rect width="24" height="24" fill="url(#j-light)" mask="url(#j-eclipse)" />
    </svg>
  );
}
