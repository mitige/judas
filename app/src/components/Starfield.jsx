import { useEffect, useRef } from "react";

/** Voile d'étoiles : dérive lente, scintillement doux, une étoile filante
    rare. Volontairement discret — une présence, pas un spectacle. */
export default function Starfield() {
  const ref = useRef(null);

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas.getContext("2d");
    let raf, w, h, stars, meteor = null;

    const resize = () => {
      w = canvas.width = window.innerWidth * devicePixelRatio;
      h = canvas.height = window.innerHeight * devicePixelRatio;
      stars = Array.from({ length: Math.floor((w * h) / 26000) }, () => ({
        x: Math.random() * w,
        y: Math.random() * h,
        r: Math.random() * 1.1 + 0.2,
        tw: Math.random() * Math.PI * 2,
        sp: 0.008 + Math.random() * 0.02,
      }));
    };
    resize();
    window.addEventListener("resize", resize);

    const tick = (t) => {
      ctx.clearRect(0, 0, w, h);
      for (const s of stars) {
        s.x -= s.sp;
        if (s.x < 0) s.x = w;
        const a = 0.25 + 0.3 * (0.5 + 0.5 * Math.sin(t / 1700 + s.tw));
        ctx.globalAlpha = a;
        ctx.fillStyle = "#bcd2ff";
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fill();
      }
      // étoile filante occasionnelle
      if (!meteor && Math.random() < 0.0006) {
        meteor = { x: Math.random() * w * 0.7 + w * 0.2, y: Math.random() * h * 0.3, life: 1 };
      }
      if (meteor) {
        meteor.x += 9; meteor.y += 4.5; meteor.life -= 0.016;
        ctx.globalAlpha = Math.max(meteor.life, 0) * 0.7;
        const g = ctx.createLinearGradient(meteor.x - 90, meteor.y - 45, meteor.x, meteor.y);
        g.addColorStop(0, "transparent");
        g.addColorStop(1, "#cfe2ff");
        ctx.strokeStyle = g;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(meteor.x - 90, meteor.y - 45);
        ctx.lineTo(meteor.x, meteor.y);
        ctx.stroke();
        if (meteor.life <= 0) meteor = null;
      }
      ctx.globalAlpha = 1;
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return <canvas id="starfield" ref={ref} />;
}
