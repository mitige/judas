import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const COLORS = [0x9cc8ff, 0xff7e6b];
const DEG = Math.PI / 180;

/** Vue 3D de l'arène. Lit l'état dans stateRef (muté par le WS, pas de
    re-render React par tick) et interpole entre deux ticks pour un rendu
    fluide à 60 fps quel que soit le TPS du flux. */
export default function Arena3D({ stateRef }) {
  const mountRef = useRef(null);

  useEffect(() => {
    const mount = mountRef.current;
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x05070d, 0.012);

    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 300);
    camera.position.set(16, 13, 24);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.maxPolarAngle = Math.PI * 0.49;
    controls.minDistance = 6;
    controls.maxDistance = 70;

    // ------------------------------------------------------------- arène
    const arenaGroup = new THREE.Group();
    scene.add(arenaGroup);
    let builtSize = null;

    function disposeObject(obj) {
      obj.geometry?.dispose?.();
      const material = obj.material;
      if (Array.isArray(material)) {
        material.forEach((m) => m.dispose?.());
      } else {
        material?.dispose?.();
      }
    }

    function clearGroup(group) {
      for (const child of [...group.children]) {
        group.remove(child);
        child.traverse?.(disposeObject);
      }
    }

    function buildArena(sx, sz) {
      clearGroup(arenaGroup);
      builtSize = `${sx}x${sz}`;
      const grid = new THREE.GridHelper(Math.max(sx, sz), Math.max(sx, sz),
                                        0x223052, 0x131c33);
      grid.scale.set(sx / Math.max(sx, sz), 1, sz / Math.max(sx, sz));
      arenaGroup.add(grid);

      const floor = new THREE.Mesh(
        new THREE.PlaneGeometry(sx, sz),
        new THREE.MeshBasicMaterial({ color: 0x0a0e18, transparent: true, opacity: 0.72 }));
      floor.rotation.x = -Math.PI / 2;
      floor.position.y = -0.01;
      arenaGroup.add(floor);

      const walls = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(sx, 3.2, sz)),
        new THREE.LineBasicMaterial({ color: 0x2a3c66, transparent: true, opacity: 0.6 }));
      walls.position.y = 1.6;
      arenaGroup.add(walls);

      // halo d'horizon sous l'arène
      const halo = new THREE.Mesh(
        new THREE.RingGeometry(Math.max(sx, sz) * 0.72, Math.max(sx, sz) * 1.45, 64),
        new THREE.MeshBasicMaterial({ color: 0x16223f, transparent: true,
                                      opacity: 0.35, side: THREE.DoubleSide }));
      halo.rotation.x = -Math.PI / 2;
      halo.position.y = -0.04;
      arenaGroup.add(halo);
    }
    buildArena(18, 18);

    // ------------------------------------------------------------ joueurs
    const players = COLORS.map((color) => {
      const g = new THREE.Group();
      const body = new THREE.Mesh(
        new THREE.BoxGeometry(0.6, 1.8, 0.6),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.1 }));
      body.position.y = 0.9;
      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(0.6, 1.8, 0.6)),
        new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95 }));
      edges.position.y = 0.9;
      const lookGeom = new THREE.BufferGeometry().setFromPoints(
        [new THREE.Vector3(0, 1.62, 0), new THREE.Vector3(0, 1.62, 1.6)]);
      const look = new THREE.Line(lookGeom,
        new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.5 }));
      const glow = new THREE.Mesh(
        new THREE.CircleGeometry(0.55, 32),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.12 }));
      glow.rotation.x = -Math.PI / 2;
      glow.position.y = 0.005;
      g.add(body, edges, look, glow);
      scene.add(g);
      return { g, body, edges, look, glow, prevHurt: 0, trail: [] };
    });

    // traînées de déplacement (points qui s'estompent)
    const trailGroup = new THREE.Group();
    scene.add(trailGroup);

    // anneaux d'impact
    const rings = [];
    function spawnRing(x, y, z, color) {
      const m = new THREE.Mesh(
        new THREE.RingGeometry(0.25, 0.34, 40),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9,
                                      side: THREE.DoubleSide }));
      m.rotation.x = -Math.PI / 2;
      m.position.set(x, y + 0.06, z);
      scene.add(m);
      rings.push({ m, life: 1 });
    }

    function clearDynamicVisuals() {
      players.forEach((P, i) => {
        for (const dot of P.trail) {
          trailGroup.remove(dot);
          dot.geometry.dispose();
          dot.material.dispose();
        }
        P.trail = [];
        P.prevHurt = 0;
        P.body.material.opacity = 0.1;
        P.edges.material.color.set(COLORS[i]);
      });
      while (rings.length) {
        const r = rings.pop();
        scene.remove(r.m);
        r.m.geometry.dispose();
        r.m.material.dispose();
      }
    }

    // ------------------------------------------------------------- resize
    function resize() {
      const w = mount.clientWidth, h = mount.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(mount);

    // ------------------------------------------------------------ animate
    let raf;
    const lerp = (a, b, t) => a + (b - a) * t;
    const lerpAngle = (a, b, t) => {
      let d = ((b - a) % 360 + 540) % 360 - 180;
      return a + d * t;
    };
    let lastResetSeq = stateRef.current?.resetSeq ?? 0;

    function animate() {
      raf = requestAnimationFrame(animate);
      const S = stateRef.current;
      const resetSeq = S.resetSeq ?? 0;
      if (resetSeq !== lastResetSeq) {
        lastResetSeq = resetSeq;
        clearDynamicVisuals();
      }
      const cur = S.cur, prev = S.prev || cur;

      if (cur?.players) {
        const { sx, sz } = cur.arena;
        if (`${sx}x${sz}` !== builtSize) buildArena(sx, sz);

        const span = Math.max(S.tCur - S.tPrev, 16);
        const alpha = Math.min((performance.now() - S.tCur) / span, 1);

        cur.players.forEach((p, i) => {
          const q = prev.players?.[i] ?? p;
          const teleport = Math.abs(p.x - q.x) + Math.abs(p.z - q.z) > 4; // reset
          const x = (teleport ? p.x : lerp(q.x, p.x, alpha)) - sx / 2;
          const y = teleport ? p.y : lerp(q.y, p.y, alpha);
          const z = (teleport ? p.z : lerp(q.z, p.z, alpha)) - sz / 2;
          const yaw = teleport ? p.yaw : lerpAngle(q.yaw, p.yaw, alpha);
          const pitch = teleport ? p.pitch : lerp(q.pitch, p.pitch, alpha);

          const P = players[i];
          P.g.position.set(x, y, z);
          P.g.rotation.y = -yaw * DEG;
          // ligne de visée (pitch appliqué localement)
          const cp = Math.cos(pitch * DEG);
          P.look.geometry.setFromPoints([
            new THREE.Vector3(0, 1.62, 0),
            new THREE.Vector3(0, 1.62 - Math.sin(pitch * DEG) * 1.6, cp * 1.6),
          ]);

          // sprint -> halo au sol plus présent
          P.glow.material.opacity = p.sprint ? 0.3 : 0.1;

          // swing -> pulse du corps
          if (p.swing) P.body.material.opacity = 0.32;
          P.body.material.opacity = Math.max(0.1, P.body.material.opacity * 0.88);

          // hit reçu -> anneau d'impact + flash des arêtes
          if (p.hurt >= 19 && P.prevHurt < 19) {
            spawnRing(x, y, z, COLORS[1 - i]);
            P.edges.material.color.set(0xffffff);
          }
          P.edges.material.color.lerp(new THREE.Color(COLORS[i]), 0.12);
          P.prevHurt = p.hurt;

          // traînée
          if (!teleport && S.frame % 3 === 0) {
            const dot = new THREE.Mesh(
              new THREE.SphereGeometry(0.035, 6, 6),
              new THREE.MeshBasicMaterial({ color: COLORS[i], transparent: true,
                                            opacity: 0.5 }));
            dot.position.set(x, y + 0.05, z);
            trailGroup.add(dot);
            P.trail.push(dot);
            if (P.trail.length > 40) {
              const old = P.trail.shift();
              trailGroup.remove(old);
              old.geometry.dispose();
              old.material.dispose();
            }
          }
          P.trail.forEach((d) => { d.material.opacity *= 0.965; });
        });
      }

      for (let i = rings.length - 1; i >= 0; i--) {
        const r = rings[i];
        r.life -= 0.04;
        r.m.scale.setScalar(1 + (1 - r.life) * 5);
        r.m.material.opacity = Math.max(r.life, 0) * 0.9;
        if (r.life <= 0) {
          scene.remove(r.m);
          r.m.geometry.dispose();
          r.m.material.dispose();
          rings.splice(i, 1);
        }
      }

      S.frame = (S.frame || 0) + 1;
      controls.update();
      renderer.render(scene, camera);
    }
    animate();

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      clearDynamicVisuals();
      clearGroup(arenaGroup);
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, [stateRef]);

  return <div ref={mountRef} style={{ position: "absolute", inset: 0 }} />;
}
