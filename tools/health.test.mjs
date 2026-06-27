import assert from "node:assert/strict";
import test from "node:test";

import { metricHealth } from "../app/src/health.js";

const flat = (v, n = 20) => Array(n).fill(v);

test("échauffement : pas de verdict avant d'être établi", () => {
  assert.equal(metricHealth("sprint_hits", [0.02, 0.03, 0.02]), null);
  assert.equal(metricHealth("hit_rate", [1, 2]), null);
});

test("sprint_hits : mort = bad, moyen = watch, sain = ok", () => {
  assert.equal(metricHealth("sprint_hits", flat(0.05)), "bad");
  assert.equal(metricHealth("sprint_hits", flat(0.18)), "watch");
  assert.equal(metricHealth("sprint_hits", flat(0.45)), "ok");
});

test("entropy : effondrement sous 50% de la médiane = bad", () => {
  assert.equal(metricHealth("entropy", [...flat(4.0, 30), 1.5]), "bad");
  assert.equal(metricHealth("entropy", [...flat(4.0, 30), 3.6]), null);
});

test("kl : fenêtre saine, trop grand = bad, gelé = watch", () => {
  assert.equal(metricHealth("approx_kl", flat(0.009)), "ok");
  assert.equal(metricHealth("approx_kl", flat(0.05)), "bad");
  assert.equal(metricHealth("approx_kl", flat(0.0001)), "watch");
});

test("eval_bot : zéro prolongé = watch, décollage = ok", () => {
  assert.equal(metricHealth("eval_bot", flat(0.0, 10)), "watch");
  assert.equal(metricHealth("eval_bot", [...flat(0.0, 6), 0.01, 0.04]), "ok");
  assert.equal(metricHealth("eval_bot", flat(0.7, 5)), "ok");
});

test("eval_first : toute régression vs random est grave", () => {
  assert.equal(metricHealth("eval_first", flat(1.0, 5)), "ok");
  assert.equal(metricHealth("eval_first", flat(0.8, 5)), "bad");
});

test("sps : chute de moitié vs médiane = watch", () => {
  assert.equal(metricHealth("sps", [...flat(150000, 30), 60000]), "watch");
  assert.equal(metricHealth("sps", flat(150000, 30)), null);
});
