// Santé des métriques — encode les seuils de docs/GUIDE.md dans l'UI.
// Retourne "ok" (sain, rendu calme), "watch" (ambre), "bad" (braise),
// ou null (pas de jugement : échauffement ou métrique neutre).

const median = (arr) => {
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
};

const slope = (values, n) => {
  const w = values.slice(-n);
  return w.length >= 4 ? w[w.length - 1] - w[0] : 0;
};

export function metricHealth(key, values) {
  if (!values || values.length < 3) return null;
  const v = values[values.length - 1];
  const recent = values.slice(-40);
  const established = values.length >= 12;   // pas de verdict pendant l'échauffement

  switch (key) {
    case "entropy": {
      // miroir du garde du trainer : effondrement sous 50 % de la médiane
      const med = median(recent);
      return med > 0 && v < 0.5 * med ? "bad" : null;
    }
    case "approx_kl":
      return v > 0.04 ? "bad"
        : v > 0.02 || v < 0.0005 ? "watch" : "ok";
    case "clip_frac":
      return v > 0.35 ? "bad" : v < 0.01 ? "watch" : "ok";
    case "loss_v": {
      const med = median(recent);
      return med > 0 && v > 3 * med ? "watch" : null;
    }
    case "league_winrate":
      return !established ? null : v < 0.25 ? "bad" : v < 0.45 ? "watch" : "ok";
    case "eval_first":
      return v < 0.9 ? "bad" : "ok";
    case "eval_past":
      return v < 0.2 ? "bad" : v < 0.35 ? "watch" : "ok";
    case "eval_bot":
      // le juge absolu : 0 prolongé = à surveiller ; tout décollage = sain
      if (v >= 0.5) return "ok";
      if (v > 0.02 || slope(values, 8) > 0.02) return "ok";
      return values.length >= 8 ? "watch" : null;
    case "hit_rate":
      return !established ? null : v < 5 ? "bad" : v < 20 ? "watch" : "ok";
    case "sprint_hits":
      return !established ? null : v < 0.1 ? "bad" : v < 0.25 ? "watch" : "ok";
    case "combo_hits":
      return !established ? null : v < 0.1 ? "bad" : v < 0.25 ? "watch" : "ok";
    case "combo_tap_frac":
    case "combo_s_tap_frac":
    case "combo_z_tap_frac":
      return !established ? null : v <= 0 ? "bad" : v < 0.02 ? "watch" : "ok";
    case "rechain_hit_frac":
    case "counter_lane_rechain_hit_frac":
    case "spar_rechain_hit_frac":
    case "rehit_rechain_hit_frac":
    case "pressure_rechain_hit_frac":
    case "fresh_rechain_hit_frac":
    case "fresh_chase_rechain_hit_frac":
    case "fresh_spar_rechain_hit_frac":
    case "fresh_rehit_rechain_hit_frac":
    case "fresh_pressure_rechain_hit_frac":
      return !established ? null : v <= 0 ? "bad" : v < 0.18 ? "watch" : "ok";
    case "rechain_taken_frac":
    case "counter_lane_rechain_taken_frac":
    case "spar_rechain_taken_frac":
    case "rehit_rechain_taken_frac":
    case "pressure_rechain_taken_frac":
    case "fresh_rechain_taken_frac":
    case "fresh_chase_rechain_taken_frac":
    case "fresh_spar_rechain_taken_frac":
    case "fresh_rehit_rechain_taken_frac":
    case "fresh_pressure_rechain_taken_frac":
      return !established ? null : v > 0.45 ? "bad" : v > 0.25 ? "watch" : "ok";
    case "counter_break_hit_frac":
    case "counter_lane_break_hit_frac":
    case "spar_counter_break_hit_frac":
    case "rehit_counter_break_hit_frac":
    case "pressure_counter_break_hit_frac":
    case "fresh_counter_break_hit_frac":
    case "fresh_chase_counter_break_hit_frac":
    case "fresh_spar_counter_break_hit_frac":
    case "fresh_rehit_counter_break_hit_frac":
    case "fresh_pressure_counter_break_hit_frac":
      return !established ? null : v <= 0 ? "bad" : v < 0.12 ? "watch" : "ok";
    case "counter_break_taken_frac":
    case "counter_lane_break_taken_frac":
    case "spar_counter_break_taken_frac":
    case "rehit_counter_break_taken_frac":
    case "pressure_counter_break_taken_frac":
    case "fresh_counter_break_taken_frac":
    case "fresh_chase_counter_break_taken_frac":
    case "fresh_spar_counter_break_taken_frac":
    case "fresh_rehit_counter_break_taken_frac":
    case "fresh_pressure_counter_break_taken_frac":
      return !established ? null : v > 0.55 ? "bad" : v > 0.35 ? "watch" : "ok";
    case "under_combo_counter_hit_rate":
    case "counter_lane_hit_rate":
      return !established ? null : v <= 0 ? "bad" : v < 2 ? "watch" : "ok";
    case "fresh_under_combo_counter_hit_frac":
    case "fresh_chase_under_combo_counter_hit_frac":
    case "fresh_spar_under_combo_counter_hit_frac":
    case "fresh_rehit_under_combo_counter_hit_frac":
    case "fresh_pressure_under_combo_counter_hit_frac":
    case "under_combo_counter_hit_frac":
      return !established ? null : v <= 0 ? "bad" : v < 0.05 ? "watch" : "ok";
    case "combo12_hits":
      return !established ? null : v <= 0 ? "watch" : "ok";
    case "fresh_combo12_state":
      return !established ? null : v < 0.05 ? "watch" : "ok";
    case "fresh_chase_combo12_state":
      return !established ? null : v <= 0 ? "watch" : "ok";
    case "fresh_spar_combo12_state":
      return !established ? null : v <= 0 ? "watch" : "ok";
    case "fresh_rehit_combo12_state":
    case "fresh_pressure_combo12_state":
      return !established ? null : v <= 0 ? "watch" : "ok";
    case "combo_max":
    case "counter_lane_combo_max":
      return !established ? null : v < 6 ? "bad" : v < 12 ? "watch" : "ok";
    case "fresh_chase_combo_max":
      return !established ? null : v < 4 ? "bad" : v < 8 ? "watch" : "ok";
    case "fresh_spar_combo_max":
      return !established ? null : v < 6 ? "bad" : v < 12 ? "watch" : "ok";
    case "fresh_rehit_combo_max":
    case "fresh_pressure_combo_max":
      return !established ? null : v < 6 ? "bad" : v < 12 ? "watch" : "ok";
    case "fresh_chase_hit_rate":
      return !established ? null : v < 10 ? "bad" : v < 30 ? "watch" : "ok";
    case "fresh_spar_hit_rate":
      return !established ? null : v < 10 ? "bad" : v < 30 ? "watch" : "ok";
    case "fresh_chase_trade_hit_frac":
      return !established ? null : v > 0.35 ? "bad" : v > 0.18 ? "watch" : "ok";
    case "fresh_spar_trade_hit_frac":
      return !established ? null : v > 0.30 ? "bad" : v > 0.15 ? "watch" : "ok";
    case "fresh_sky_frac":
      return !established ? null : v > 0.05 ? "bad" : v > 0.02 ? "watch" : "ok";
    case "engage_rate":
      return !established ? null : v < 0.1 ? "bad" : v < 0.15 ? "watch" : "ok";
    case "elo": {
      if (!established) return null;
      const s = slope(values, 40);
      return s < -60 ? "watch" : s > 20 ? "ok" : null;
    }
    case "sps": {
      const med = median(recent);
      return med > 0 && v < 0.5 * med ? "watch" : null;
    }
    default:
      return null;
  }
}
