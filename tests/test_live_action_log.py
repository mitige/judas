from pathlib import Path

from tools.live_action_log import (
    DEFAULT_REQUIRE_MODEL,
    LIVE_GUARDED_MODEL_MARKERS,
    analyze_lines,
    latest_session_lines,
    main,
    verdict_text,
)


ROOT = Path(__file__).resolve().parents[1]


def _line(tick: int, forward: int = 1, strafe: int = 0, jump: bool = False,
          pitch: float = 1.0, pitch_err: float = 1.0,
          attack: bool = False,
          sprint: bool = True,
          model: str = "models/combo_god_leaderboard10_combo12-safe_latest.pts",
          dist: float = 2.6,
          own_hurt: int = 0,
          opp_hurt: int = 0,
          own_hits: int = 0,
          opp_hits: int = 0) -> str:
    return (
        f"tick={tick} model={model} forward={forward} strafe={strafe} "
        f"jump={str(jump).lower()} sprint={str(sprint).lower()} "
        f"attack={str(attack).lower()} "
        f"dyaw=0.0 dpitch=1.0 ownPitch={pitch:.3f} yawErr=0.0 "
        f"pitchErr={pitch_err:.3f} dist={dist:.3f} "
        f"ownHurt={own_hurt} oppHurt={opp_hurt} "
        f"ownHits={own_hits} oppHits={opp_hits}"
    )


def test_live_action_log_reports_pass_for_body_aim_with_boxing_strafe():
    lines = ["event=start model=models/combo_god_leaderboard10_combo12-safe_latest.pts history=8"]
    lines += [_line(i, strafe=1) for i in range(24)]

    report = analyze_lines(lines)
    text = verdict_text(
        report,
        min_strafe_frac=0.05,
        require_model="combo_god_leaderboard10_combo12-safe_latest",
    )

    assert text.startswith("PASS ")
    assert "escape=0/0" in text
    assert "strafe=24" in text
    assert "min_strafe_frac=0.050" in text
    assert "opener_strafe=20/20" in text
    assert "opener_strafe_frac=1.000" in text
    assert "opener_strafe_hold=20/20" in text
    assert "opener_strafe_hold_frac=1.000" in text
    assert "opener_pressure=20/20" in text
    assert "opener_pressure_frac=1.000" in text
    assert "strafe_flips=0" in text
    assert "strafe_hold_avg=24.00" in text
    assert "attack_cps=0.00" in text
    assert "max_attack_cps=10.00" in text
    assert "max_tap_back_frac=0.000" in text
    assert "hit_wtap=0/0" in text
    assert "min_hit_wtap_frac=0.750" in text


def test_live_action_log_default_model_requires_leaderboard_model():
    leaderboard = analyze_lines(
        ["event=start model=models/combo_god_leaderboard10_combo12-safe_latest.pts"]
        + [_line(i, strafe=1) for i in range(24)]
    )
    attn = analyze_lines(
        ["event=start model=models/combo_god_attn96_combo12-best.pts"]
        + [_line(i, strafe=1, model="models/combo_god_attn96_combo12-best.pts")
           for i in range(24)]
    )
    countertap = analyze_lines(
        ["event=start model=models/combo_god_countertap96_combo12-safe_latest.pts"]
        + [_line(i, strafe=1, model="models/combo_god_countertap96_combo12-safe_latest.pts")
           for i in range(24)]
    )
    legacy = analyze_lines(
        ["event=start model=models/combo_god_directpad_lock_combo12-safe_latest.pts"]
        + [_line(i, strafe=1, model="models/combo_god_directpad_lock_combo12-safe_latest.pts")
           for i in range(24)]
    )
    wrong = analyze_lines(
        ["event=start model=models/other.pts"]
        + [_line(i, model="models/other.pts") for i in range(24)]
    )

    assert "combo_god_attn96_combo12" in LIVE_GUARDED_MODEL_MARKERS
    assert verdict_text(leaderboard, require_model=DEFAULT_REQUIRE_MODEL).startswith("PASS ")
    assert verdict_text(attn, require_model=DEFAULT_REQUIRE_MODEL).startswith("PASS ")
    assert verdict_text(countertap, require_model=DEFAULT_REQUIRE_MODEL).startswith("FAIL ")
    assert verdict_text(legacy, require_model=DEFAULT_REQUIRE_MODEL).startswith("FAIL ")
    assert verdict_text(wrong, require_model=DEFAULT_REQUIRE_MODEL).startswith("FAIL ")


def test_live_action_log_rejects_escape_or_sky():
    escape = analyze_lines([_line(i, jump=i == 5) for i in range(24)])
    assert verdict_text(escape).startswith("FAIL ")
    assert "escape=0/1" in verdict_text(escape)

    no_strafe = analyze_lines([_line(i) for i in range(24)])
    text = verdict_text(no_strafe, min_strafe_frac=0.05)
    assert text.startswith("FAIL ")
    assert "strafe=0" in text

    sky = analyze_lines([_line(i, pitch=70.0 if i > 12 else 1.0) for i in range(24)])
    assert verdict_text(sky).startswith("FAIL ")
    assert "sky_frac=" in verdict_text(sky)


def test_live_action_log_rejects_any_tap_back_for_no_back_contract():
    lines = [_line(i, forward=-1 if i in (5, 6) else 1,
                   sprint=False if i in (5, 6) else True)
             for i in range(24)]
    report = analyze_lines(lines)
    text = verdict_text(report)

    assert text.startswith("FAIL ")
    assert "escape=0/0" in text
    assert "tap_back=2" in text

    sprint_back = analyze_lines([_line(i, forward=-1 if i == 5 else 1) for i in range(24)])
    assert verdict_text(sprint_back).startswith("FAIL ")
    assert "escape=1/0" in verdict_text(sprint_back)


def test_live_action_log_rejects_twitchy_strafe_flips():
    lines = [
        _line(i, strafe=1 if i % 2 == 0 else -1)
        for i in range(24)
    ]
    report = analyze_lines(lines)
    text = verdict_text(report, min_strafe_frac=0.05, max_strafe_flip_frac=0.20)

    assert text.startswith("FAIL ")
    assert report.strafe_flips == 23
    assert "strafe_flips=23" in text
    assert "max_strafe_flip_frac=0.200" in text


def test_live_action_log_rejects_one_tick_strafe_bursts_through_neutral():
    pattern = [1, 0, -1, 0]
    lines = [_line(i, strafe=pattern[i % len(pattern)]) for i in range(24)]
    report = analyze_lines(lines)
    text = verdict_text(
        report,
        min_strafe_frac=0.05,
        max_strafe_flip_frac=0.10,
        min_strafe_hold_avg=3.0,
    )

    assert text.startswith("FAIL ")
    assert report.strafe_flips == 0
    assert report.strafe_runs == 12
    assert "strafe_hold_avg=1.00" in text
    assert "min_strafe_hold_avg=3.00" in text


def test_live_action_log_rejects_straight_opener_even_with_late_strafe():
    lines = [_line(i, strafe=0) for i in range(20)]
    lines += [_line(i + 20, strafe=1) for i in range(24)]
    report = analyze_lines(lines, opener_ticks=20)
    text = verdict_text(
        report,
        min_strafe_frac=0.50,
        min_opener_strafe_frac=0.75,
    )

    assert report.strafe == 24
    assert text.startswith("FAIL ")
    assert "strafe_frac=0.545" in text
    assert "opener_strafe=0/20" in text
    assert "opener_strafe_frac=0.000" in text
    assert "min_opener_strafe_frac=0.750" in text


def test_live_action_log_rejects_passive_lateral_opener():
    lines = [_line(i, forward=0, strafe=1) for i in range(20)]
    lines += [_line(i + 20, forward=1, strafe=1) for i in range(24)]
    report = analyze_lines(lines, opener_ticks=20)
    text = verdict_text(
        report,
        min_strafe_frac=0.50,
        min_opener_strafe_frac=0.75,
        min_opener_pressure_frac=0.60,
    )

    assert report.opener_strafe == 20
    assert report.opener_pressure == 0
    assert text.startswith("FAIL ")
    assert "opener_strafe=20/20" in text
    assert "opener_strafe_frac=1.000" in text
    assert "opener_pressure=0/20" in text
    assert "opener_pressure_frac=0.000" in text
    assert "min_opener_pressure_frac=0.600" in text


def test_live_action_log_rejects_opener_side_switch_jitter():
    lines = [_line(i, strafe=1) for i in range(10)]
    lines += [_line(i + 10, strafe=-1) for i in range(24)]
    report = analyze_lines(lines, opener_ticks=20)
    text = verdict_text(
        report,
        min_strafe_frac=0.50,
        min_opener_strafe_frac=0.75,
        min_opener_strafe_hold_frac=0.70,
        min_opener_pressure_frac=0.60,
    )

    assert report.opener_strafe == 20
    assert report.opener_strafe_hold == 10
    assert report.opener_pressure == 20
    assert text.startswith("FAIL ")
    assert "opener_strafe=20/20" in text
    assert "opener_strafe_frac=1.000" in text
    assert "opener_strafe_hold=10/20" in text
    assert "opener_strafe_hold_frac=0.500" in text
    assert "min_opener_strafe_hold_frac=0.700" in text


def test_live_action_log_rejects_20_cps_attack_spam():
    lines = [_line(i, strafe=1, attack=True) for i in range(24)]
    report = analyze_lines(lines)
    text = verdict_text(report, max_attack_cps=10.0)

    assert text.startswith("FAIL ")
    assert "attack=24" in text
    assert "attack_cps=20.00" in text
    assert "max_attack_cps=10.00" in text


def test_live_action_log_reports_combo_streaks_from_hit_counters():
    lines = ["event=start model=models/combo_god_leaderboard10_combo12-safe_latest.pts"]
    lines += [_line(i, strafe=1) for i in range(20)]
    lines += [
        _line(i + 20, forward=0, sprint=False, strafe=1, own_hits=i)
        for i in range(1, 14)
    ]
    lines += [_line(i + 33, strafe=1, own_hits=13, opp_hits=i) for i in range(1, 5)]
    report = analyze_lines(lines)
    text = verdict_text(
        report,
        min_samples=37,
        min_strafe_frac=0.05,
        min_max_own_combo=12,
        max_max_opp_combo=4,
    )

    assert report.max_own_combo == 13
    assert report.max_opp_combo == 4
    assert text.startswith("PASS ")
    assert "max_own_combo=13" in text
    assert "min_max_own_combo=12" in text
    assert "max_opp_combo=4" in text
    assert "hit_wtap=13/13" in text
    assert "hit_wtap_frac=1.000" in text


def test_live_action_log_accepts_next_tick_wtap_after_hit_counter():
    lines = ["event=start model=models/combo_god_leaderboard10_combo12-safe_latest.pts"]
    lines += [_line(i, strafe=1) for i in range(20)]
    lines += [_line(20, forward=1, sprint=True, strafe=1, own_hits=1)]
    lines += [_line(21, forward=0, sprint=False, strafe=1, own_hits=1)]
    lines += [_line(i, strafe=1, own_hits=1) for i in range(22, 24)]
    report = analyze_lines(lines)
    text = verdict_text(report)

    assert report.own_hit_events == 1
    assert report.own_hit_wtap == 1
    assert text.startswith("PASS ")
    assert "hit_wtap=1/1" in text
    assert "hit_wtap_frac=1.000" in text


def test_live_action_log_rejects_hits_without_wtap_release():
    lines = ["event=start model=models/combo_god_leaderboard10_combo12-safe_latest.pts"]
    lines += [_line(i, strafe=1) for i in range(20)]
    lines += [_line(20, forward=1, sprint=True, strafe=1, own_hits=1)]
    lines += [_line(i, strafe=1, own_hits=1) for i in range(21, 24)]
    report = analyze_lines(lines)
    text = verdict_text(report)

    assert report.own_hit_events == 1
    assert report.own_hit_wtap == 0
    assert text.startswith("FAIL ")
    assert "hit_wtap=0/1" in text
    assert "hit_wtap_frac=0.000" in text
    assert "min_hit_wtap_frac=0.750" in text


def test_live_action_log_can_reject_under_combo_without_counter_clicks():
    lines = [
        _line(i, strafe=1, attack=False, own_hurt=12, opp_hurt=0, dist=2.8)
        for i in range(24)
    ]
    report = analyze_lines(lines)
    text = verdict_text(
        report,
        min_strafe_frac=0.05,
        min_under_combo_attack_frac=0.50,
    )

    assert text.startswith("FAIL ")
    assert report.under_combo_samples == 24
    assert report.under_combo_attack == 0
    assert "under_combo_attack=0/24" in text
    assert "min_under_combo_attack_frac=0.500" in text


def test_live_action_log_can_reject_under_combo_attacks_without_counter_hits():
    lines = [
        _line(i, strafe=1, attack=True, own_hurt=12, opp_hurt=0, dist=2.8)
        for i in range(24)
    ]
    report = analyze_lines(lines)
    text = verdict_text(
        report,
        min_strafe_frac=0.05,
        min_under_combo_attack_frac=0.50,
        min_under_combo_counter_hit_frac=0.05,
    )

    assert text.startswith("FAIL ")
    assert report.under_combo_samples == 24
    assert report.under_combo_attack == 24
    assert report.under_combo_counter_hits == 0
    assert "under_combo_attack=24/24" in text
    assert "under_combo_counter_hit=0/24" in text
    assert "min_under_combo_counter_hit_frac=0.050" in text


def test_live_action_log_accepts_under_combo_counter_hits():
    lines = ["event=start model=models/combo_god_leaderboard10_combo12-safe_latest.pts"]
    lines += [_line(i, strafe=1, attack=False, own_hurt=12, opp_hurt=0) for i in range(20)]
    lines += [
        _line(i + 20, forward=0, sprint=False, strafe=1, attack=True,
              own_hurt=12, opp_hurt=0, own_hits=i + 1)
        for i in range(4)
    ]
    report = analyze_lines(lines)
    text = verdict_text(
        report,
        min_strafe_frac=0.05,
        min_under_combo_attack_frac=0.10,
        min_under_combo_counter_hit_frac=0.10,
    )

    assert text.startswith("PASS ")
    assert report.under_combo_samples == 24
    assert report.under_combo_counter_hits == 4
    assert "under_combo_counter_hit=4/24" in text
    assert "under_combo_counter_hit_frac=0.167" in text


def test_live_action_log_uses_latest_session():
    lines = ["event=start model=old"] + [_line(i, jump=True, model="old") for i in range(24)]
    lines += ["event=start model=models/combo_god_leaderboard10_combo12-safe_latest.pts"]
    lines += [_line(i, strafe=1) for i in range(24)]

    report = analyze_lines(latest_session_lines(lines))

    assert report.jump == 0
    assert "safe_latest" in report.model


def test_live_action_log_cli_strict(tmp_path):
    log = tmp_path / "live.log"
    log.write_text("\n".join(["event=start model=models/combo_god_leaderboard10_combo12-safe_latest.pts"] + [_line(i, strafe=1) for i in range(24)]))

    assert main(["--strict", str(log)]) == 0


def test_check_live_actions_script_defaults_to_runs_log():
    ps1 = (ROOT / "scripts/check_live_actions.ps1").read_text(encoding="utf-8")
    bat = (ROOT / "scripts/check_live_actions.bat").read_text(encoding="utf-8")

    assert "judas-live-actions.log" in ps1
    assert "tools\\live_action_log.py" in ps1
    assert "MaxAttackCps" in ps1
    assert "MinStrafeFrac" in ps1
    assert "OpenerTicks" in ps1
    assert "MinOpenerStrafeFrac" in ps1
    assert "MinOpenerStrafeHoldFrac" in ps1
    assert "MinOpenerPressureFrac" in ps1
    assert "MaxStrafeFlipFrac" in ps1
    assert "MinStrafeHoldAvg" in ps1
    assert "MinHitWtapFrac" in ps1
    assert "MinUnderComboCounterHitFrac" in ps1
    assert "RequireModel" in ps1
    assert "combo_god_attn96_combo12" in ps1
    assert "combo_god_bodyaim96_combo12" in ps1
    assert "$MinStrafeFrac = 0.50" in ps1
    assert "--max-attack-cps" in ps1
    assert "--min-strafe-frac" in ps1
    assert "--opener-ticks" in ps1
    assert "--min-opener-strafe-frac" in ps1
    assert "--min-opener-strafe-hold-frac" in ps1
    assert "--min-opener-pressure-frac" in ps1
    assert "--max-strafe-flip-frac" in ps1
    assert "--min-strafe-hold-avg" in ps1
    assert "--min-hit-wtap-frac" in ps1
    assert "--min-under-combo-counter-hit-frac" in ps1
    assert "--require-model" in ps1
    assert "max-tap-back-frac" not in ps1
    assert "$Reset" in ps1
    assert "ExecutionPolicy Bypass" in bat
    assert "check_live_actions.ps1" in bat
