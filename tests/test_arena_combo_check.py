from pathlib import Path

from tools.arena_combo_check import (
    ArenaComboReport,
    MatchEvent,
    events_text,
    main,
    report_passes,
    verdict_text,
)


ROOT = Path(__file__).resolve().parents[1]


def _report(
    draws=0,
    max_combo=(12, 1),
    matches=8,
    wins=None,
    hits_per_min=(24.0, 2.0),
    close_frac=0.25,
    sky_frac=(0.0, 0.0),
    back_frac=(0.0, 0.0),
    tap_back_frac=(0.0, 0.0),
    strafe_frac=(0.60, 0.60),
    opener_strafe_frac=(0.90, 0.90),
    opener_strafe_hold_frac=(0.90, 0.90),
    opener_pressure_frac=(0.80, 0.80),
    jump_frac=(0.0, 0.0),
):
    wins = wins if wins is not None else (matches - draws, 0)
    events = tuple(
        MatchEvent(winner=0, combo=(3, 0), max_combo=max_combo,
                   wins=(min(i + 1, wins[0]), wins[1]), draws=draws)
        for i in range(matches)
    )
    return ArenaComboReport(
        matches=matches,
        ticks=matches * 600,
        wins=wins,
        draws=draws,
        max_combo=max_combo,
        hits=(96, 8),
        swings=(240, 40),
        hits_per_min=hits_per_min,
        swings_per_min=(60.0, 10.0),
        avg_dist=2.6,
        close_frac=close_frac,
        avg_abs_pitch=(8.0, 10.0),
        sky_frac=sky_frac,
        back_frac=back_frac,
        tap_back_frac=tap_back_frac,
        strafe_frac=strafe_frac,
        opener_strafe_frac=opener_strafe_frac,
        opener_strafe_hold_frac=opener_strafe_hold_frac,
        opener_pressure_frac=opener_pressure_frac,
        jump_frac=jump_frac,
        events=events,
        sample=True,
        target_hits=50,
        spawn_gap=8.0,
        arena_size=40.0,
        cps=10.0,
        rot_speed=190.0,
    )


def test_arena_combo_report_passes_when_model_a_actually_boxes():
    report = _report(max_combo=(30, 1), wins=(8, 0))

    assert report_passes(report, required_matches=8, min_combo=12, max_draws=0)
    text = verdict_text(report, required_matches=8, min_combo=12, max_draws=0)
    assert "PASS" in text
    assert "draws=0" in text
    assert "model_combo=30/1" in text
    assert "threshold=12/0" in text
    assert "hits_min=24.0/2.0" in text
    assert "close=0.25" in text
    assert "cps=10.0" in text
    assert "rot=190.0" in text
    assert "escape=0.00/0.60/0.00" in text
    assert "opener_strafe=0.90" in text
    assert "opener_strafe_hold=0.90" in text


def test_arena_combo_report_fails_old_draw_or_short_combo_regression():
    assert not report_passes(_report(draws=8, max_combo=(4, 0), wins=(0, 0)))
    assert not report_passes(_report(draws=0, max_combo=(4, 0), wins=(8, 0)))
    assert "FAIL" in verdict_text(_report(draws=8, max_combo=(4, 0), wins=(0, 0)))


def test_arena_combo_report_fails_when_only_opponent_combines():
    report = _report(max_combo=(1, 30), wins=(0, 8), hits_per_min=(0.0, 24.0))

    assert report.best_combo == 30
    assert not report_passes(report, required_matches=8, min_combo=12, max_draws=0)
    text = verdict_text(report, required_matches=8, min_combo=12, max_draws=0)
    assert "FAIL" in text
    assert "model_combo=1/30" in text
    assert "wins=0/8" in text


def test_arena_combo_report_can_require_role_b_to_box():
    passive_b = _report(max_combo=(30, 1), wins=(8, 0),
                        hits_per_min=(24.0, 2.0))
    active_b = _report(max_combo=(0, 30), wins=(0, 8),
                       hits_per_min=(0.0, 24.0))

    assert not report_passes(
        passive_b, required_matches=8, min_combo=0, min_wins_a=0,
        min_hits_a=0.0, min_combo_b=12, min_wins_b=1, min_hits_b=18.0)
    assert report_passes(
        active_b, required_matches=8, min_combo=0, min_wins_a=0,
        min_hits_a=0.0, min_combo_b=12, min_wins_b=1, min_hits_b=18.0)
    text = verdict_text(
        active_b, required_matches=8, min_combo=0, min_wins_a=0,
        min_hits_a=0.0, min_combo_b=12, min_wins_b=1, min_hits_b=18.0)
    assert "model_combo=0/30" in text
    assert "threshold=0/12" in text
    assert "min_wins=0/1" in text


def test_arena_combo_report_fails_on_far_or_sky_aiming_model_a():
    assert not report_passes(_report(max_combo=(20, 1), close_frac=0.0))
    assert not report_passes(_report(max_combo=(20, 1), sky_frac=(0.90, 0.05)))


def test_arena_combo_report_fails_on_escape_actions_model_a():
    assert not report_passes(_report(max_combo=(20, 1), back_frac=(0.05, 0.0)))
    assert not report_passes(_report(max_combo=(20, 1), tap_back_frac=(0.01, 0.0)))
    assert report_passes(_report(max_combo=(20, 1), strafe_frac=(0.60, 0.60)))
    assert not report_passes(_report(max_combo=(20, 1), strafe_frac=(0.20, 0.60)))
    assert not report_passes(_report(
        max_combo=(20, 1),
        strafe_frac=(0.60, 0.60),
        opener_strafe_frac=(0.20, 0.90),
    ))
    assert not report_passes(_report(
        max_combo=(20, 1),
        strafe_frac=(0.60, 0.60),
        opener_strafe_frac=(0.90, 0.90),
        opener_strafe_hold_frac=(0.45, 0.90),
        opener_pressure_frac=(0.90, 0.90),
    ))
    assert not report_passes(_report(
        max_combo=(20, 1),
        strafe_frac=(0.60, 0.60),
        opener_strafe_frac=(0.90, 0.90),
        opener_pressure_frac=(0.20, 0.90),
    ))
    assert not report_passes(_report(max_combo=(20, 1), jump_frac=(0.05, 0.0)))
    text = verdict_text(_report(max_combo=(20, 1), jump_frac=(0.05, 0.0)))
    assert "escape=0.00/0.60/0.05" in text
    assert "limit=0.01/0.50-1.00/0.01" in text
    assert "min_opener_strafe=0.75" in text
    text = verdict_text(_report(max_combo=(20, 1), opener_strafe_hold_frac=(0.45, 0.90)))
    assert "opener_strafe_hold=0.45" in text
    assert "min_opener_strafe_hold=0.70" in text
    text = verdict_text(_report(max_combo=(20, 1), opener_pressure_frac=(0.20, 0.90)))
    assert "opener_pressure=0.20" in text
    assert "min_opener_pressure=0.60" in text


def test_arena_combo_report_fails_on_sky_or_escape_actions_model_b():
    assert not report_passes(_report(max_combo=(20, 1), sky_frac=(0.0, 0.05)))
    assert not report_passes(_report(max_combo=(20, 1), back_frac=(0.0, 0.05)))
    assert not report_passes(_report(max_combo=(20, 1), strafe_frac=(0.60, 0.0)))
    assert not report_passes(_report(
        max_combo=(20, 1),
        strafe_frac=(0.60, 0.60),
        opener_strafe_frac=(0.90, 0.20),
    ))
    assert not report_passes(_report(
        max_combo=(20, 1),
        strafe_frac=(0.60, 0.60),
        opener_strafe_frac=(0.90, 0.90),
        opener_strafe_hold_frac=(0.90, 0.45),
        opener_pressure_frac=(0.90, 0.90),
    ))
    assert not report_passes(_report(
        max_combo=(20, 1),
        strafe_frac=(0.60, 0.60),
        opener_strafe_frac=(0.90, 0.90),
        opener_pressure_frac=(0.90, 0.20),
    ))
    assert not report_passes(_report(max_combo=(20, 1), jump_frac=(0.0, 0.05)))
    text = verdict_text(_report(max_combo=(20, 1), strafe_frac=(0.60, 0.0)))
    assert "escape_b=0.00/0.00/0.00" in text
    assert "limit_b=0.01/0.50-1.00/0.01" in text
    assert "min_opener_strafe_b=0.75" in text
    text = verdict_text(_report(max_combo=(20, 1), opener_strafe_hold_frac=(0.90, 0.45)))
    assert "opener_strafe_hold_b=0.45" in text
    assert "min_opener_strafe_hold_b=0.70" in text
    text = verdict_text(_report(max_combo=(20, 1), opener_pressure_frac=(0.90, 0.20)))
    assert "opener_pressure_b=0.20" in text
    assert "min_opener_pressure_b=0.60" in text


def test_arena_combo_events_text_is_compact():
    text = events_text(_report(max_combo=(12, 1), matches=2).events)

    assert "#1:w=0" in text
    assert "best=12/1" in text


def test_arena_combo_cli_reports_missing_model(capsys):
    assert main(["--model-a", "missing-a.pt", "--model-b", "__chase_bot__"]) == 2
    assert "MISSING model=missing-a.pt" in capsys.readouterr().out




def test_arena_combo_cli_accepts_combo_pad_sentinel(capsys):
    assert main(["--model-a", "missing-a.pt", "--model-b", "__combo_pad__"]) == 2
    assert "MISSING model=missing-a.pt" in capsys.readouterr().out


def test_arena_combo_cli_accepts_combo_spar_sentinel(capsys):
    assert main(["--model-a", "missing-a.pt", "--model-b", "__combo_spar__"]) == 2
    assert "MISSING model=missing-a.pt" in capsys.readouterr().out


def test_arena_combo_check_scripts_wire_defaults():
    ps1 = (ROOT / "scripts/check_arena_combo.ps1").read_text()
    bat = (ROOT / "scripts/check_arena_combo.bat").read_text()

    assert "models/combo_god_leaderboard10_combo12-safe_latest.pts" in ps1
    assert "models/combo_god_countertap96_combo12-safe_latest.pts" in ps1
    assert "models\\combo_god_leaderboard10_combo12-safe_latest.pts" in ps1
    assert "models\\combo_god_countertap96_combo12-safe_latest.pts" in ps1
    assert "models\\combo_god_directpad_lock_combo12-safe_latest.pts" in ps1
    assert "__combo_spar__" in ps1
    assert "---ROLE A PROOF---" in ps1
    assert "---ROLE B PROOF---" in ps1
    assert "---MIRROR PROOF---" in ps1
    assert "NoMirrorProof" in ps1
    assert "MirrorMaxDraws" in ps1
    assert "$MinCloseFrac = 0.04" in ps1
    assert "$MinStrafeFracA = 0.50" in ps1
    assert "$MinStrafeFracB = 0.50" in ps1
    assert "$MinOpenerStrafeFracA = 0.75" in ps1
    assert "$MinOpenerStrafeFracB = 0.75" in ps1
    assert "$MinOpenerStrafeHoldFracA = 0.70" in ps1
    assert "$MinOpenerStrafeHoldFracB = 0.70" in ps1
    assert "$MinOpenerPressureFracA = 0.60" in ps1
    assert "$MinOpenerPressureFracB = 0.60" in ps1
    assert "runs/combo_god_aggro/latest.pt" not in ps1
    assert "runs/combo_god_consistent/latest.pt" not in ps1
    assert "--min-combo" in ps1
    assert "--min-combo-b" in ps1
    assert "--max-draws" in ps1
    assert "--min-hits-a" in ps1
    assert "--min-hits-b" in ps1
    assert "--min-wins-b" in ps1
    assert "--max-sky-frac-a" in ps1
    assert "--max-sky-frac-b" in ps1
    assert "--max-back-frac-a" in ps1
    assert "--max-back-frac-b" in ps1
    assert "--max-tap-back-frac-a" in ps1
    assert "--max-tap-back-frac-b" in ps1
    assert "--min-strafe-frac-a" in ps1
    assert "--min-strafe-frac-b" in ps1
    assert "--min-opener-strafe-frac-a" in ps1
    assert "--min-opener-strafe-frac-b" in ps1
    assert "--min-opener-strafe-hold-frac-a" in ps1
    assert "--min-opener-strafe-hold-frac-b" in ps1
    assert "--max-strafe-frac-a" in ps1
    assert "--max-strafe-frac-b" in ps1
    assert "--max-jump-frac-a" in ps1
    assert "--max-jump-frac-b" in ps1
    assert "--max-steps" in ps1
    assert "arena_combo_check.py" in ps1
    assert '"--model-b", $ModelA' in ps1
    assert '"--min-combo", 1' in ps1
    assert '"--min-combo-b", 1' in ps1
    assert '"--min-hits-b", $MinHitsA' in ps1
    assert "ExecutionPolicy Bypass" in bat
    assert "check_arena_combo.ps1" in bat


def test_guide_documents_arena_combo_runtime_check():
    guide = (ROOT / "docs/GUIDE.md").read_text(encoding="utf-8")

    assert "Verifier le dieu du combo en arene" in guide
    assert "scripts\\check_arena_combo.bat -Events" in guide
    assert "combo_god_leaderboard10_combo12-safe_latest.pts" in guide
    assert "combo_god_countertap96_combo12-safe_latest.pts" in guide
    assert "MIRROR PROOF" in guide
    assert "combo_god_consistent/latest.pt" not in guide
    assert "PASS" in guide
    assert "FAIL" in guide
    assert "MISSING" in guide
