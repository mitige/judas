from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable

import websockets

DEFAULT_MODEL = "models/combo_god_leaderboard10_combo12-safe_latest.pts"


@dataclass(frozen=True)
class LiveWsResult:
    case: str
    actions: int
    back: int
    strafe: int
    jump: int
    final_yaw: float
    final_pitch: float
    first: dict
    neutral: int = 0
    sprint: int = 0
    attack: int = 0
    max_back: int = 0
    min_strafe: int = 0
    min_neutral: int = 0
    max_neutral: int | None = None
    max_neutral_frac: float | None = None
    min_sprint: int = 0
    max_sprint: int | None = None
    min_attack: int = 0
    max_attack: int | None = None
    max_strafe: int | None = None
    strafe_sign_flips: int = 0
    max_strafe_sign_flips: int | None = None

    @property
    def ok(self) -> bool:
        sprint_ok = self.max_sprint is None or self.sprint <= self.max_sprint
        attack_ok = self.max_attack is None or self.attack <= self.max_attack
        strafe_ok = self.max_strafe is None or self.strafe <= self.max_strafe
        neutral_ok = self.max_neutral is None or self.neutral <= self.max_neutral
        neutral_frac_ok = (
            self.max_neutral_frac is None
            or self.neutral / max(self.actions, 1) <= self.max_neutral_frac
        )
        strafe_flips_ok = (
            self.max_strafe_sign_flips is None
            or self.strafe_sign_flips <= self.max_strafe_sign_flips
        )
        return (
            self.back <= self.max_back
            and self.strafe >= self.min_strafe
            and self.jump == 0
            and abs(self.final_pitch) < 60.0
            and self.neutral >= self.min_neutral
            and neutral_ok
            and neutral_frac_ok
            and self.sprint >= self.min_sprint
            and self.attack >= self.min_attack
            and attack_ok
            and strafe_ok
            and sprint_ok
            and strafe_flips_ok
        )


@dataclass(frozen=True)
class CaseSpec:
    name: str
    frames: tuple[tuple[dict, dict], ...]
    ticks: int | None = None
    measure_from: int = 0
    max_back: int = 0
    min_strafe: int = 0
    min_neutral: int = 0
    max_neutral: int | None = None
    max_neutral_frac: float | None = None
    min_sprint: int = 0
    max_sprint: int | None = None
    min_attack: int = 0
    max_attack: int | None = None
    max_strafe: int | None = None
    max_strafe_sign_flips: int | None = None


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _player(
    x: float,
    z: float,
    yaw: float = 0.0,
    pitch: float = 0.0,
    *,
    hurt: int = 0,
    hits: int = 0,
) -> dict:
    return {
        "x": x, "y": 64.0, "z": z,
        "vx": 0.0, "vy": 0.0, "vz": 0.0,
        "yaw": yaw, "pitch": pitch,
        "onGround": True, "sprinting": False,
        "hurtTime": 0, "hurtResistantTime": hurt,
        "jumpTicks": 0, "hits": hits,
    }


def _same(me: dict, target: dict) -> tuple[tuple[dict, dict], ...]:
    return ((me, target),)


def _target_hurt_decay(
    z: float,
    start_hurt: int,
    *,
    frames: int = 16,
) -> tuple[tuple[dict, dict], ...]:
    return tuple(
        (_player(0.0, 0.0), _player(0.0, z, hurt=max(start_hurt - i, 0)))
        for i in range(frames)
    )


def _post_opener_frames(
    then_me: dict,
    then_target: dict,
    *,
    warmup_ticks: int = 48,
    measure_ticks: int = 16,
) -> tuple[tuple[dict, dict], ...]:
    warmup = [(_player(0.0, 0.0), _player(0.0, 7.0)) for _ in range(warmup_ticks)]
    measured = [(then_me, then_target) for _ in range(measure_ticks)]
    return tuple(warmup + measured)


def _cases() -> list[CaseSpec]:
    return [
        CaseSpec("front_far", _same(_player(0.0, 0.0), _player(0.0, 7.0))),
        CaseSpec("right_far", _same(_player(0.0, 0.0), _player(3.5, 6.0))),
        CaseSpec("left_far", _same(_player(0.0, 0.0), _player(-3.5, 6.0))),
        CaseSpec(
            "opener_front_strafe",
            _same(_player(0.0, 0.0), _player(0.0, 7.0)),
            min_strafe=12,
            min_sprint=16,
            max_neutral=0,
            max_strafe_sign_flips=0,
        ),
        CaseSpec(
            "opener_right_strafe",
            _same(_player(0.0, 0.0), _player(3.5, 6.0)),
            min_strafe=12,
            min_sprint=16,
            max_neutral=0,
            max_strafe_sign_flips=0,
        ),
        CaseSpec(
            "opener_left_strafe",
            _same(_player(0.0, 0.0), _player(-3.5, 6.0)),
            min_strafe=12,
            min_sprint=16,
            max_neutral=0,
            max_strafe_sign_flips=0,
        ),
        CaseSpec(
            "post_opener_strafe",
            _post_opener_frames(_player(0.0, 0.0), _player(0.0, 7.0)),
            ticks=64,
            measure_from=48,
            min_strafe=16,
            min_sprint=16,
            max_neutral=0,
            max_strafe_sign_flips=0,
        ),
        CaseSpec(
            "post_opener_reset_cap",
            _post_opener_frames(_player(0.0, 0.0), _player(0.0, 2.80, hurt=12)),
            ticks=64,
            measure_from=48,
            min_strafe=16,
            max_neutral=2,
            min_sprint=14,
            max_strafe_sign_flips=0,
        ),
        CaseSpec(
            "combo_too_close_s_tap",
            _target_hurt_decay(2.10, 16),
            max_back=0,
            min_strafe=8,
            min_sprint=1,
            min_attack=1,
        ),
        CaseSpec(
            "combo_wait_rehit",
            _target_hurt_decay(2.80, 16),
            max_back=0,
            min_strafe=8,
            min_sprint=1,
            min_attack=1,
        ),
        CaseSpec(
            "combo_press_rehit",
            _same(_player(0.0, 0.0), _player(0.0, 2.80, hurt=11)),
            max_back=0,
            min_strafe=1,
            min_sprint=16,
            min_attack=1,
        ),
        CaseSpec(
            "combo_landed_reset",
            (
                (_player(0.0, 0.0, hits=0), _player(0.0, 2.80, hurt=0)),
                (_player(0.0, 0.0, hits=1), _player(0.0, 2.80, hurt=16)),
                (_player(0.0, 0.0, hits=1), _player(0.0, 2.80, hurt=15)),
                (_player(0.0, 0.0, hits=1), _player(0.0, 2.80, hurt=14)),
                (_player(0.0, 0.0, hits=1), _player(0.0, 2.80, hurt=13)),
                (_player(0.0, 0.0, hits=1), _player(0.0, 2.80, hurt=12)),
                (_player(0.0, 0.0, hits=1), _player(0.0, 2.80, hurt=11)),
                (_player(0.0, 0.0, hits=1), _player(0.0, 2.80, hurt=10)),
            ),
            max_back=0,
            min_strafe=4,
            min_sprint=1,
            min_attack=1,
        ),
        CaseSpec(
            "under_combo_counter",
            _same(_player(0.0, 0.0, hurt=16), _player(0.0, 2.60)),
            min_strafe=1,
            min_attack=8,
        ),
        CaseSpec(
            "under_combo_far_reentry",
            _same(_player(0.0, 0.0, hurt=16), _player(0.0, 6.60, hits=4)),
            min_sprint=16,
            max_strafe=0,
            max_attack=0,
            max_back=0,
        ),
    ]


async def _run_case(uri: str, spec: CaseSpec, ticks: int, timeout: float) -> LiveWsResult:
    async with websockets.connect(uri, open_timeout=timeout, close_timeout=2) as ws:
        actions: list[dict] = []
        yaw = float(spec.frames[0][0].get("yaw", 0.0))
        pitch = float(spec.frames[0][0].get("pitch", 0.0))
        total_ticks = spec.ticks or ticks
        for tick in range(total_ticks):
            base_me, base_target = spec.frames[min(tick, len(spec.frames) - 1)]
            me = dict(base_me)
            target = dict(base_target)
            me["yaw"] = yaw
            me["pitch"] = pitch
            await ws.send(json.dumps({"t": "state", "tick": tick, "self": me, "target": target}))
            action = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
            actions.append(action)
            yaw += float(action["dyaw"])
            pitch = max(-90.0, min(90.0, pitch + float(action["dpitch"])))
    measured_actions = actions[spec.measure_from:]
    strafe_signs = [int(a["strafe"]) for a in measured_actions if int(a["strafe"]) != 0]
    strafe_sign_flips = sum(
        1 for prev, cur in zip(strafe_signs, strafe_signs[1:]) if prev != cur
    )
    return LiveWsResult(
        case=spec.name,
        actions=len(measured_actions),
        back=sum(1 for a in measured_actions if int(a["forward"]) < 0),
        strafe=sum(1 for a in measured_actions if int(a["strafe"]) != 0),
        jump=sum(1 for a in measured_actions if bool(a["jump"])),
        final_yaw=float(yaw),
        final_pitch=float(pitch),
        first=measured_actions[0] if measured_actions else {},
        neutral=sum(1 for a in measured_actions if int(a["forward"]) == 0),
        sprint=sum(1 for a in measured_actions if bool(a["sprint"])),
        attack=sum(1 for a in measured_actions if bool(a["attack"])),
        max_back=spec.max_back,
        min_strafe=spec.min_strafe,
        min_neutral=spec.min_neutral,
        max_neutral=spec.max_neutral,
        max_neutral_frac=spec.max_neutral_frac,
        min_sprint=spec.min_sprint,
        max_sprint=spec.max_sprint,
        min_attack=spec.min_attack,
        max_attack=spec.max_attack,
        max_strafe=spec.max_strafe,
        strafe_sign_flips=strafe_sign_flips,
        max_strafe_sign_flips=spec.max_strafe_sign_flips,
    )


async def run_live_ws_check(host: str, port: int, ticks: int, timeout: float) -> list[LiveWsResult]:
    uri = f"ws://{host}:{port}/live"
    out: list[LiveWsResult] = []
    for spec in _cases():
        for attempt in range(3):
            try:
                out.append(await _run_case(uri, spec, ticks, timeout))
                break
            except Exception as exc:
                if "HTTP 403" not in str(exc) or attempt >= 2:
                    raise
                await asyncio.sleep(0.10 * (attempt + 1))
        await asyncio.sleep(0.05)
    return out


def format_result(r: LiveWsResult) -> str:
    status = "PASS" if r.ok else "FAIL"
    return (
        f"{status} {r.case} actions={r.actions} "
        f"back_strafe_jump={r.back}/{r.strafe}/{r.jump} "
        f"neutral={r.neutral} sprint={r.sprint} attack={r.attack} "
        f"max_strafe={r.max_strafe if r.max_strafe is not None else '-'} "
        f"max_attack={r.max_attack if r.max_attack is not None else '-'} "
        f"strafe_flips={r.strafe_sign_flips} "
        f"final_yaw={r.final_yaw:.2f} final_pitch={r.final_pitch:.2f} "
        f"first={json.dumps(r.first, separators=(',', ':'))}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--ticks", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--no-load", action="store_true", help="do not load model/params before checking")
    args = ap.parse_args(argv)

    base = f"http://{args.host}:{args.port}"
    try:
        if not args.no_load:
            _post_json(f"{base}/live/load", {"model": args.model}, args.timeout)
            _post_json(f"{base}/live/params", {
                "enabled": True,
                "max_cps": 10,
                "max_rot_speed": 195,
                "arena": {"origin_x": 0, "origin_z": 0, "size_x": 40, "size_z": 40, "floor_y": 0},
            }, args.timeout)
        status = _get_json(f"{base}/status", args.timeout)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"FAIL daemon_unreachable error={exc}")
        return 2

    live = status.get("live") or {}
    print(f"LIVE model={live.get('model')} enabled={live.get('enabled')} rot={live.get('max_rot_speed')} cps={live.get('max_cps')}")
    try:
        results = asyncio.run(run_live_ws_check(args.host, args.port, args.ticks, args.timeout))
    except Exception as exc:
        print(f"FAIL websocket error={exc}")
        return 3
    for result in results:
        print(format_result(result))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
