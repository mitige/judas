"""Aides communes aux tests sim_ref."""

import math
import shutil
import subprocess
from pathlib import Path

from sim_ref import Action, BoxingConfig, BoxingMatch, HumanizationConfig
from sim_ref.constants import PLAYER_EYE_HEIGHT

ROOT = Path(__file__).resolve().parent.parent

HAS_CPU_CHECK_COMPILER = bool(shutil.which("g++") or shutil.which("cl"))


def build_cpu_check(out_dir: Path, define: str | None = None) -> Path:
    """Compile tools/cpu_check.cpp (logique exacte du kernel) -> binaire.

    g++ si présent, sinon cl (MSVC, mis sur le PATH par scripts/env.bat).
    """
    name = "judas_cpu_check_" + (define or "float")
    src = ROOT / "tools" / "cpu_check.cpp"
    include = ROOT / "sim" / "csrc"
    if shutil.which("g++"):
        out = out_dir / name
        cmd = ["g++", "-O2", "-I", str(include)]
        if define:
            cmd.append(f"-D{define}")
        cmd += ["-o", str(out), str(src)]
    else:
        out = out_dir / f"{name}.exe"
        cmd = ["cl", "/nologo", "/O2", "/EHsc", f"/I{include}"]
        if define:
            cmd.append(f"/D{define}")
        cmd += [f"/Fe:{out}", f"/Fo{out_dir}\\", str(src)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"compilation cpu_check échouée ({cmd[0]}):\n"
            f"{proc.stdout}\n{proc.stderr}")
    return out


def free_humanization() -> HumanizationConfig:
    """Aucune limite humaine (tests de physique pure)."""
    return HumanizationConfig(max_cps=20.0, max_rot_speed=360.0, action_delay=0)


def make_match(**cfg_kwargs) -> BoxingMatch:
    cfg = BoxingConfig(
        humanization=(free_humanization(), free_humanization()),
        **cfg_kwargs,
    )
    return BoxingMatch(cfg)


def aim_action(attacker, target, **kwargs) -> Action:
    """Action dont dyaw/dpitch pointent exactement le centre de la cible."""
    dx = target.x - attacker.x
    dz = target.z - attacker.z
    dy = (target.y + 0.9) - (attacker.y + PLAYER_EYE_HEIGHT)
    dist_h = math.sqrt(dx * dx + dz * dz)
    yaw_to = math.degrees(math.atan2(-dx, dz))
    pitch_to = math.degrees(-math.atan2(dy, dist_h))
    dyaw = _wrap_degrees(yaw_to - attacker.yaw)
    dpitch = pitch_to - attacker.pitch
    return Action(dyaw=dyaw, dpitch=dpitch, **kwargs)


def _wrap_degrees(angle: float) -> float:
    angle = math.fmod(angle, 360.0)
    if angle >= 180.0:
        angle -= 360.0
    if angle < -180.0:
        angle += 360.0
    return angle
