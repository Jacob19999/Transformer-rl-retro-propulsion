"""
PID gain tuning script (Stage 18).

Implements training.md §8.4:
1) Linearize vehicle dynamics at a hover trim point (numerical Jacobians A, B).
2) Ziegler–Nichols ultimate gain method per loop (altitude, lateral-x/y, roll, pitch).
3) Grid-search refinement around ZN gains, scored by success rate (tiebreak: CEP).
4) Save best gains back to `simulation/configs/pid.yaml`.

This script is intentionally self-contained and uses only the existing simulation stack:
- `EDFLandingEnv` for episode rollouts and success/CEP scoring
- `VehicleDynamics.derivs` for plant linearization
- `PIDController` for policy rollout

Run (example):
    python -m simulation.training.scripts.tune_pid --episodes 50 --seed 0
"""

from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from simulation.config_loader import load_config
from simulation.dynamics.vehicle import CONTROL_DIM, STATE_DIM
from simulation.training.controllers.pid_controller import PIDController
from simulation.training.edf_landing_env import EDFLandingEnv


@dataclass(frozen=True, slots=True)
class LinearizationResult:
    A: np.ndarray  # (STATE_DIM, STATE_DIM)
    B: np.ndarray  # (STATE_DIM, CONTROL_DIM)
    eigenvalues: np.ndarray  # (STATE_DIM,)


@dataclass(frozen=True, slots=True)
class EpisodeStats:
    landed: bool
    crashed: bool
    out_of_bounds: bool
    cep: float
    steps: int
    termination_reason: str


@dataclass(frozen=True, slots=True)
class EvalSummary:
    success_rate: float
    mean_cep_success: float
    mean_steps: float


@dataclass(frozen=True, slots=True)
class LogConfig:
    quiet: bool = False
    episode_every: int = 1


def _configs_dir() -> Path:
    # tune_pid.py -> scripts/ -> training/ -> simulation/
    return Path(__file__).resolve().parents[2] / "configs"


def _deep_update(base: dict[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    for k, v in patch.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_update(dict(out[k]), v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def _make_deterministic_root_config(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    """Build a root env config dict suitable for tuning (deterministic disturbances)."""
    if cfg is None:
        base = _configs_dir()
        v = load_config(base / "default_vehicle.yaml")
        e = load_config(base / "default_environment.yaml")
        r = load_config(base / "reward.yaml")
        dr = load_config(base / "domain_randomization.yaml")
        cfg = {
            "vehicle": v.get("vehicle", v),
            "environment": e.get("environment", e),
            "reward": r.get("reward", r),
            **{k: v for k, v in dr.items() if k not in ("vehicle", "environment", "reward")},
        }
    else:
        cfg = dict(cfg.get("training", cfg))  # allow a top-level 'training' key

    # Override to be deterministic/stable for tuning.
    env = dict(cfg.get("environment", cfg.get("env", {})))
    atm = dict(env.get("atmosphere", {}))
    atm["randomize_T"] = 0.0
    atm["randomize_P"] = 0.0
    env["atmosphere"] = atm

    wind = dict(env.get("wind", {}))
    wind["mean_vector_range_lo"] = [0.0, 0.0, 0.0]
    wind["mean_vector_range_hi"] = [0.0, 0.0, 0.0]
    wind["turbulence_intensity"] = 0.0
    wind["gust_prob"] = 0.0
    wind["gust_magnitude_range"] = [0.0, 0.0]
    env["wind"] = wind
    cfg = dict(cfg)
    cfg["environment"] = env

    # Disable DR features for repeatability.
    cfg["actuator_delay"] = dict(cfg.get("actuator_delay", {}))
    cfg["actuator_delay"]["enabled"] = False
    cfg["obs_latency"] = dict(cfg.get("obs_latency", {}))
    cfg["obs_latency"]["enabled"] = False

    # Disable observation noise (Stage 13) for tuning repeatability.
    cfg["observation"] = dict(cfg.get("observation", {}))
    cfg["observation"]["noise_std"] = 0.0

    return cfg


def _load_pid_yaml(path: Path | None = None) -> dict[str, Any]:
    p = path or (_configs_dir() / "pid.yaml")
    return load_config(p)


def _pid_cfg_with_gains(
    base_pid_yaml: Mapping[str, Any],
    *,
    altitude: Mapping[str, float] | None = None,
    lateral_x: Mapping[str, float] | None = None,
    lateral_y: Mapping[str, float] | None = None,
    roll: Mapping[str, float] | None = None,
    pitch: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Return a PID config dict suitable for PIDController(config)."""
    root = dict(base_pid_yaml)
    pid = dict(root.get("pid", root))
    outer = dict(pid.get("outer_loop", {}))
    inner = dict(pid.get("inner_loop", {}))

    if altitude is not None:
        alt = dict(outer.get("altitude", {}))
        alt.update({k: float(v) for k, v in altitude.items()})
        outer["altitude"] = alt
    if lateral_x is not None:
        lx = dict(outer.get("lateral_x", {}))
        lx.update({k: float(v) for k, v in lateral_x.items()})
        outer["lateral_x"] = lx
    if lateral_y is not None:
        ly = dict(outer.get("lateral_y", {}))
        ly.update({k: float(v) for k, v in lateral_y.items()})
        outer["lateral_y"] = ly

    if roll is not None:
        r = dict(inner.get("roll", {}))
        r.update({k: float(v) for k, v in roll.items()})
        inner["roll"] = r
    if pitch is not None:
        p = dict(inner.get("pitch", {}))
        p.update({k: float(v) for k, v in pitch.items()})
        inner["pitch"] = p

    pid["outer_loop"] = outer
    pid["inner_loop"] = inner
    root["pid"] = pid
    return root


def _set_custom_initial_state(
    env: EDFLandingEnv,
    *,
    p_ned: np.ndarray,
    v_b: np.ndarray | None = None,
    q: np.ndarray | None = None,
    omega: np.ndarray | None = None,
    T: float | None = None,
    seed: int | None = None,
) -> None:
    """Overwrite the current episode initial state (after env.reset())."""
    p_ned = np.asarray(p_ned, dtype=float).reshape(3)
    v_b = np.zeros(3, dtype=float) if v_b is None else np.asarray(v_b, dtype=float).reshape(3)
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float) if q is None else np.asarray(q, dtype=float).reshape(4)
    omega = np.zeros(3, dtype=float) if omega is None else np.asarray(omega, dtype=float).reshape(3)

    if T is None:
        T = float(env.vehicle.mass * env.vehicle.g)

    init14 = np.concatenate([p_ned, v_b, q, omega, np.array([float(T)], dtype=float)], dtype=float)
    env.vehicle.reset(init14, seed=seed)
    env.h0 = max(float(-p_ned[2]), 0.0)


def _episode_rollout(env: EDFLandingEnv, pid_yaml: Mapping[str, Any], *, seed: int) -> EpisodeStats:
    obs, _info = env.reset(seed=seed)
    ctrl = PIDController(pid_yaml)
    ctrl.reset()

    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    steps = 0

    while not (terminated or truncated):
        action = ctrl.get_action(obs)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                obs, _reward, terminated, truncated, info = env.step(action)
        except Exception as e:  # pragma: no cover
            # Treat any numerical/physics error as a failed episode for tuning.
            terminated = True
            truncated = False
            info = {
                "landed": False,
                "crashed": True,
                "out_of_bounds": True,
                "cep": float("inf"),
                "termination_reason": f"exception:{type(e).__name__}",
            }
        steps += 1

    landed = bool(info.get("landed", False))
    crashed = bool(info.get("crashed", False))
    oob = bool(info.get("out_of_bounds", False))
    cep = float(info.get("cep", float("inf")))
    reason = str(info.get("termination_reason", ""))
    return EpisodeStats(
        landed=landed,
        crashed=crashed,
        out_of_bounds=oob,
        cep=cep,
        steps=int(steps),
        termination_reason=reason,
    )


def evaluate_pid(
    env_cfg: Mapping[str, Any] | None,
    pid_yaml: Mapping[str, Any],
    *,
    seeds: Iterable[int],
    log: LogConfig | None = None,
    label: str = "",
) -> EvalSummary:
    log = log or LogConfig()
    env = EDFLandingEnv(_make_deterministic_root_config(env_cfg))
    seeds_l = [int(s) for s in seeds]
    stats: list[EpisodeStats] = []
    for i, s in enumerate(seeds_l, start=1):
        ep = _episode_rollout(env, pid_yaml, seed=int(s))
        stats.append(ep)
        if (not log.quiet) and int(log.episode_every) > 0 and (i % int(log.episode_every) == 0):
            tag = f"{label} " if label else ""
            print(
                f"{tag}episode {i}/{len(seeds_l)} seed={s} "
                f"landed={ep.landed} crashed={ep.crashed} oob={ep.out_of_bounds} "
                f"cep={ep.cep:.3f} steps={ep.steps} reason={ep.termination_reason}",
                flush=True,
            )
    successes = [s for s in stats if s.landed]
    success_rate = float(len(successes) / max(1, len(stats)))
    mean_cep_success = float(np.mean([s.cep for s in successes])) if successes else float("inf")
    mean_steps = float(np.mean([s.steps for s in stats])) if stats else 0.0
    return EvalSummary(
        success_rate=success_rate,
        mean_cep_success=mean_cep_success,
        mean_steps=mean_steps,
    )


def _normalize_quat_in_state(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=float).copy()
    q = y[6:10]
    n = float(np.linalg.norm(q))
    if n > 1e-12:
        y[6:10] = q / n
    else:
        y[6:10] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return y


def linearize_dynamics(
    env_cfg: Mapping[str, Any] | None,
    *,
    hover_altitude_m: float = 5.0,
    dt: float = 1e-5,
    seed: int = 0,
) -> LinearizationResult:
    """Numerically linearize VehicleDynamics about a hover trim point."""
    env = EDFLandingEnv(_make_deterministic_root_config(env_cfg))
    _obs, _info = env.reset(seed=seed)

    # Hover trim (plant state/action, physical units)
    T_hover = float(env.vehicle.mass * env.vehicle.g)
    x0 = np.zeros(STATE_DIM, dtype=float)
    x0[0:3] = np.array([0.0, 0.0, -float(hover_altitude_m)], dtype=float)  # NED
    x0[3:6] = 0.0  # v_b
    x0[6:10] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)  # q
    x0[10:13] = 0.0  # omega
    x0[13] = T_hover  # internal thrust state
    x0[14:18] = 0.0  # servo actuals
    u0 = np.array([T_hover, 0.0, 0.0, 0.0, 0.0], dtype=float)

    # Ensure the env's vehicle is at x0 for consistent environment queries.
    env.vehicle.state = x0.copy()
    env.vehicle.time = 0.0

    A = np.zeros((STATE_DIM, STATE_DIM), dtype=float)
    B = np.zeros((STATE_DIM, CONTROL_DIM), dtype=float)

    for i in range(STATE_DIM):
        eps = float(dt * (1.0 + abs(float(x0[i]))))
        xp = x0.copy()
        xm = x0.copy()
        xp[i] += eps
        xm[i] -= eps
        xp = _normalize_quat_in_state(xp)
        xm = _normalize_quat_in_state(xm)
        fp = env.vehicle.derivs(xp, u0, 0.0)
        fm = env.vehicle.derivs(xm, u0, 0.0)
        A[:, i] = (fp - fm) / (2.0 * eps)

    for j in range(CONTROL_DIM):
        eps = float(dt * (1.0 + abs(float(u0[j]))))
        up = u0.copy()
        um = u0.copy()
        up[j] += eps
        um[j] -= eps
        fp = env.vehicle.derivs(x0, up, 0.0)
        fm = env.vehicle.derivs(x0, um, 0.0)
        B[:, j] = (fp - fm) / (2.0 * eps)

    eig = np.linalg.eigvals(A)
    return LinearizationResult(A=A, B=B, eigenvalues=eig)


def _find_peaks(y: np.ndarray) -> np.ndarray:
    """Return indices of simple local maxima."""
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.size < 3:
        return np.array([], dtype=int)
    dy1 = y[1:-1] - y[0:-2]
    dy2 = y[2:] - y[1:-1]
    peaks = np.where((dy1 > 0.0) & (dy2 <= 0.0))[0] + 1
    return peaks.astype(int)


def _estimate_period_and_amplitude(sig: np.ndarray, dt: float) -> tuple[float | None, float | None]:
    sig = np.asarray(sig, dtype=float).reshape(-1)
    peaks = _find_peaks(sig)
    if peaks.size < 3:
        return None, None
    # Use the last few peaks to estimate steady oscillation.
    sel = peaks[-min(5, peaks.size):]
    t = sel.astype(float) * float(dt)
    periods = np.diff(t)
    Tu = float(np.median(periods)) if periods.size > 0 else None
    amps = np.abs(sig[sel])
    amp = float(np.median(amps))
    return Tu, amp


def _zn_sweep_find_Ku_Tu(
    *,
    env_cfg: Mapping[str, Any] | None,
    base_pid_yaml: Mapping[str, Any],
    loop_name: str,
    kp_values: Iterable[float],
    sim_time_s: float,
    seed: int,
    hover_altitude_m: float,
    inner_hold: Mapping[str, Any] | None = None,
) -> tuple[float | None, float | None]:
    """Sweep Kp to find ultimate gain Ku and oscillation period Tu."""
    env = EDFLandingEnv(_make_deterministic_root_config(env_cfg))

    dt_policy = float(env.dt_policy)
    steps = int(math.ceil(sim_time_s / dt_policy))

    def make_init(loop: str) -> None:
        # Start from a clean reset, then overwrite state for repeatability.
        _obs, _info = env.reset(seed=seed)
        T_hover = float(env.vehicle.mass * env.vehicle.g)
        # Base hover state
        p = np.array([0.0, 0.0, -float(hover_altitude_m)], dtype=float)
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        # Excite only the loop under test with *small* perturbations so that
        # the controller stays in its linear region across the Kp sweep.
        if loop == "roll":
            ang = float(np.deg2rad(3.0))
            q = np.array([math.cos(ang / 2), math.sin(ang / 2), 0.0, 0.0], dtype=float)
        elif loop == "pitch":
            ang = float(np.deg2rad(3.0))
            q = np.array([math.cos(ang / 2), 0.0, math.sin(ang / 2), 0.0], dtype=float)
        elif loop == "lateral_x":
            p = np.array([0.3, 0.0, -float(hover_altitude_m)], dtype=float)
        elif loop == "lateral_y":
            p = np.array([0.0, 0.3, -float(hover_altitude_m)], dtype=float)
        elif loop == "altitude":
            p = np.array([0.0, 0.0, -(float(hover_altitude_m) + 0.3)], dtype=float)

        _set_custom_initial_state(env, p_ned=p, q=q, T=T_hover, seed=seed)

    def error_from_obs(loop: str, obs: np.ndarray, *, hover_target: float) -> float:
        o = np.asarray(obs, dtype=float).reshape(-1)
        if loop == "altitude":
            h_agl = float(o[16])
            return float(hover_target - h_agl)
        if loop == "lateral_x":
            return float(o[0])  # target offset body x
        if loop == "lateral_y":
            return float(o[1])  # target offset body y
        if loop == "roll":
            g_body = o[6:9]
            g2 = float(g_body[2]) if abs(float(g_body[2])) > 1e-9 else 1e-9
            roll_est = float(np.arctan2(float(g_body[1]), g2))
            return float(0.0 - roll_est)
        if loop == "pitch":
            g_body = o[6:9]
            g2 = float(g_body[2]) if abs(float(g_body[2])) > 1e-9 else 1e-9
            pitch_est = float(np.arctan2(-float(g_body[0]), g2))
            return float(0.0 - pitch_est)
        raise ValueError(f"Unknown loop {loop!r}")

    kp_values_l = list(kp_values)
    for idx, kp in enumerate(kp_values_l, start=1):
        # Build a controller config with only the loop under test having (Kp, Kd=0, Ki=0).
        gains_patch: dict[str, Any] = {"pid": {"outer_loop": {}, "inner_loop": {}}}
        if loop_name == "altitude":
            gains_patch["pid"]["outer_loop"]["altitude"] = {
                "Kp": float(kp),
                "Ki": 0.0,
                "Kd": 0.0,
                "target_h_agl": float(hover_altitude_m),
            }
        elif loop_name in ("lateral_x", "lateral_y"):
            gains_patch["pid"]["outer_loop"][loop_name] = {"Kp": float(kp), "Kd": 0.0}
        elif loop_name in ("roll", "pitch"):
            gains_patch["pid"]["inner_loop"][loop_name] = {"Kp": float(kp), "Kd": 0.0}
        else:
            raise ValueError(f"Unknown loop {loop_name!r}")

        # Zero-out all other gains unless explicitly held.
        pid_yaml = _pid_cfg_with_gains(
            base_pid_yaml,
            altitude={"Kp": 0.0, "Ki": 0.0, "Kd": 0.0, "target_h_agl": float(hover_altitude_m)},
            lateral_x={"Kp": 0.0, "Kd": 0.0},
            lateral_y={"Kp": 0.0, "Kd": 0.0},
            roll={"Kp": 0.0, "Kd": 0.0},
            pitch={"Kp": 0.0, "Kd": 0.0},
        )
        if inner_hold:
            pid_yaml = _deep_update(pid_yaml, inner_hold)
        pid_yaml = _deep_update(pid_yaml, gains_patch)

        make_init(loop_name)
        ctrl = PIDController(pid_yaml)
        ctrl.reset()

        obs = env._get_obs()
        sig = []
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        diverged = False

        for _ in range(steps):
            err = error_from_obs(loop_name, obs, hover_target=float(hover_altitude_m))
            sig.append(float(err))
            action = ctrl.get_action(obs)
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    obs, _r, terminated, truncated, info = env.step(action)
            except Exception:  # pragma: no cover
                diverged = True
                break

            # Safety: abort if state explodes (prevents environment/atmosphere errors).
            x = np.asarray(env.vehicle.state, dtype=float)
            if not np.all(np.isfinite(x)):
                diverged = True
                break
            p = x[0:3]
            v_b = x[3:6]
            omega = x[10:13]
            if (
                float(np.linalg.norm(p)) > 1e4
                or float(np.linalg.norm(v_b)) > 1e3
                or float(np.linalg.norm(omega)) > 200.0
            ):
                diverged = True
                break
            if terminated or truncated:
                break

        sig_arr = np.asarray(sig, dtype=float)
        if diverged or sig_arr.size < 50:
            print(
                f"[ZN] loop={loop_name} kp={float(kp):.6g} ({idx}/{len(kp_values_l)}) -> diverged/too_short",
                flush=True,
            )
            continue

        # Use the last 70% of the signal to avoid transients.
        start = int(0.3 * sig_arr.size)
        Tu, amp = _estimate_period_and_amplitude(sig_arr[start:], dt_policy)
        if Tu is None or amp is None:
            print(
                f"[ZN] loop={loop_name} kp={float(kp):.6g} ({idx}/{len(kp_values_l)}) -> no_oscillation_detected",
                flush=True,
            )
            continue

        # Heuristic "sustained oscillation": nontrivial amplitude and not terminated by crash.
        if amp > 1e-3 and not bool(info.get("crashed", False)):
            print(
                f"[ZN] loop={loop_name} Ku~{float(kp):.6g} Tu~{float(Tu):.3f}s amp~{float(amp):.6g}",
                flush=True,
            )
            return float(kp), float(Tu)

        print(
            f"[ZN] loop={loop_name} kp={float(kp):.6g} ({idx}/{len(kp_values_l)}) -> not_sustained (amp={float(amp):.6g})",
            flush=True,
        )
    return None, None


def ziegler_nichols_gains(Ku: float, Tu: float) -> tuple[float, float, float]:
    """ZN ultimate gain PID: Kp=0.6Ku, Ki=1.2Ku/Tu, Kd=0.075Ku*Tu."""
    Ku_f = float(Ku)
    Tu_f = max(float(Tu), 1e-9)
    Kp = 0.6 * Ku_f
    Ki = 1.2 * Ku_f / Tu_f
    Kd = 0.075 * Ku_f * Tu_f
    return float(Kp), float(Ki), float(Kd)


def grid_search_refinement(
    env_cfg: Mapping[str, Any] | None,
    base_pid_yaml: Mapping[str, Any],
    zn_pid_yaml: Mapping[str, Any],
    *,
    episodes_per_eval: int,
    seed0: int,
    span: float = 0.30,
    steps_per_axis: int = 5,
    mode: str = "coordinate",
    log: LogConfig | None = None,
) -> tuple[Mapping[str, Any], EvalSummary]:
    """Refine gains around ZN using either 'coordinate' or 'full' grid search."""
    log = log or LogConfig()
    multipliers = np.linspace(1.0 - span, 1.0 + span, int(steps_per_axis), dtype=float)

    def scaled(pid_yaml: Mapping[str, Any], *, m: dict[str, float]) -> Mapping[str, Any]:
        y = dict(pid_yaml)
        pid = dict(y.get("pid", y))
        outer = dict(pid.get("outer_loop", {}))
        inner = dict(pid.get("inner_loop", {}))
        for name in ("altitude", "lateral_x", "lateral_y"):
            if name in outer and name in m:
                section = dict(outer[name])
                for k in ("Kp", "Ki", "Kd"):
                    if k in section:
                        section[k] = float(section[k]) * float(m[name])
                outer[name] = section
        for name in ("roll", "pitch"):
            if name in inner and name in m:
                section = dict(inner[name])
                for k in ("Kp", "Kd"):
                    if k in section:
                        section[k] = float(section[k]) * float(m[name])
                inner[name] = section
        pid["outer_loop"] = outer
        pid["inner_loop"] = inner
        y["pid"] = pid
        return y

    score_calls = 0

    def score(pid_yaml: Mapping[str, Any]) -> EvalSummary:
        nonlocal score_calls
        score_calls += 1
        seeds = range(int(seed0), int(seed0) + int(episodes_per_eval))
        label = f"[grid#{score_calls}]"
        if not log.quiet:
            print(f"{label} scoring episodes={int(episodes_per_eval)}", flush=True)
        return evaluate_pid(env_cfg, pid_yaml, seeds=seeds, log=log, label=label)

    axes = ["altitude", "lateral_x", "lateral_y", "roll", "pitch"]
    best_yaml: Mapping[str, Any] = zn_pid_yaml
    best_score = score(best_yaml)

    if mode == "coordinate":
        current = zn_pid_yaml
        for ax in axes:
            local_best_yaml = current
            local_best = score(local_best_yaml)
            for mult in multipliers:
                cand = scaled(current, m={ax: float(mult)})
                s = score(cand)
                if (s.success_rate > local_best.success_rate) or (
                    math.isclose(s.success_rate, local_best.success_rate) and s.mean_cep_success < local_best.mean_cep_success
                ):
                    local_best_yaml = cand
                    local_best = s
            current = local_best_yaml
        best_yaml = current
        best_score = score(best_yaml)
        return best_yaml, best_score

    if mode != "full":
        raise ValueError("mode must be 'coordinate' or 'full'")

    # Full factorial across axes multipliers.
    for combo in itertools.product(multipliers, repeat=len(axes)):
        m = {ax: float(v) for ax, v in zip(axes, combo, strict=True)}
        cand = scaled(zn_pid_yaml, m=m)
        s = score(cand)
        if (s.success_rate > best_score.success_rate) or (
            math.isclose(s.success_rate, best_score.success_rate) and s.mean_cep_success < best_score.mean_cep_success
        ):
            best_yaml = cand
            best_score = s

    return best_yaml, best_score


def save_best_gains_to_pid_yaml(best_pid_yaml: Mapping[str, Any], *, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(best_pid_yaml), f, sort_keys=False)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage 18 PID gain tuning.")
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    p.add_argument("--hover_altitude_m", type=float, default=5.0, help="Hover altitude for trim/tuning.")
    p.add_argument("--linearize_dt", type=float, default=1e-5, help="Finite-difference step for linearization.")
    p.add_argument("--episodes", type=int, default=100, help="Episodes per evaluation (grid search scoring).")
    p.add_argument("--grid_span", type=float, default=0.30, help="± span around ZN gains (e.g. 0.30 = ±30%).")
    p.add_argument("--grid_steps", type=int, default=5, help="Steps per axis in grid search.")
    p.add_argument("--grid_mode", choices=["coordinate", "full"], default="coordinate")
    p.add_argument(
        "--out_pid_yaml",
        type=str,
        default=str(_configs_dir() / "pid.yaml"),
        help="Where to write tuned PID YAML.",
    )
    p.add_argument("--no_save", action="store_true", help="Do not overwrite pid.yaml.")
    p.add_argument("--quiet", action="store_true", help="Suppress per-episode logging.")
    p.add_argument(
        "--episode_log_every",
        type=int,
        default=1,
        help="Print every N episodes (1 = print all).",
    )
    args = p.parse_args(argv)
    log_cfg = LogConfig(quiet=bool(args.quiet), episode_every=int(args.episode_log_every))

    base_pid = _load_pid_yaml()
    env_cfg = None  # use repo defaults, overridden to deterministic for tuning

    lin = linearize_dynamics(
        env_cfg,
        hover_altitude_m=float(args.hover_altitude_m),
        dt=float(args.linearize_dt),
        seed=int(args.seed),
    )
    # ZN sweeps: tune inner loops first; then lateral; then altitude.
    # Per-loop Kp ranges are sized so the small perturbations (3 deg / 0.3 m)
    # keep the controller in its linear region across most of the sweep.
    kp_inner = np.geomspace(0.1, 20.0, num=35)
    kp_lateral = np.geomspace(0.01, 5.0, num=35)
    kp_alt = np.geomspace(0.01, 10.0, num=35)

    if not log_cfg.quiet:
        print("[ZN] starting sweeps", flush=True)

    Ku_roll, Tu_roll = _zn_sweep_find_Ku_Tu(
        env_cfg=env_cfg,
        base_pid_yaml=base_pid,
        loop_name="roll",
        kp_values=kp_inner,
        sim_time_s=15.0,
        seed=int(args.seed),
        hover_altitude_m=float(args.hover_altitude_m),
    )
    Ku_pitch, Tu_pitch = _zn_sweep_find_Ku_Tu(
        env_cfg=env_cfg,
        base_pid_yaml=base_pid,
        loop_name="pitch",
        kp_values=kp_inner,
        sim_time_s=15.0,
        seed=int(args.seed) + 1,
        hover_altitude_m=float(args.hover_altitude_m),
    )

    # Hold tuned inner loop gains while tuning lateral loops.
    inner_hold: dict[str, Any] = {"pid": {"inner_loop": {}}}
    if Ku_roll is not None and Tu_roll is not None:
        Kp, _Ki, Kd = ziegler_nichols_gains(Ku_roll, Tu_roll)
        inner_hold["pid"]["inner_loop"]["roll"] = {"Kp": Kp, "Kd": Kd}
    if Ku_pitch is not None and Tu_pitch is not None:
        Kp, _Ki, Kd = ziegler_nichols_gains(Ku_pitch, Tu_pitch)
        inner_hold["pid"]["inner_loop"]["pitch"] = {"Kp": Kp, "Kd": Kd}

    Ku_lx, Tu_lx = _zn_sweep_find_Ku_Tu(
        env_cfg=env_cfg,
        base_pid_yaml=base_pid,
        loop_name="lateral_x",
        kp_values=kp_lateral,
        sim_time_s=20.0,
        seed=int(args.seed) + 2,
        hover_altitude_m=float(args.hover_altitude_m),
        inner_hold=inner_hold,
    )
    Ku_ly, Tu_ly = _zn_sweep_find_Ku_Tu(
        env_cfg=env_cfg,
        base_pid_yaml=base_pid,
        loop_name="lateral_y",
        kp_values=kp_lateral,
        sim_time_s=20.0,
        seed=int(args.seed) + 3,
        hover_altitude_m=float(args.hover_altitude_m),
        inner_hold=inner_hold,
    )

    # Hold tuned inner+lateral while tuning altitude.
    outer_hold: dict[str, Any] = {"pid": {"outer_loop": {}}}
    if Ku_lx is not None and Tu_lx is not None:
        Kp, _Ki, Kd = ziegler_nichols_gains(Ku_lx, Tu_lx)
        outer_hold["pid"]["outer_loop"]["lateral_x"] = {"Kp": Kp, "Kd": Kd}
    if Ku_ly is not None and Tu_ly is not None:
        Kp, _Ki, Kd = ziegler_nichols_gains(Ku_ly, Tu_ly)
        outer_hold["pid"]["outer_loop"]["lateral_y"] = {"Kp": Kp, "Kd": Kd}

    hold_all = _deep_update(inner_hold, outer_hold)

    Ku_alt, Tu_alt = _zn_sweep_find_Ku_Tu(
        env_cfg=env_cfg,
        base_pid_yaml=base_pid,
        loop_name="altitude",
        kp_values=kp_alt,
        sim_time_s=20.0,
        seed=int(args.seed) + 4,
        hover_altitude_m=float(args.hover_altitude_m),
        inner_hold=hold_all,
    )

    # Build ZN gains (fallback to existing YAML values if Ku/Tu not found).
    pid = dict(base_pid.get("pid", base_pid))
    outer0 = dict(pid.get("outer_loop", {}))
    inner0 = dict(pid.get("inner_loop", {}))

    def fallback_outer(name: str) -> dict[str, float]:
        return {k: float(outer0.get(name, {}).get(k, 0.0)) for k in ("Kp", "Ki", "Kd") if k in outer0.get(name, {})}

    def fallback_inner(name: str) -> dict[str, float]:
        return {k: float(inner0.get(name, {}).get(k, 0.0)) for k in ("Kp", "Kd") if k in inner0.get(name, {})}

    alt_g = (
        {"Kp": ziegler_nichols_gains(Ku_alt, Tu_alt)[0], "Ki": ziegler_nichols_gains(Ku_alt, Tu_alt)[1], "Kd": ziegler_nichols_gains(Ku_alt, Tu_alt)[2]}
        if (Ku_alt is not None and Tu_alt is not None)
        else fallback_outer("altitude")
    )
    lx_g = (
        {"Kp": ziegler_nichols_gains(Ku_lx, Tu_lx)[0], "Kd": ziegler_nichols_gains(Ku_lx, Tu_lx)[2]}
        if (Ku_lx is not None and Tu_lx is not None)
        else {"Kp": float(outer0.get("lateral_x", {}).get("Kp", 0.0)), "Kd": float(outer0.get("lateral_x", {}).get("Kd", 0.0))}
    )
    ly_g = (
        {"Kp": ziegler_nichols_gains(Ku_ly, Tu_ly)[0], "Kd": ziegler_nichols_gains(Ku_ly, Tu_ly)[2]}
        if (Ku_ly is not None and Tu_ly is not None)
        else {"Kp": float(outer0.get("lateral_y", {}).get("Kp", 0.0)), "Kd": float(outer0.get("lateral_y", {}).get("Kd", 0.0))}
    )
    roll_g = (
        {"Kp": ziegler_nichols_gains(Ku_roll, Tu_roll)[0], "Kd": ziegler_nichols_gains(Ku_roll, Tu_roll)[2]}
        if (Ku_roll is not None and Tu_roll is not None)
        else fallback_inner("roll")
    )
    pitch_g = (
        {"Kp": ziegler_nichols_gains(Ku_pitch, Tu_pitch)[0], "Kd": ziegler_nichols_gains(Ku_pitch, Tu_pitch)[2]}
        if (Ku_pitch is not None and Tu_pitch is not None)
        else fallback_inner("pitch")
    )

    zn = _pid_cfg_with_gains(
        base_pid,
        altitude=alt_g,
        lateral_x=lx_g,
        lateral_y=ly_g,
        roll=roll_g,
        pitch=pitch_g,
    )

    best_yaml, best_score = grid_search_refinement(
        env_cfg=env_cfg,
        base_pid_yaml=base_pid,
        zn_pid_yaml=zn,
        episodes_per_eval=int(args.episodes),
        seed0=int(args.seed),
        span=float(args.grid_span),
        steps_per_axis=int(args.grid_steps),
        mode=str(args.grid_mode),
        log=log_cfg,
    )

    if not bool(args.no_save):
        save_best_gains_to_pid_yaml(best_yaml, out_path=Path(args.out_pid_yaml))

    # Minimal stdout summary (useful when running manually).
    print("Linearization eigenvalues (A):")
    print(lin.eigenvalues)
    print("Best score:")
    print(best_score)
    print(f"Saved: {not bool(args.no_save)} -> {args.out_pid_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

