from system.lib import minescript as ms
import time, math, random
from dataclasses import dataclass


@dataclass
class HumanLookConfig:
    # Speed mapping (deg/sec)
    min_speed: float = 35.0
    max_speed: float = 260.0
    min_angle: float = 5.0
    max_angle: float = 140.0

    # Duration clamps (sec)
    min_duration: float = 0.045
    max_duration: float = 0.75

    # Path curvature (fraction of distance -> lateral offset)
    max_curve_intensity: float = 0.14

    # Overshoot behavior
    base_overshoot: float = 0.012
    max_overshoot: float = 0.040
    overshoot_bias: float = 0.55

    # Jitter (hand tremor) during motion
    jitter_deg: float = 0.04
    jitter_smooth: float = 0.85
    jitter_scale_small_moves: float = 0.35

    # Cadence / timing
    target_hz: float = 120.0
    step_jitter: float = 0.22

    # End micro-settle
    micro_settle_deg: float = 0.06
    micro_settle_steps: int = 3

    # General
    deadzone_deg: float = 0.05
    pitch_min: float = -89.9
    pitch_max: float = 89.9

    # Gating (base)
    no_overshoot_deg: float = 8.0
    no_curve_deg: float = 10.0
    lock_cone_deg: float = 0.85
    lock_time: float = 0.08
    lock_servo_speed: float = 110.0

    # -------- Urgent mode tuning (scales/targets) --------
    urgent_speed_mult_max: float = 1.8        # up to +80% speed
    urgent_duration_mult_min: float = 0.55    # as low as 55% of base duration
    urgent_curve_scale: float = 0.25          # reduce curvature aggressively
    urgent_no_curve_deg: float = 22.0         # raise straight-line threshold
    urgent_no_overshoot_deg: float = 18.0     # disable overshoot more often
    urgent_lock_cone_deg: float = 0.60        # tighter hold cone
    urgent_lock_time: float = 0.05            # shorter hold
    urgent_lock_servo_speed: float = 220.0    # faster tiny corrections


CFG = HumanLookConfig()

# ----- Helpers -----

def _wrap_deg(a: float) -> float:
    """Wrap to [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0

def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def _hypot2(a: float, b: float) -> float:
    return math.sqrt(a * a + b * b)

def _min_jerk(t: float) -> float:
    """Quintic minimum-jerk easing (0..1 -> 0..1)."""
    t3 = t * t * t
    t4 = t3 * t
    t5 = t4 * t
    return 10.0 * t3 - 15.0 * t4 + 6.0 * t5

def _map_speed(ang: float, cfg: HumanLookConfig) -> float:
    """Angle -> deg/sec, easing between min and max."""
    if ang <= cfg.min_angle:
        return cfg.min_speed * (0.6 + 0.8 * (ang / max(1e-6, cfg.min_angle)))
    if ang >= cfg.max_angle:
        return cfg.max_speed
    r = (ang - cfg.min_angle) / (cfg.max_angle - cfg.min_angle)
    r = r ** 0.7
    return cfg.min_speed + (cfg.max_speed - cfg.min_speed) * r

def _lateral_curve(ang: float, cfg: HumanLookConfig) -> float:
    """How strong the curved path should be (as a fraction of ang)."""
    r = _clamp((ang - cfg.min_angle) / (cfg.max_angle - cfg.min_angle), 0.0, 1.0)
    return cfg.max_curve_intensity * (r ** 1.2)

def _overshoot_frac_raw(ang: float, cfg: HumanLookConfig) -> float:
    """Baseline overshoot fraction (distance-aware + random)."""
    r = _clamp((ang - cfg.min_angle) / (cfg.max_angle - cfg.min_angle), 0.0, 1.0)
    base = cfg.base_overshoot + (cfg.max_overshoot - cfg.base_overshoot) * (r ** 0.9)
    return base * (0.75 + 0.5 * random.random())

def _sleep_step(dt: float, cfg: HumanLookConfig):
    jitter = 1.0 + cfg.step_jitter * (random.random() - 0.5) * 2.0
    time.sleep(max(0.001, dt * jitter))

# ----- Core movement -----

def _move_segment(a0: float, b0: float, dy: float, dp: float, duration: float,
                  lateral_frac: float, cfg: HumanLookConfig,
                  jitter_state: dict | None = None,
                  jitter_scale_override: float | None = None):
    """
    Move from (a0,b0) by (dy,dp) in 'duration' seconds with curved path & smooth jitter.
    lateral_frac: 0..~0.3 times total distance, applied orthogonal to direction.
    """
    hz = cfg.target_hz
    dt = 1.0 / hz
    steps = max(1, int(duration * hz))

    ang = max(_hypot2(dy, dp), 1e-6)
    ux, uy = dy / ang, dp / ang
    px, py = -uy, ux

    if jitter_state is None:
        jitter_state = {"jy": 0.0, "jp": 0.0}
    jy, jp = jitter_state["jy"], jitter_state["jp"]

    base_scale = cfg.jitter_scale_small_moves if ang < 12.0 else 1.0
    if jitter_scale_override is not None:
        base_scale = _clamp(jitter_scale_override, 0.0, 1.0)

    for i in range(1, steps + 1):
        t = i / steps
        s = _min_jerk(t)

        byaw = dy * s
        bpitch = dp * s

        bell = math.sin(math.pi * s)
        lat = lateral_frac * ang * bell
        lyaw = px * lat
        lpitch = py * lat

        # Smooth jitter
        jy = cfg.jitter_smooth * jy + (1.0 - cfg.jitter_smooth) * (random.random() - 0.5)
        jp = cfg.jitter_smooth * jp + (1.0 - cfg.jitter_smooth) * (random.random() - 0.5)
        jitter_yaw = cfg.jitter_deg * jy * base_scale
        jitter_pitch = cfg.jitter_deg * jp * base_scale

        yaw = _wrap_deg(a0 + byaw + lyaw + jitter_yaw)
        pitch = _clamp(b0 + bpitch + lpitch + jitter_pitch, cfg.pitch_min, cfg.pitch_max)

        ms.player_set_orientation(yaw, pitch)
        _sleep_step(dt, cfg)

    jitter_state["jy"] = jy
    jitter_state["jp"] = jp
    return jitter_state


def _urgency_value(urgent) -> float:
    """Coerce urgent flag to a float in [0,1]."""
    if isinstance(urgent, bool):
        return 1.0 if urgent else 0.0
    try:
        return _clamp(float(urgent), 0.0, 1.0)
    except Exception:
        return 0.0


# ----- Public API -----

def look(target_yaw: float, target_pitch: float,
         cfg: HumanLookConfig = CFG,
         urgent: float | bool = 0.0):
    """
    Human-like mouse-look towards (target_yaw, target_pitch) in degrees.
    'urgent' in [0,1] (or True/False) makes motion faster, straighter, lower-overshoot,
    and tighter at the end — useful for tight cuts between nodes.
    """
    u = _urgency_value(urgent)

    a, b = ms.player_orientation()
    dy = _wrap_deg(target_yaw - a)
    tp = _clamp(target_pitch, cfg.pitch_min, cfg.pitch_max)
    dp = tp - b

    ang = _hypot2(dy, dp)
    if ang < cfg.deadzone_deg:
        return

    # --- Thresholds blend with urgency ---
    no_curve_thresh = (1.0 - u) * cfg.no_curve_deg + u * cfg.urgent_no_curve_deg
    no_overshoot_thresh = (1.0 - u) * cfg.no_overshoot_deg + u * cfg.urgent_no_overshoot_deg

    no_curve = ang <= no_curve_thresh
    no_overshoot = ang <= no_overshoot_thresh

    # Seam guard widens with urgency
    wrap_band = (1.0 - u) * 18.0 + u * 24.0
    near_wrap = abs(abs(dy) - 180.0) <= wrap_band

    # --- Speed / duration ---
    speed = _map_speed(ang, cfg) * (1.0 + u * (cfg.urgent_speed_mult_max - 1.0))
    base_T = _clamp(ang / max(1e-6, speed), cfg.min_duration, cfg.max_duration)
    # compress duration under urgency
    duration = base_T * (1.0 - u * (1.0 - cfg.urgent_duration_mult_min))
    duration *= (0.9 + 0.2 * random.random())

    # --- Curvature ---
    if no_curve:
        lateral_frac = 0.0
    else:
        lateral_frac = _lateral_curve(ang, cfg) * (0.85 + 0.3 * random.random())
        lateral_frac *= (1.0 - u * (1.0 - cfg.urgent_curve_scale))  # reduce with urgency

    # --- Overshoot ---
    if no_overshoot or near_wrap:
        overshoot_y = 0.0
        overshoot_p = 0.0
    else:
        over_frac = _overshoot_frac_raw(ang, cfg) * (1.0 - 0.85 * u)  # heavily reduced with urgency
        axis_mix = cfg.overshoot_bias if abs(dy) >= abs(dp) else (cfg.overshoot_bias * 0.8)
        overshoot_y = dy * over_frac * (axis_mix + 0.22 * (random.random() - 0.5))
        overshoot_p = dp * over_frac * ((1.0 - axis_mix) + 0.22 * (random.random() - 0.5))

    # --- Main move (lower jitter if urgent) ---
    main_jitter_scale = 1.0 - 0.85 * u
    jitter_state = _move_segment(
        a, b,
        dy + overshoot_y, dp + overshoot_p,
        duration,
        lateral_frac,
        cfg,
        jitter_state=None,
        jitter_scale_override=main_jitter_scale
    )

    # --- Correction segment ---
    corr_dy = -overshoot_y
    corr_dp = -overshoot_p
    corr_ang = _hypot2(corr_dy, corr_dp)

    if corr_ang >= cfg.deadzone_deg * 0.5:
        corr_speed = max(cfg.min_speed * 0.7, _map_speed(corr_ang, cfg) * (0.6 + 0.25 * u))
        corr_T = _clamp(corr_ang / corr_speed,
                        cfg.min_duration * (0.55 - 0.10 * u),
                        cfg.min_duration * (1.45 - 0.25 * u))
        _move_segment(
            _wrap_deg(a + dy + overshoot_y),
            _clamp(b + dp + overshoot_p, cfg.pitch_min, cfg.pitch_max),
            corr_dy, corr_dp,
            corr_T,
            lateral_frac * 0.35,
            cfg,
            jitter_state=jitter_state,
            jitter_scale_override=0.25 * (1.0 - u)  # close to jitterless if urgent
        )

    # --- Snap-to if still off (always straight, jitterless) ---
    sy, sp = ms.player_orientation()
    rem_dy = _wrap_deg(target_yaw - sy)
    rem_dp = _clamp(target_pitch, cfg.pitch_min, cfg.pitch_max) - sp
    rem_ang = _hypot2(rem_dy, rem_dp)

    if rem_ang > cfg.deadzone_deg * 0.6:
        lock_speed = (1.0 - u) * cfg.lock_servo_speed + u * cfg.urgent_lock_servo_speed
        snap_T = _clamp(rem_ang / lock_speed,
                        cfg.min_duration * (0.45 - 0.10 * u),
                        cfg.min_duration * (0.9 - 0.15 * u))
        _move_segment(
            sy, sp, rem_dy, rem_dp,
            snap_T,
            0.0,
            cfg,
            jitter_state=jitter_state,
            jitter_scale_override=0.0
        )

    # --- Short lock hold (tiny, fast servo) ---
    lock_deadline = time.time() + ((1.0 - u) * cfg.lock_time + u * cfg.urgent_lock_time)
    lock_cone = (1.0 - u) * cfg.lock_cone_deg + u * cfg.urgent_lock_cone_deg
    lock_speed = (1.0 - u) * cfg.lock_servo_speed + u * cfg.urgent_lock_servo_speed

    while time.time() < lock_deadline:
        cy, cp = ms.player_orientation()
        edy = _wrap_deg(target_yaw - cy)
        edp = _clamp(target_pitch, cfg.pitch_min, cfg.pitch_max) - cp
        eang = _hypot2(edy, edp)
        if eang <= lock_cone:
            _sleep_step(1.0 / cfg.target_hz, cfg)
            continue
        nib_T = _clamp(eang / lock_speed,
                       cfg.min_duration * 0.30,
                       cfg.min_duration * 0.55)
        _move_segment(
            cy, cp, edy, edp,
            nib_T,
            0.0,
            cfg,
            jitter_state=jitter_state,
            jitter_scale_override=0.0
        )

    # Micro-settle (kept short; slightly reduced when urgent)
    for k in range(max(1, int(cfg.micro_settle_steps - round(1.0 * u)))):
        sy, sp = ms.player_orientation()
        jitter = cfg.micro_settle_deg * (0.6 ** k)
        nudge_y = (random.random() - 0.5) * 2.0 * jitter
        nudge_p = (random.random() - 0.5) * 2.0 * jitter * 0.7
        _move_segment(sy, sp, nudge_y, nudge_p,
                      cfg.min_duration * (0.32 - 0.06 * u),
                      0.0, cfg, jitter_scale_override=0.25 * (1.0 - u))

    ms.player_set_orientation(_wrap_deg(target_yaw), _clamp(target_pitch, cfg.pitch_min, cfg.pitch_max))
