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
    max_curve_intensity: float = 0.14      # tighter by default

    # Overshoot behavior
    base_overshoot: float = 0.012          # slightly smaller baseline
    max_overshoot: float = 0.040
    overshoot_bias: float = 0.55

    # Jitter (hand tremor) during motion
    jitter_deg: float = 0.04
    jitter_smooth: float = 0.85
    jitter_scale_small_moves: float = 0.35 # more suppression on small corrections

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

    # New: “tight lock” & gating
    no_overshoot_deg: float = 8.0          # below this, force 0 overshoot
    no_curve_deg: float = 10.0             # below this, path stays straight
    lock_cone_deg: float = 0.85            # hold aim inside this cone
    lock_time: float = 0.08                # seconds of cone hold
    lock_servo_speed: float = 110.0        # deg/sec for tiny servo nibbles


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

    # Orthogonal unit vector in yaw-pitch plane
    ang = max(_hypot2(dy, dp), 1e-6)
    ux, uy = dy / ang, dp / ang
    px, py = -uy, ux

    # Smooth jitter state
    if jitter_state is None:
        jitter_state = {"jy": 0.0, "jp": 0.0}
    jy, jp = jitter_state["jy"], jitter_state["jp"]

    # Scale jitter for smaller moves or override for lock phase
    base_scale = cfg.jitter_scale_small_moves if ang < 12.0 else 1.0
    if jitter_scale_override is not None:
        base_scale = jitter_scale_override

    for i in range(1, steps + 1):
        t = i / steps
        s = _min_jerk(t)

        # Base along-track position
        byaw = dy * s
        bpitch = dp * s

        # Lateral curve (bell-shaped along the trajectory)
        bell = math.sin(math.pi * s)  # 0 at ends, 1 in middle
        lat = lateral_frac * ang * bell
        lyaw = px * lat
        lpitch = py * lat

        # Smooth jitter (IIR filtered random)
        jy = cfg.jitter_smooth * jy + (1.0 - cfg.jitter_smooth) * (random.random() - 0.5)
        jp = cfg.jitter_smooth * jp + (1.0 - cfg.jitter_smooth) * (random.random() - 0.5)
        jitter_yaw = cfg.jitter_deg * jy * base_scale
        jitter_pitch = cfg.jitter_deg * jp * base_scale

        yaw = a0 + byaw + lyaw + jitter_yaw
        pitch = _clamp(b0 + bpitch + lpitch + jitter_pitch, cfg.pitch_min, cfg.pitch_max)

        # Wrap yaw to Minecraft’s expected range
        yaw = _wrap_deg(yaw)

        ms.player_set_orientation(yaw, pitch)
        _sleep_step(dt, cfg)

    # store jitter state if needed by caller
    jitter_state["jy"] = jy
    jitter_state["jp"] = jp
    return jitter_state

# ----- Public API -----

def look(target_yaw: float, target_pitch: float, cfg: HumanLookConfig = CFG):
    """
    Human-like mouse-look towards (target_yaw, target_pitch) in degrees.
    Keeps your original signature but adds smarter motion.
    """
    a, b = ms.player_orientation()
    # Compute shortest yaw delta & clamped pitch delta
    dy = _wrap_deg(target_yaw - a)
    tp = _clamp(target_pitch, cfg.pitch_min, cfg.pitch_max)
    dp = tp - b

    ang = _hypot2(dy, dp)
    if ang < cfg.deadzone_deg:
        return

    # ----- Adaptive gating: disable risky behaviors near the target -----
    no_overshoot = ang <= cfg.no_overshoot_deg
    no_curve = ang <= cfg.no_curve_deg

    # If we are near yaw wrap (+/-180°), kill overshoot to avoid long spins
    wrap_proximity = abs(abs(dy) - 180.0)
    near_wrap = wrap_proximity <= 18.0  # within 18° of the seam

    # Distance->speed->duration mapping with light randomness
    speed = _map_speed(ang, cfg)
    base_T = _clamp(ang / max(1e-6, speed), cfg.min_duration, cfg.max_duration)
    # Small moves: slightly shorter; big moves: allow full base_T ±10%
    lerp = 0.9 + 0.2 * random.random()
    duration = base_T * (0.92 * lerp if ang < 15.0 else lerp)

    # Curved path intensity
    lateral_frac = 0.0 if no_curve else _lateral_curve(ang, cfg) * (0.85 + 0.3 * random.random())

    # Overshoot
    if no_overshoot or near_wrap:
        overshoot_y = 0.0
        overshoot_p = 0.0
    else:
        over_frac = _overshoot_frac_raw(ang, cfg)
        # slightly taper overshoot if pitch is dominant to avoid vertical wobble
        axis_mix = cfg.overshoot_bias if abs(dy) >= abs(dp) else (cfg.overshoot_bias * 0.8)
        overshoot_y = dy * over_frac * (axis_mix + 0.22 * (random.random() - 0.5))
        overshoot_p = dp * over_frac * ((1.0 - axis_mix) + 0.22 * (random.random() - 0.5))

    # 1) Main move to (possibly) overshot point
    jitter_state = _move_segment(
        a, b,
        dy + overshoot_y, dp + overshoot_p,
        duration,
        lateral_frac,
        cfg,
        jitter_state=None
    )

    # 2) Micro-correction back to target (short & straighter)
    corr_dy = -overshoot_y
    corr_dp = -overshoot_p
    corr_ang = _hypot2(corr_dy, corr_dp)

    if corr_ang >= cfg.deadzone_deg * 0.5:
        corr_speed = max(cfg.min_speed * 0.7, _map_speed(corr_ang, cfg) * 0.6)
        corr_T = _clamp(corr_ang / corr_speed, cfg.min_duration * 0.55, cfg.min_duration * 1.45)
        _move_segment(
            _wrap_deg(a + dy + overshoot_y),
            _clamp(b + dp + overshoot_p, cfg.pitch_min, cfg.pitch_max),
            corr_dy, corr_dp,
            corr_T,
            lateral_frac * 0.35,  # straighter correction
            cfg,
            jitter_state=jitter_state
        )

    # 3) Snap-to if still off a hair (fast, straight, jitterless)
    sy, sp = ms.player_orientation()
    rem_dy = _wrap_deg(target_yaw - sy)
    rem_dp = _clamp(target_pitch, cfg.pitch_min, cfg.pitch_max) - sp
    rem_ang = _hypot2(rem_dy, rem_dp)

    if rem_ang > cfg.deadzone_deg * 0.6:
        snap_speed = max(cfg.min_speed, cfg.lock_servo_speed)
        snap_T = _clamp(rem_ang / snap_speed, cfg.min_duration * 0.45, cfg.min_duration * 0.9)
        _move_segment(
            sy, sp, rem_dy, rem_dp,
            snap_T,
            0.0,  # no curve
            cfg,
            jitter_state=jitter_state,
            jitter_scale_override=0.0  # suppress jitter for final line-up
        )

    # 4) Very short “lock cone” servo: keep aim steady so we don't drift past nodes
    #    (human-like tiny hand hold). Time-capped.
    lock_deadline = time.time() + cfg.lock_time
    while time.time() < lock_deadline:
        cy, cp = ms.player_orientation()
        edy = _wrap_deg(target_yaw - cy)
        edp = _clamp(target_pitch, cfg.pitch_min, cfg.pitch_max) - cp
        eang = _hypot2(edy, edp)
        if eang <= cfg.lock_cone_deg:
            # hold without motion but continue frame pacing (prevents robotic hard stop)
            _sleep_step(1.0 / cfg.target_hz, cfg)
            continue
        # tiny corrective nibble straight towards target, no curve, no jitter
        nib_speed = cfg.lock_servo_speed
        nib_T = _clamp(eang / nib_speed, cfg.min_duration * 0.35, cfg.min_duration * 0.6)
        _move_segment(
            cy, cp, edy, edp,
            nib_T,
            0.0,
            cfg,
            jitter_state=jitter_state,
            jitter_scale_override=0.0
        )

    # 5) A couple of micro-settle nudges (very small, very quick) for human feel
    for k in range(cfg.micro_settle_steps):
        sy, sp = ms.player_orientation()
        jitter = cfg.micro_settle_deg * (0.6 ** k)
        nudge_y = (random.random() - 0.5) * 2.0 * jitter
        nudge_p = (random.random() - 0.5) * 2.0 * jitter * 0.7
        _move_segment(sy, sp, nudge_y, nudge_p,
                      cfg.min_duration * 0.32,
                      0.0, cfg, jitter_scale_override=0.25)

    # Final exact set (prevents drift from accumulating)
    ms.player_set_orientation(_wrap_deg(target_yaw), _clamp(target_pitch, cfg.pitch_min, cfg.pitch_max))
