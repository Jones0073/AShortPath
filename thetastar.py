import minescript
import math, heapq, threading, time
from AShortPath.rotation import look
from itertools import count

AIRLIKE = {"minecraft:air", "minecraft:light"}
LIQUIDS = {"minecraft:water", "minecraft:lava"}
IGNORE_BLOCKS = {
    "minecraft:air","minecraft:water","minecraft:lava","minecraft:carpet","minecraft:stairs","minecraft:sign","minecraft:flower","minecraft:torch",
    "minecraft:ladder","minecraft:vine","minecraft:grass","minecraft:snow","minecraft:sapling","minecraft:button","minecraft:pressure_plate",
    "minecraft:tripwire","minecraft:candle","minecraft:bell","minecraft:banner","minecraft:skull","minecraft:head"
}
MAX_VERTICAL_STEP = 1
EYE_HEIGHT = 1.62
BODY_RADIUS = 0.3        # ~player half-width
BODY_SAMPLE_LEVELS = (0.2, 0.9, 1.5)  # feet->shoulder samples for body LOS
ENTITY_HEIGHT = 1.8      # player/capsule height


# ---- Utilities --------------------------------------------------------------

def is_bottom_slab(b: str) -> bool:
    """True for bottom (non-double) slabs (pass-through headroom logic)."""
    return ("slab" in b) and ("double" not in b) and (
        "bottom" in b or "type=bottom" in b or "_bottom" in b
    )

def is_passable(block_id: str) -> bool:
    """
    Occupancy test: can the player's body occupy this block space?
    Air, ignorable thin blocks, and *bottom slabs* are passable.
    (Full blocks, liquids, top slabs, etc. are not.)
    """
    if not block_id:
        return True
    b = block_id.lower()
    if b in AIRLIKE:
        return True
    if b in IGNORE_BLOCKS:
        return True
    if is_bottom_slab(b):
        return True
    return False

def is_supportive(block_id: str) -> bool:
    """
    Floor support test: can you stand on this block?
    Supportive if solid OR bottom slab (half-block floor).
    Liquids and thin deco do NOT count as support.
    """
    if not block_id:
        return False
    b = block_id.lower()
    if b in LIQUIDS or b in AIRLIKE:
        return False
    if b in IGNORE_BLOCKS:
        return False
    # bottom slab works as a (half-height) floor
    if is_bottom_slab(b):
        return True
    # anything else that's not airlike/ignorable/liquid is solid
    return True

def get_block(x, y, z, cache):
    key = (x, y, z)
    if key not in cache:
        cache[key] = minescript.getblock(x, y, z)
    return cache[key]

def has_clearance(x, y, z, cache) -> bool:
    """Player clearance: body block + head block must both be passable."""
    current = get_block(x, y, z, cache)
    above   = get_block(x, y + 1, z, cache)
    return is_passable(current) and is_passable(above)


def LOS(start, end, cache) -> bool:
    """
    Line-of-sight AND straight-line traversability:
      - DDA voxel walk for eye ray (with your slab/ignore rules)
      - Body capsule LOS (multiple heights + XZ offsets)
      - Ground/clearance check along XZ that forbids >1-block steps

    Returns False if anything in between would block *movement* even if the eye ray would be clear.
    """
    x0, y0, z0 = start
    x1, y1, z1 = end

    # --- helpers -------------------------------------------------------------

    EXACT_IGNORES = IGNORE_BLOCKS if isinstance(IGNORE_BLOCKS, set) else set(IGNORE_BLOCKS)
    PREFIX_IGNORES = tuple()  # keep empty unless you truly need prefixes

    def is_airlike(b: str) -> bool:
        return (b in AIRLIKE)

    def is_ignored(b: str) -> bool:
        return (b in EXACT_IGNORES or b.startswith(PREFIX_IGNORES))

    def is_solid(block: str) -> bool:
        """Conservative 'solid for movement' test (slabs counted as solid for floors)."""
        if not block:
            return False
        b = block.lower()
        if is_airlike(b) or is_ignored(b):
            return False
        # Bottom slabs are floors (solid for traversability). Fences/panes etc are solid here.
        return True

    def cell_blocks_eye(gx, gy, gz, t_here, t_next, sy, dy) -> bool:
        """Block test for the *eye ray* (slabs can be see-through near voxel top)."""
        block = get_block(gx, gy, gz, cache)
        if not block:
            return False
        b = block.lower()
        if is_airlike(b) or is_ignored(b):
            return False
        if is_bottom_slab(b):
            # allow eye to pass if near the very top of this voxel interval
            t_sample = min(t_next, 1.0) - 1e-6
            if t_sample < t_here:
                t_sample = t_here + 1e-6
            y_at = sy + dy * t_sample
            local_y = y_at - math.floor(y_at)
            if local_y > 0.8:
                return False
        return True

    def cell_blocks_body(gx, gy, gz) -> bool:
        """Block test for *body rays* and movement (slabs are solid)."""
        return is_solid(get_block(gx, gy, gz, cache))

    def ray_clear(sx, sy, sz, ex, ey, ez, for_eye: bool) -> bool:
        """3D DDA; for_eye applies slab near-top exception, otherwise slabs are solid."""
        dx, dy, dz = ex - sx, ey - sy, ez - sz
        if dx == 0 and dy == 0 and dz == 0:
            return True

        # small push so we start inside the first voxel
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        nx, ny, nz = dx / length, dy / length, dz / length
        sx += nx * 1e-6; sy += ny * 1e-6; sz += nz * 1e-6

        gx, gy, gz = math.floor(sx), math.floor(sy), math.floor(sz)
        gx_end, gy_end, gz_end = math.floor(ex), math.floor(ey), math.floor(ez)

        step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
        step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)
        step_z = 1 if dz > 0 else (-1 if dz < 0 else 0)

        inf = float('inf')
        tMaxX = ((gx + (step_x > 0)) - sx) / dx if step_x != 0 else inf
        tMaxY = ((gy + (step_y > 0)) - sy) / dy if step_y != 0 else inf
        tMaxZ = ((gz + (step_z > 0)) - sz) / dz if step_z != 0 else inf

        tDeltaX = abs(1.0 / dx) if step_x != 0 else inf
        tDeltaY = abs(1.0 / dy) if step_y != 0 else inf
        tDeltaZ = abs(1.0 / dz) if step_z != 0 else inf

        t_here = 0.0

        while True:
            if for_eye:
                if cell_blocks_eye(gx, gy, gz, t_here, min(tMaxX, tMaxY, tMaxZ), sy, dy):
                    return False
            else:
                if cell_blocks_body(gx, gy, gz):
                    return False

            if tMaxX < tMaxY and tMaxX < tMaxZ:
                gx += step_x; t_here = tMaxX; tMaxX += tDeltaX
            elif tMaxY < tMaxZ:
                gy += step_y; t_here = tMaxY; tMaxY += tDeltaY
            else:
                gz += step_z; t_here = tMaxZ; tMaxZ += tDeltaZ

            if gx == gx_end and gy == gy_end and gz == gz_end:
                # check the final voxel too
                if for_eye:
                    if cell_blocks_eye(gx, gy, gz, t_here, 1.0, sy, dy):
                        return False
                else:
                    if cell_blocks_body(gx, gy, gz):
                        return False
                break

            if t_here > 1.0 + 1e-6:
                break

        return True

    def offsets_for_radius(r):
        # 4 corners + center to approximate circular footprint
        return [(-r, -r), (-r, r), (r, -r), (r, r), (0.0, 0.0)]

    # --- 1) true LOS (eye) ---------------------------------------------------
    eye_y0 = y0 + EYE_HEIGHT
    eye_y1 = y1 + EYE_HEIGHT

    for ox, oz in offsets_for_radius(0.25):
        if not ray_clear(x0 + ox, eye_y0, z0 + oz, x1 + ox, eye_y1, z1 + oz, for_eye=True):
            return False  # eye is blocked

    # --- 2) body capsule LOS (reject "see it but body can't pass") ----------
    for h in BODY_SAMPLE_LEVELS:
        sy = y0 + h
        ey = y1 + h
        for ox, oz in offsets_for_radius(BODY_RADIUS):
            if not ray_clear(x0 + ox, sy, z0 + oz, x1 + ox, ey, z1 + oz, for_eye=False):
                return False  # body hits something at some height

    # --- 3) traversability/step check along XZ -------------------------------
    # forbid straight-line travel if at any crossed column the vertical step to usable floor
    # exceeds STEP_MAX or the column lacks ENTITY_HEIGHT clearance.
    def find_floor_y(bx: int, y_hint: float, bz: int):
        """Find a reasonable 'top solid' around y_hint (scan a small band)."""
        yh = int(math.floor(y_hint))
        # scan down then up a little; enough to catch 2-block risers
        for d in range(0, 4):
            yy = yh - d
            if is_solid(get_block(bx, yy, bz, cache)):
                return yy
        for d in range(1, 4):
            yy = yh + d
            if is_solid(get_block(bx, yy, bz, cache)):
                return yy
        return None

    def has_clearance_from_floor(bx: int, floor_y: int, height: float) -> bool:
        """Require empty space above floor for the entity height."""
        top_needed = floor_y + math.ceil(height)
        for yy in range(floor_y + 1, top_needed + 1):
            b = get_block(bx, yy, bz, cache)
            if b and not is_airlike(b.lower()) and not is_ignored(b.lower()):
                return False
        return True

    dx, dz = x1 - x0, z1 - z0
    steps = max(1, int(math.ceil(max(abs(dx), abs(dz)))))  # 2D supercover-ish sampling
    prev_floor_top = None

    for i in range(1, steps + 1):
        t = i / float(steps)
        xi = x0 + dx * t
        zi = z0 + dz * t
        yi_hint = y0 + (y1 - y0) * t

        # check multiple XZ offsets to cover the body radius
        worst_step_ok = True
        for ox, oz in offsets_for_radius(BODY_RADIUS):
            bx, bz = int(math.floor(xi + ox)), int(math.floor(zi + oz))
            floor_y = find_floor_y(bx, yi_hint, bz)
            if floor_y is None:
                return False  # unknown/support missing -> treat as blocked

            # clearance above floor
            if not has_clearance_from_floor(bx, floor_y, ENTITY_HEIGHT):
                return False

            floor_top = floor_y + 1.0  # top face you stand on
            if prev_floor_top is not None:
                if abs(floor_top - prev_floor_top) > MAX_VERTICAL_STEP + 1e-6:
                    worst_step_ok = False
            prev_floor_top = floor_top

        if not worst_step_ok:
            return False  # requires >1 block climb somewhere along the line

    return True



# ---- A* Node ---------------------------------------------------------------

HEURISTIC_WEIGHT = 1.5  # set slightly >1.0 to be greedier if desired (e.g., 1.05)
# TODO: really high right now to cut time on long distances by alot (60-70% speedup)

class Node:
    __slots__ = ("pos","parent","G","H")
    def __init__(self, pos, parent=None):
        self.pos = tuple(map(math.floor, pos))
        self.parent = parent
        self.G = 0.0
        self.H = 0.0

    @property
    def F(self):
        return self.G + HEURISTIC_WEIGHT * self.H

    def __lt__(self, other):
        # Not relied upon (we tie-break in heap entries), but keep deterministic:
        return (self.F, self.H, self.pos) < (other.F, other.H, other.pos)

    def heuristic(self, goal):
        px, py, pz = self.pos
        gx, gy, gz = goal
        return math.sqrt((px - gx)**2 + (py - gy)**2 + (pz - gz)**2)

    def _diagonal_clearance_ok(self, cx, cy, cz, nx, ny, nz, cache) -> bool:
        dx, dy, dz = nx - cx, ny - cy, nz - cz

        # Only care when there's a diagonal component in XZ.
        if abs(dx) == 1 and abs(dz) == 1:
            base_y = min(cy, ny)

            if dy > 0:
                # Step UP by 1: allow a solid riser at (nx, cy, nz).
                # Require the other orthogonal column to be clear so we don't corner-cut.
                if not has_clearance(cx, base_y, cz + dz, cache):
                    return False
            else:
                # Same level or stepping down: both orthogonal adjacents must be clear.
                if not has_clearance(cx + dx, base_y, cz, cache):
                    return False
                if not has_clearance(cx, base_y, cz + dz, cache):
                    return False

        # No extra checks for cardinal moves or pure vertical; existing checks suffice.
        return True


    def is_walkable(self, pos, cache) -> bool:
        nx, ny, nz = pos
        cx, cy, cz = self.pos
        dy = ny - cy

        if abs(dy) > MAX_VERTICAL_STEP:
            return False

        below = get_block(nx, ny - 1, nz, cache)
        if not is_supportive(below):
            return False

        # Body + head clearance at target
        if not has_clearance(nx, ny, nz, cache):
            return False
        
        # If stepping UP, we also need headroom above current position to rise into
        if dy > 0 and not has_clearance(cx, cy + 1, cz, cache):
            return False

        # Corner-cut prevention applies to ALL moves (same-level and step up/down).
        if not self._diagonal_clearance_ok(cx, cy, cz, nx, ny, nz, cache):
            return False

        return True

    def neighbors(self, cache):
        # Minecraft-aware stepping: try same-level, then +1 (step up), then -1 (step down)
        x, y, z = self.pos
        result = []

        dirs = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]

        def add_if_walkable(px, py, pz):
            p = (px, py, pz)
            if self.is_walkable(p, cache):
                # minescript.echo(f"Walkable neighbor: {p}")
                result.append(p)
                return True
            return False

        for dx, dz in dirs:
            # 1) same level
            if add_if_walkable(x + dx, y, z + dz):
                continue

            # 2) step up by 1 if allowed
            if MAX_VERTICAL_STEP >= 1 and add_if_walkable(x + dx, y + 1, z + dz):
                continue

            # 3) step down by 1 if allowed
            if MAX_VERTICAL_STEP >= 1 and add_if_walkable(x + dx, y - 1, z + dz):
                continue

        return result


# ---- Helpers ---------------------------------------------------------------

def reconstruct_path(node):
    path = []
    while node:
        path.append(node.pos)
        node = node.parent
    return path[::-1]

def smooth_path(path, cache):
    """Simple string-pulling using LOS; keeps endpoints."""
    if len(path) <= 2:
        return path
    smoothed = [path[0]]
    anchor = 0
    for i in range(2, len(path)):
        if not LOS(path[anchor], path[i], cache):
            smoothed.append(path[i - 1])
            anchor = i - 1
    smoothed.append(path[-1])
    return smoothed

# ---- Pathfinding -----------------------------------------------------------

def path_find(start, goal, do_smooth=True):
    start_time = time.time()
    start = tuple(map(math.floor, start))
    goal  = tuple(map(math.floor, goal))

    cache = {}
    start_node = Node(start)
    start_node.H = start_node.heuristic(goal)

    # (F, H, tie, Node) — tie breaker keeps heap operations predictable
    pq = []
    _tie = count()
    heapq.heappush(pq, (start_node.F, start_node.H, next(_tie), start_node))

    closed = set()
    node_map = {start: start_node}

    while pq:
        _, _, _, current = heapq.heappop(pq)
        if current.pos in closed:
            continue

        if current.pos == goal:
            raw_path = reconstruct_path(current)
            final_path = smooth_path(raw_path, cache) if do_smooth else raw_path
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(goal, start)))
            minescript.echo(
                f"Pathfinding took {time.time() - start_time:.3f}s | "
                f"Nodes: {len(final_path)} | Distance: {dist:.2f}"
            )
            return final_path

        closed.add(current.pos)

        for npos in current.neighbors(cache):
            if npos in closed:
                continue

            # LOS-based parent shortcut (Theta*-style)
            if current.parent and current.parent.pos[1] == npos[1] and LOS(current.parent.pos, npos, cache):
                parent_candidate = current.parent
                base = parent_candidate.pos
            else:
                parent_candidate = current
                base = current.pos

            step_cost = math.sqrt(sum((a - b) ** 2 for a, b in zip(base, npos)))
            tentative_G = parent_candidate.G + step_cost

            neighbor = node_map.get(npos)
            if neighbor is None:
                neighbor = Node(npos, parent_candidate)
                neighbor.G = tentative_G
                neighbor.H = neighbor.heuristic(goal)
                node_map[npos] = neighbor
                heapq.heappush(pq, (neighbor.F, neighbor.H, next(_tie), neighbor))
            elif tentative_G + 1e-9 < neighbor.G:
                neighbor.parent = parent_candidate
                neighbor.G = tentative_G
                # Push an updated entry; old one will be skipped when popped (closed-set guards this).
                heapq.heappush(pq, (neighbor.F, neighbor.H, next(_tie), neighbor))

    raise ValueError("No path found")

# ---- Movement --------------------------------------------------------------

def jump_loop(path_ref):
    last_jump_time = 0
    while True:
        time.sleep(0.01)
        now = time.time()
        if now - last_jump_time < 0.25 or not path_ref or not path_ref[0]:
            continue

        px, py, pz = map(float, minescript.player_position())
        foot_y = math.floor(py)

        # nearest point in XY (ignoring Y for lateral guidance)
        path = path_ref[0]
        nearest_index = min(
            range(len(path)),
            key=lambda i: (px - (path[i][0] + 0.5)) ** 2 + (pz - (path[i][2] + 0.5)) ** 2
        )

        # next higher waypoint relative to the player's current foot height
        nxt = next((p for p in path[nearest_index:] if p[1] > math.floor(py)), None)
        if not nxt:
            continue

        dx, dz = px - nxt[0], pz - nxt[2]
        dy = nxt[1] - py
        if dx*dx + dy*dy + dz*dz <= 4 and dy > 0:
            block_below = minescript.getblock(math.floor(px), foot_y - 1, math.floor(pz))
            # jump only if standing on something solid (avoid jumping in air/liquid)
            if is_supportive(block_below):
                minescript.player_press_jump(True)
                time.sleep(0.35)
                minescript.player_press_jump(False)
                last_jump_time = time.time()

# -- pathing -------------------------------------------------------------------
from typing import List, Tuple, Optional


Vec3 = Tuple[float, float, float]
Block = Tuple[float, float, float]

def _center(b: Block) -> Vec3:
    x, y, z = b
    return (x + 0.5, y, z + 0.5)

def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

def _mul(a: Vec3, s: float) -> Vec3:
    return (a[0]*s, a[1]*s, a[2]*s)

def _dot(a: Vec3, b: Vec3) -> float:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def _len(a: Vec3) -> float:
    return math.sqrt(_dot(a, a))

def _norm(a: Vec3) -> Vec3:
    L = _len(a)
    return (0.0, 0.0, 0.0) if L == 0.0 else (a[0]/L, a[1]/L, a[2]/L)

def _closest_point_on_segment(p: Vec3, a: Vec3, b: Vec3) -> Tuple[Vec3, float]:
    """Return (closest_point, t) where t in [0,1] along AB (XZ-plane weighted, mild Y)."""
    ab = (b[0]-a[0], (b[1]-a[1])*0.25, b[2]-a[2])  # de-emphasize Y for ground nav
    ap = (p[0]-a[0], (p[1]-a[1])*0.25, p[2]-a[2])
    ab2 = _dot(ab, ab)
    t = 0.0 if ab2 == 0 else max(0.0, min(1.0, _dot(ap, ab)/ab2))
    return _add(a, _mul(ab, t)), t

def _advance_along_path(points: List[Vec3], idx: int, t: float, ds: float) -> Tuple[int, float, Vec3]:
    """
    Move forward ds along the polyline starting from segment (idx, t).
    Returns (new_idx, new_t, position).
    """
    i, u = idx, t
    pos = None
    while ds > 0 and i < len(points)-1:
        a, b = points[i], points[i+1]
        seg = _sub(b, a)
        seg_len = _len(seg)
        if seg_len == 0:
            i += 1
            u = 0.0
            continue
        rem = (1.0 - u) * seg_len
        if ds < rem:
            u += ds / seg_len
            pos = _add(a, _mul(seg, u))
            ds = 0
        else:
            ds -= rem
            i += 1
            u = 0.0
            pos = b
    if pos is None:
        pos = points[min(i, len(points)-1)]
    return i, u, pos

def path_walk_to(
    goal: Optional[Tuple[float, float, float]] = None,
    path: Optional[List[Block]] = None,
    distance: float = 1.0,
    look_ahead: int = 1,          # kept for API compatibility (unused in new logic)
    lookahead_distance: float = 2.5,  # pure-pursuit radius in blocks
    accel: float = 0.20,
    min_threshold: float = 0.05,
    max_pitch_down: float = 35.0,     # don't stare too far down
):
    """
    Improved path follower with:
    - projection-based segment advancement (prevents circling back)
    - pure-pursuit target at arc-length 'lookahead_distance'
    - movement computed from geometry (not current view), so smoothed 'look' can't induce loops
    """
    # Start/keep jump helper
    if not getattr(path_walk_to, "_jump_running", False):
        path_ref = [None]
        path_walk_to._path_ref = path_ref
        # threading.Thread(target=jump_loop, args=(path_ref,), daemon=True).start()
        path_walk_to._jump_running = True
    else:
        path_ref = path_walk_to._path_ref

    # Acquire path
    if path is None:
        if goal is None:
            return
        start = tuple(map(float, minescript.player_position()))
        path = path_find(start, goal)

    # Pre-center nodes once
    centers: List[Vec3] = [ _center(b) for b in path ]
    if not centers:
        return

    # publish for jump loop
    path_ref[0] = path

    # State along polyline: segment index i and param t in [0,1]
    i = 0
    t = 0.0

    # Vel smoothing
    forward_v = 0.0
    strafe_v  = 0.0

    # Stop when within 'distance' of final target
    final_target = centers[-1]

    # Small hysteresis radius for node passing
    pass_eps = max(0.35, distance * 0.35)

    while True:
        px, py, pz = map(float, minescript.player_position())
        p = (px, py, pz)

        # Goal reached?
        if _len(_sub(final_target, p)) <= max(distance, 0.75):
            break

        # Ensure valid segment
        if i >= len(centers) - 1:
            # We're on (last node, none). Snap to last and finish.
            i = len(centers) - 2
            t = 1.0

        a, b = centers[i], centers[i+1]

        # Closest point on current segment; if past the end, advance segment.
        closest, t_on = _closest_point_on_segment(p, a, b)
        if t_on >= 1.0 - 1e-4:
            # We are at/past this segment end; advance if more segments remain
            if i < len(centers) - 2:
                i += 1
                t = 0.0
                continue
            else:
                # last segment, keep t
                t = 1.0
        else:
            t = max(t, t_on)  # never move backward along the segment

        # If we are clearly past the *node center* too, apply an extra pass condition
        if _len(_sub(b, p)) + pass_eps < _len(_sub(a, p)):
            # Player is closer to next node than current — allow advancing
            if i < len(centers) - 2:
                i += 1
                t = 0.0

        # Pure pursuit: lookahead arc-length from our (i,t) state
        li, lt, target = _advance_along_path(centers, i, t, lookahead_distance)

        # Compute desired yaw/pitch to target (cap pitch)
        to = _sub(target, p)
        flat = math.hypot(to[0], to[2])
        yaw = math.degrees(math.atan2(to[2], to[0])) - 90.0
        pitch = -math.degrees(math.atan2(to[1], max(1e-6, flat)))
        pitch = max(-max_pitch_down, min(60.0, pitch))  # clamp

        # Apply look (your implementation may smooth; that’s fine now)
        look(yaw, pitch)

        # Movement: drive towards *target direction*, independent of camera smoothing
        move_dir = _norm((to[0], 0.0, to[2]))  # keep ground movement planar
        if move_dir == (0.0, 0.0, 0.0):
            # Avoid division issues at target; release keys
            minescript.player_press_forward(False)
            minescript.player_press_backward(False)
            minescript.player_press_left(False)
            minescript.player_press_right(False)
            continue

        # Derive forward/right from desired yaw (not current camera to avoid feedback loop)
        yaw_rad = math.radians(yaw + 90.0)
        forward_dir = (math.cos(yaw_rad), 0.0, math.sin(yaw_rad))
        right_dir   = (math.sin(yaw_rad), 0.0, -math.cos(yaw_rad))

        forward_target = _dot(forward_dir, move_dir)
        strafe_target  = _dot(right_dir,   move_dir)

        forward_v += (forward_target - forward_v) * accel
        strafe_v  += (strafe_target  - strafe_v ) * accel

        # Deadzone & key presses
        fpos = forward_v >  min_threshold
        fneg = forward_v < -min_threshold
        spos = strafe_v  >  min_threshold
        sneg = strafe_v  < -min_threshold

        minescript.player_press_forward(fpos)
        minescript.player_press_backward(fneg)
        minescript.player_press_left(sneg)
        minescript.player_press_right(spos)

    # Release keys at the end
    minescript.player_press_forward(False)
    minescript.player_press_backward(False)
    minescript.player_press_left(False)
    minescript.player_press_right(False)
