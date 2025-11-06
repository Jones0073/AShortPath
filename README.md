# AShortPath

A lightweight path-finding **Minecraft bot** for the **Minescript** modloader.
AShortPath uses the **Theta*** algorithm to produce smooth, short routes in voxel worlds—often shorter than classic A* paths thanks to line-of-sight shortcuts.

> Credit & inspiration: [hyshenn/thetaSTAR](https://github.com/hyshenn/thetaSTAR)

---

## Features

* 🚀 Fast, greedy-any-angle routing with **Theta***
* 🧭 Simple, minimal API (`path_find`, `path_walk_to`)
* 🧱 Works directly with Minecraft `(x, y, z)` block coordinates
* 🧠 Automatically handles “no path” cases

---

## Quick Start

Create a small Minescript script and run it in your world:

```python
from AShortPath.thetastar import path_walk_to, path_find
import minescript

# Start at player position; set your target coordinates here:
start = minescript.player_position()
goal = (600, 72, 1200)

path = path_find(start, goal)

if not path:
    minescript.echo("No path found.")
else:
    minescript.echo(f"Path length: {len(path)}")
    # Walk the computed path.
    path_walk_to(path=path, distance=1)
```

### What you’ll see

* If a path exists, the bot walks it and prints the path length in chat.
* If no path exists (e.g., blocked or unreachable), you get a clear message instead.

---

## Installation

AShortPath is a pure-script drop-in.

1. **Copy** the `AShortPath/` folder into your Minescript scripts directory (or any location that is on your Minescript/Python path).
2. **Import** it from your script:

   ```python
   from AShortPath.thetastar import path_find, path_walk_to
   ```

> If your environment uses a custom loader, make sure `AShortPath` is importable (e.g., added to `PYTHONPATH` or your Minescript modules directory).

---

## API

### `path_find(start, goal) -> list[tuple[int, int, int]] | None`

Compute a path between two block coordinates.

* **start**: `(x, y, z)` – typically `minescript.player_position()`
* **goal**: `(x, y, z)` – your destination
* **returns**: A list of `(x, y, z)` waypoints, including start/goal, or `None` if unreachable.

### `path_walk_to(path, distance: int = 1) -> None`

Walk an existing path.

* **path**: The list returned by `path_find`
* **distance**: Stop when within this many blocks of the goal (default `1`)

> Separation of concerns: compute once with `path_find`, then hand the path to `path_walk_to`. This lets you inspect/modify the path before moving (e.g., visualize or trim it).

---

## How It Works (In Short)

AShortPath uses **Theta***, an any-angle variant of A*:

* It performs line-of-sight checks between a node and its parent, letting it **“cut corners”** across free space.
* The result is typically **shorter** and **smoother** than grid-constrained A* paths.
* The world is treated as a voxel grid using collision checks to avoid solid blocks.

---

## Tips & Gotchas

* **Coordinate system**: Use integer `(x, y, z)` block positions.
* **Performance**: Planning cost grows with search radius and obstacle density. Prefer realistic goals over extreme distances when possible.

---

## Contributing

Bug reports and improvements are welcome!
Please keep contributions focused and well-scoped (small PRs are easier to review and test).

---

## Credits

* **Algorithm**: Theta*
* **Inspiration & basis**: [hyshenn/thetaSTAR](https://github.com/hyshenn/thetaSTAR)