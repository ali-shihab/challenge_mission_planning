from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Any

Point2 = Tuple[float, float]
GridCell = Tuple[int, int]


@dataclass
class GridConfig:
    resolution_m: float = 0.5
    inflation_m: float = 0.6
    bounds_margin_m: float = 2.0


class OccupancyGrid2D:
    """
    2D occupancy grid built from scenario obstacles projected onto XY.
    Obstacles are treated conservatively: any cuboid whose vertical span
    intersects the flight band is projected into XY and inflated.
    """

    def __init__(
        self,
        *,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        resolution_m: float,
    ) -> None:
        self.min_x = float(min_x)
        self.max_x = float(max_x)
        self.min_y = float(min_y)
        self.max_y = float(max_y)
        self.resolution_m = float(resolution_m)

        self.width = max(1, int(math.ceil((self.max_x - self.min_x) / self.resolution_m)) + 1)
        self.height = max(1, int(math.ceil((self.max_y - self.min_y) / self.resolution_m)) + 1)

        self.occupied: set[GridCell] = set()

    def in_bounds(self, cell: GridCell) -> bool:
        i, j = cell
        return 0 <= i < self.width and 0 <= j < self.height

    def world_to_grid(self, x: float, y: float) -> GridCell:
        i = int(round((x - self.min_x) / self.resolution_m))
        j = int(round((y - self.min_y) / self.resolution_m))
        i = min(max(i, 0), self.width - 1)
        j = min(max(j, 0), self.height - 1)
        return (i, j)

    def grid_to_world(self, cell: GridCell) -> Point2:
        i, j = cell
        x = self.min_x + i * self.resolution_m
        y = self.min_y + j * self.resolution_m
        return (x, y)

    def is_occupied(self, cell: GridCell) -> bool:
        return cell in self.occupied

    def is_free(self, cell: GridCell) -> bool:
        return self.in_bounds(cell) and cell not in self.occupied

    def mark_rect_occupied(self, *, min_x: float, max_x: float, min_y: float, max_y: float) -> None:
        c0 = self.world_to_grid(min_x, min_y)
        c1 = self.world_to_grid(max_x, max_y)

        imin, imax = sorted((c0[0], c1[0]))
        jmin, jmax = sorted((c0[1], c1[1]))

        for i in range(imin, imax + 1):
            for j in range(jmin, jmax + 1):
                if self.in_bounds((i, j)):
                    self.occupied.add((i, j))

    def neighbors8(self, cell: GridCell) -> List[Tuple[GridCell, float]]:
        i, j = cell
        out: List[Tuple[GridCell, float]] = []
        for di, dj, cost in [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        ]:
            nxt = (i + di, j + dj)
            if self.is_free(nxt):
                out.append((nxt, cost))
        return out

    def line_is_free_world(self, a: Point2, b: Point2) -> bool:
        """
        Conservative grid line-of-sight check using dense interpolation.
        """
        dist = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(2, int(math.ceil(dist / max(1e-6, self.resolution_m * 0.5))))
        for k in range(n + 1):
            t = k / n
            x = a[0] + t * (b[0] - a[0])
            y = a[1] + t * (b[1] - a[1])
            if not self.is_free(self.world_to_grid(x, y)):
                return False
        return True


def _obstacle_vertical_span(obs: Dict[str, Any]) -> Tuple[float, float]:
    zc = float(obs["z"])
    h = float(obs["h"])
    return (zc - 0.5 * h, zc + 0.5 * h)


def _viewpoint_positions_xy(scenario: Dict[str, Any]) -> List[Point2]:
    out: List[Point2] = []
    for vp in scenario["viewpoint_poses"].values():
        out.append((float(vp["x"]), float(vp["y"])))
    return out


def build_occupancy_grid_from_scenario(
    scenario: Dict[str, Any],
    *,
    start_xy: Point2,
    flight_z: float,
    config: GridConfig,
) -> OccupancyGrid2D:
    """
    Build a 2D occupancy grid by projecting only obstacles whose vertical
    span could interfere with the current flight altitude.
    """
    points_xy = [start_xy] + _viewpoint_positions_xy(scenario)

    min_x = min(p[0] for p in points_xy)
    max_x = max(p[0] for p in points_xy)
    min_y = min(p[1] for p in points_xy)
    max_y = max(p[1] for p in points_xy)

    for obs in scenario.get("obstacles", {}).values():
        ox = float(obs["x"])
        oy = float(obs["y"])
        w = float(obs["w"])
        d = float(obs["d"])
        min_x = min(min_x, ox - 0.5 * w)
        max_x = max(max_x, ox + 0.5 * w)
        min_y = min(min_y, oy - 0.5 * d)
        max_y = max(max_y, oy + 0.5 * d)

    min_x -= config.bounds_margin_m
    max_x += config.bounds_margin_m
    min_y -= config.bounds_margin_m
    max_y += config.bounds_margin_m

    grid = OccupancyGrid2D(
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        resolution_m=config.resolution_m,
    )

    z_band_half = max(0.5, config.inflation_m)

    for obs in scenario.get("obstacles", {}).values():
        z0, z1 = _obstacle_vertical_span(obs)

        # Conservative vertical filtering: only ignore an obstacle if it is
        # clearly far from the current flight altitude.
        if flight_z < (z0 - z_band_half) or flight_z > (z1 + z_band_half):
            continue

        ox = float(obs["x"])
        oy = float(obs["y"])
        w = float(obs["w"])
        d = float(obs["d"])

        half_w = 0.5 * w + config.inflation_m
        half_d = 0.5 * d + config.inflation_m

        grid.mark_rect_occupied(
            min_x=ox - half_w,
            max_x=ox + half_w,
            min_y=oy - half_d,
            max_y=oy + half_d,
        )

    return grid
