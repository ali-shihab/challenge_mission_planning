from __future__ import annotations

import math
from typing import Dict, List, Tuple

from planners.grid_map import OccupancyGrid2D

Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]


def cells_to_world_path(grid: OccupancyGrid2D, cells: List[Tuple[int, int]]) -> List[Point2]:
    return [grid.grid_to_world(c) for c in cells]


def simplify_world_path(grid: OccupancyGrid2D, path_xy: List[Point2]) -> List[Point2]:
    """
    Greedy line-of-sight simplification.
    """
    if len(path_xy) <= 2:
        return path_xy[:]

    simplified: List[Point2] = [path_xy[0]]
    i = 0
    while i < len(path_xy) - 1:
        j = len(path_xy) - 1
        while j > i + 1:
            if grid.line_is_free_world(path_xy[i], path_xy[j]):
                break
            j -= 1
        simplified.append(path_xy[j])
        i = j

    return simplified


def densify_stride(path_xy: List[Point2], stride: int) -> List[Point2]:
    if stride <= 1 or len(path_xy) <= 2:
        return path_xy[:]

    out = [path_xy[0]]
    for i in range(1, len(path_xy) - 1, stride):
        out.append(path_xy[i])
    if out[-1] != path_xy[-1]:
        out.append(path_xy[-1])
    return out


def heading_between(a: Point2, b: Point2, fallback_yaw: float) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return fallback_yaw
    return math.atan2(dy, dx)


def build_subgoals_from_xy_path(
    *,
    path_xy: List[Point2],
    start_z: float,
    goal_z: float,
    final_yaw: float,
) -> List[Dict[str, float]]:
    """
    Convert an XY path into 3D subgoals.
    Intermediate subgoals face along the path.
    Final subgoal uses the viewpoint yaw.
    """
    if not path_xy:
        return []

    n = len(path_xy)
    out: List[Dict[str, float]] = []

    for i, xy in enumerate(path_xy):
        if n == 1:
            z = goal_z
        else:
            alpha = i / (n - 1)
            z = start_z + alpha * (goal_z - start_z)

        if i < n - 1:
            yaw = heading_between(path_xy[i], path_xy[i + 1], final_yaw)
        else:
            yaw = final_yaw

        out.append({
            "x": float(xy[0]),
            "y": float(xy[1]),
            "z": float(z),
            "yaw": float(yaw),
        })

    return out

