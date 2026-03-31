from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Tuple

from planners.grid_map import GridCell, OccupancyGrid2D


def heuristic(a: GridCell, b: GridCell) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.hypot(dx, dy)


def reconstruct_path(came_from: Dict[GridCell, GridCell], current: GridCell) -> List[GridCell]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def nearest_free_cell(grid: OccupancyGrid2D, start: GridCell, max_radius: int = 20) -> Optional[GridCell]:
    if grid.is_free(start):
        return start

    si, sj = start
    for r in range(1, max_radius + 1):
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if max(abs(di), abs(dj)) != r:
                    continue
                c = (si + di, sj + dj)
                if grid.is_free(c):
                    return c
    return None


def astar_search(
    grid: OccupancyGrid2D,
    start: GridCell,
    goal: GridCell,
) -> Optional[List[GridCell]]:
    start = nearest_free_cell(grid, start)
    goal = nearest_free_cell(grid, goal)

    if start is None or goal is None:
        return None

    open_heap: List[Tuple[float, int, GridCell]] = []
    heapq.heappush(open_heap, (heuristic(start, goal), 0, start))

    came_from: Dict[GridCell, GridCell] = {}
    g_score: Dict[GridCell, float] = {start: 0.0}
    closed: set[GridCell] = set()
    tie = 1

    while open_heap:
        _, _, current = heapq.heappop(open_heap)

        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            return reconstruct_path(came_from, current)

        for nxt, step_cost in grid.neighbors8(current):
            tentative = g_score[current] + step_cost
            if tentative < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative
                f = tentative + heuristic(nxt, goal)
                heapq.heappush(open_heap, (f, tie, nxt))
                tie += 1

    return None
