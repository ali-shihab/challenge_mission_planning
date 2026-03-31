from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, List, Tuple, Any


Point3 = Tuple[float, float, float]


@dataclass
class OrderedMission:
    ordered_ids: List[str]
    original_ids: List[str]
    estimated_length_before_m: float
    estimated_length_after_m: float


def viewpoint_position(vp: Dict[str, Any]) -> Point3:
    return (float(vp["x"]), float(vp["y"]), float(vp["z"]))


def euclidean(a: Point3, b: Point3) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def path_length(start: Point3, ordered_ids: List[str], viewpoints: Dict[str, Dict[str, Any]]) -> float:
    total = 0.0
    cur = start
    for vid in ordered_ids:
        nxt = viewpoint_position(viewpoints[vid])
        total += euclidean(cur, nxt)
        cur = nxt
    return total


def nearest_neighbour_order(
    start: Point3,
    viewpoint_ids: List[str],
    viewpoints: Dict[str, Dict[str, Any]],
) -> List[str]:
    remaining = set(viewpoint_ids)
    order: List[str] = []
    cur = start

    while remaining:
        best_id = min(
            remaining,
            key=lambda vid: euclidean(cur, viewpoint_position(viewpoints[vid])),
        )
        order.append(best_id)
        cur = viewpoint_position(viewpoints[best_id])
        remaining.remove(best_id)

    return order


def two_opt_open_path(
    start: Point3,
    ordered_ids: List[str],
    viewpoints: Dict[str, Dict[str, Any]],
    max_passes: int = 20,
) -> List[str]:
    """
    2-opt improvement for an OPEN path starting at `start`.
    We do not force return-to-start.
    """
    if len(ordered_ids) < 4:
        return ordered_ids[:]

    best = ordered_ids[:]
    best_len = path_length(start, best, viewpoints)

    improved = True
    passes = 0

    while improved and passes < max_passes:
        improved = False
        passes += 1

        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                candidate = best[:i] + list(reversed(best[i:j + 1])) + best[j + 1:]
                cand_len = path_length(start, candidate, viewpoints)
                if cand_len + 1e-9 < best_len:
                    best = candidate
                    best_len = cand_len
                    improved = True

        if not improved:
            break

    return best


def build_ordered_mission(
    scenario: Dict[str, Any],
    strategy: str = "input",
    start: Point3 = (0.0, 0.0, 1.0),
) -> OrderedMission:
    viewpoints: Dict[str, Dict[str, Any]] = scenario["viewpoint_poses"]
    original_ids = list(viewpoints.keys())

    before = path_length(start, original_ids, viewpoints)

    if strategy == "input":
        ordered = original_ids[:]
    elif strategy == "nn":
        ordered = nearest_neighbour_order(start, original_ids, viewpoints)
    elif strategy == "nn_2opt":
        ordered = nearest_neighbour_order(start, original_ids, viewpoints)
        ordered = two_opt_open_path(start, ordered, viewpoints)
    else:
        raise ValueError(f"Unknown ordering strategy: {strategy}")

    after = path_length(start, ordered, viewpoints)

    return OrderedMission(
        ordered_ids=ordered,
        original_ids=original_ids,
        estimated_length_before_m=before,
        estimated_length_after_m=after,
    )
