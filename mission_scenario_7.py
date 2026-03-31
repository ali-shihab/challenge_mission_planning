#!/usr/bin/env python3
"""Single-drone mission with instrumentation, ordering, watchdogs, pose verification, A* local routing, and ArUco inspection checks."""

__authors__ = 'Rafael Perez-Segui'
__copyright__ = 'Copyright (c) 2024 Universidad Politécnica de Madrid'
__license__ = 'BSD-3-Clause'

import argparse
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from time import sleep
from typing import Any, Dict, Optional

import rclpy
import yaml
from as2_python_api.drone_interface import DroneInterface
import cv2

from mission_logging.run_logger import RunLogger
from mission_perception.aruco_tracker import ArucoTracker
from planners.ordering import build_ordered_mission
from planners.grid_map import GridConfig, build_occupancy_grid_from_scenario
from planners.astar import astar_search
from planners.path_utils import (
    cells_to_world_path,
    simplify_world_path,
    densify_stride,
    build_subgoals_from_xy_path,
)

TAKE_OFF_HEIGHT = 1.0
TAKE_OFF_SPEED = 2.0
SLEEP_TIME = 0.05
SPEED = 1.0
LAND_SPEED = 0.5


def yaw_wrap(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_error_abs(a: float, b: float) -> float:
    return abs(yaw_wrap(a - b))


def euclidean3(a: list[float], b: list[float]) -> float:
    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2 +
        (a[2] - b[2]) ** 2
    )


def read_scenario(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


_SCRIPT_DIR = Path(__file__).parent


def _find_model_sdf(model_type: str) -> Optional[str]:
    """Search GZ_SIM_RESOURCE_PATH (and known fallback dirs) for {model_type}/{model_type}.sdf."""
    gz_path = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    search_paths = [p for p in gz_path.split(':') if p] + [
        str(_SCRIPT_DIR / 'config_sim' / 'gazebo' / 'models'),
        str(_SCRIPT_DIR / 'config_sim' / 'world' / 'models'),
    ]
    for base in search_paths:
        candidate = Path(base) / model_type / f'{model_type}.sdf'
        if candidate.is_file():
            return str(candidate)
    return None


def respawn_scenario_objects(scenario_file: str, world_name: str = 'empty') -> None:
    """Delete existing scenario objects from Gazebo and respawn from the given scenario file."""
    world_config_dir = _SCRIPT_DIR / 'config_sim' / 'world'
    world_file = world_config_dir / 'world.yaml'

    # Read current world.yaml to know what to remove
    old_object_names: list = []
    if world_file.exists():
        with open(world_file) as f:
            old_world = yaml.safe_load(f) or {}
        old_object_names = [obj['model_name'] for obj in old_world.get('objects', [])]

    # Regenerate world.yaml and obstacle SDF models from the new scenario
    subprocess.run(
        [sys.executable,
         str(_SCRIPT_DIR / 'utils' / 'generate_world_from_scenario.py'),
         scenario_file,
         str(world_config_dir)],
        check=True,
    )

    # Ensure generated obstacle models are on the Gazebo resource path
    models_dir = str(world_config_dir / 'models')
    gz_path = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    if models_dir not in gz_path:
        os.environ['GZ_SIM_RESOURCE_PATH'] = gz_path + ':' + models_dir

    # Remove old objects from the running Gazebo world
    for name in old_object_names:
        subprocess.run(
            ['ign', 'service',
             '-s', f'/world/{world_name}/remove',
             '--reqtype', 'ignition.msgs.Entity',
             '--reptype', 'ignition.msgs.Boolean',
             '--timeout', '2000',
             '--req', f'name: "{name}" type: 2'],
            capture_output=True,
        )

    # Spawn new objects from the regenerated world.yaml
    with open(world_file) as f:
        new_world = yaml.safe_load(f) or {}

    for obj in new_world.get('objects', []):
        model_type = obj['model_type']
        model_name = obj['model_name']
        xyz = obj['xyz']
        rpy = obj['rpy']

        sdf_file = _find_model_sdf(model_type)
        if sdf_file is None:
            print(f'WARNING: SDF not found for model type {model_type!r}, skipping spawn')
            continue

        subprocess.run(
            ['ros2', 'run', 'ros_gz_sim', 'create',
             '-world', world_name,
             '-file', sdf_file,
             '-name', model_name,
             '-allow_renaming', 'false',
             '-x', str(xyz[0]),
             '-y', str(xyz[1]),
             '-z', str(xyz[2]),
             '-R', str(rpy[0]),
             '-P', str(rpy[1]),
             '-Y', str(rpy[2])],
            capture_output=True,
        )

    print(f'Gazebo world respawned from {scenario_file}')


def get_latest_pose(logger: RunLogger) -> Optional[Dict[str, float]]:
    if hasattr(logger, "get_latest_pose"):
        try:
            pose = logger.get_latest_pose()
            if pose is not None:
                return dict(pose)
        except Exception:
            pass

    if hasattr(logger, "_odom_lock") and hasattr(logger, "_latest_pose"):
        try:
            with logger._odom_lock:  # type: ignore[attr-defined]
                pose = logger._latest_pose  # type: ignore[attr-defined]
                return dict(pose) if pose is not None else None
        except Exception:
            pass

    return None


def wait_for_pose(logger: RunLogger, timeout_s: float = 5.0, poll_s: float = 0.1) -> Optional[Dict[str, float]]:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        pose = get_latest_pose(logger)
        if pose is not None:
            return pose
        time.sleep(poll_s)
    return None


def pose_error_to_goal(
    logger: RunLogger,
    goal_xyz: list[float],
    goal_yaw: float,
) -> Optional[Dict[str, float]]:
    pose = get_latest_pose(logger)
    if pose is None:
        return None

    pos_err = euclidean3(
        [pose["x"], pose["y"], pose["z"]],
        goal_xyz,
    )
    yaw_err = yaw_error_abs(float(pose.get("yaw", 0.0)), goal_yaw)
    return {
        "pos_err_m": pos_err,
        "yaw_err_rad": yaw_err,
        "yaw_err_deg": math.degrees(yaw_err),
        "pose": pose,
    }


def verify_pose_reached(
    logger: RunLogger,
    goal_xyz: list[float],
    goal_yaw: float,
    pos_tolerance_m: float,
    yaw_tolerance_deg: float,
) -> Dict[str, Any]:
    err = pose_error_to_goal(logger, goal_xyz, goal_yaw)
    if err is None:
        return {
            "verified": False,
            "reason": "no_pose",
        }

    verified = (
        err["pos_err_m"] <= pos_tolerance_m and
        err["yaw_err_deg"] <= yaw_tolerance_deg
    )
    return {
        "verified": verified,
        "reason": "ok" if verified else "pose_out_of_tolerance",
        **err,
        "pos_tolerance_m": pos_tolerance_m,
        "yaw_tolerance_deg": yaw_tolerance_deg,
    }


def _start_sampling_thread(node, logger: RunLogger, stop_flag: threading.Event, hz: float = 10.0) -> threading.Thread:
    period = 1.0 / max(1e-6, hz)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    def loop():
        while not stop_flag.is_set():
            try:
                executor.spin_once(timeout_sec=period)
            except Exception:
                pass
            try:
                logger.tick()
            except Exception:
                pass

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def run_blocking_action_with_watchdog(
    *,
    action_fn,
    logger: RunLogger,
    action_name: str,
    goal_xyz: list[float],
    goal_yaw: float,
    timeout_s: float,
    stuck_timeout_s: float,
    progress_epsilon_m: float,
    poll_s: float = 0.2,
    initial_grace_s: float = 2.0,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "thread_returned": False,
        "thread_success": False,
        "watchdog_reason": None,
        "elapsed_s": 0.0,
    }

    done = threading.Event()

    def target():
        try:
            ok = action_fn()
            result["thread_success"] = bool(ok)
        except Exception as e:
            result["exception_type"] = type(e).__name__
            result["exception_msg"] = str(e)
            result["thread_success"] = False
        finally:
            result["thread_returned"] = True
            done.set()

    worker = threading.Thread(target=target, daemon=True)
    worker.start()

    t0 = time.time()
    last_progress_t = t0
    last_progress_pose = get_latest_pose(logger)

    logger.event(f"{action_name}_watchdog_start", {
        "goal": goal_xyz,
        "yaw": goal_yaw,
        "timeout_s": timeout_s,
        "stuck_timeout_s": stuck_timeout_s,
        "progress_epsilon_m": progress_epsilon_m,
        "initial_grace_s": initial_grace_s,
    })

    while not done.is_set():
        now = time.time()
        elapsed = now - t0

        if elapsed > timeout_s:
            result["watchdog_reason"] = "timeout"
            logger.event(f"{action_name}_watchdog_timeout", {
                "elapsed_s": elapsed,
                "timeout_s": timeout_s,
            })
            break

        pose = get_latest_pose(logger)
        if pose is not None and last_progress_pose is not None:
            moved = euclidean3(
                [pose["x"], pose["y"], pose["z"]],
                [last_progress_pose["x"], last_progress_pose["y"], last_progress_pose["z"]],
            )
            if moved >= progress_epsilon_m:
                last_progress_pose = pose
                last_progress_t = now

        if elapsed > initial_grace_s and (now - last_progress_t) > stuck_timeout_s:
            err = pose_error_to_goal(logger, goal_xyz, goal_yaw)
            result["watchdog_reason"] = "stuck_no_progress"
            logger.event(f"{action_name}_watchdog_stuck", {
                "elapsed_s": elapsed,
                "seconds_since_progress": now - last_progress_t,
                "stuck_timeout_s": stuck_timeout_s,
                "goal_error": err,
            })
            break

        time.sleep(poll_s)

    result["elapsed_s"] = time.time() - t0

    if done.is_set():
        logger.event(f"{action_name}_watchdog_done", {
            "elapsed_s": result["elapsed_s"],
            "thread_success": result["thread_success"],
        })
    else:
        logger.event(f"{action_name}_watchdog_abort", {
            "elapsed_s": result["elapsed_s"],
            "watchdog_reason": result["watchdog_reason"],
        })

    return result


def safe_manual(drone_interface: DroneInterface, logger: RunLogger) -> bool:
    logger.event("manual_start")
    print("Manual")
    try:
        success = bool(drone_interface.manual())
    except Exception as e:
        logger.event("manual_exception", {"type": type(e).__name__, "msg": str(e)})
        print(f"Manual exception: {e}")
        return False
    logger.event("manual_done", {"success": success})
    print(f"Manual success: {success}")
    return success


def safe_disarm(drone_interface: DroneInterface, logger: RunLogger) -> bool:
    if not hasattr(drone_interface, "disarm"):
        logger.event("disarm_unavailable")
        print("Disarm unavailable on DroneInterface")
        return False

    logger.event("disarm_start")
    print("Disarm")
    try:
        success = bool(drone_interface.disarm())
    except Exception as e:
        logger.event("disarm_exception", {"type": type(e).__name__, "msg": str(e)})
        print(f"Disarm exception: {e}")
        return False
    logger.event("disarm_done", {"success": success})
    print(f"Disarm success: {success}")
    return success


def drone_start(drone_interface: DroneInterface, logger: RunLogger) -> bool:
    print("Start mission")

    logger.event("arm_start")
    print("Arm")
    success = bool(drone_interface.arm())
    logger.event("arm_done", {"success": success})
    print(f"Arm success: {success}")
    if not success:
        return False

    logger.event("offboard_start")
    print("Offboard")
    success = bool(drone_interface.offboard())
    logger.event("offboard_done", {"success": success})
    print(f"Offboard success: {success}")
    if not success:
        return False

    logger.event("takeoff_start", {"height": TAKE_OFF_HEIGHT, "speed": TAKE_OFF_SPEED})
    print("Take Off")
    success = bool(drone_interface.takeoff(height=TAKE_OFF_HEIGHT, speed=TAKE_OFF_SPEED))
    logger.event("takeoff_done", {"success": success})
    print(f"Take Off success: {success}")
    return success


def monitored_goto(
    *,
    drone_interface: DroneInterface,
    logger: RunLogger,
    action_name: str,
    goal_xyz: list[float],
    goal_yaw: float,
    speed: float,
    timeout_s: float,
    stuck_timeout_s: float,
    progress_epsilon_m: float,
) -> Dict[str, Any]:
    def action():
        return drone_interface.go_to.go_to_point_with_yaw(goal_xyz, angle=goal_yaw, speed=speed)

    return run_blocking_action_with_watchdog(
        action_fn=action,
        logger=logger,
        action_name=action_name,
        goal_xyz=goal_xyz,
        goal_yaw=goal_yaw,
        timeout_s=timeout_s,
        stuck_timeout_s=stuck_timeout_s,
        progress_epsilon_m=progress_epsilon_m,
    )


def drone_return_to_start(
    *,
    drone_interface: DroneInterface,
    start_pose: dict,
    logger: RunLogger,
    goto_timeout_s: float,
    stuck_timeout_s: float,
    progress_epsilon_m: float,
    pos_tolerance_m: float,
    yaw_tolerance_deg: float,
    dwell_s: float,
) -> bool:
    logger.event("return_to_start_start", {"start_pose": start_pose})

    goal = [float(start_pose["x"]), float(start_pose["y"]), float(start_pose["z"])]
    yaw = float(start_pose.get("yaw", 0.0))

    print(f"Returning to start pose: {goal}, yaw={yaw}")

    wd = monitored_goto(
        drone_interface=drone_interface,
        logger=logger,
        action_name="return_to_start",
        goal_xyz=goal,
        goal_yaw=yaw,
        speed=SPEED,
        timeout_s=goto_timeout_s,
        stuck_timeout_s=stuck_timeout_s,
        progress_epsilon_m=progress_epsilon_m,
    )

    if not wd["thread_returned"]:
        logger.event("return_to_start_done", {
            "success": False,
            "reason": wd["watchdog_reason"],
            "elapsed_s": wd["elapsed_s"],
        })
        print(f"Return-to-start failed: {wd['watchdog_reason']}")
        return False

    if not wd["thread_success"]:
        logger.event("return_to_start_done", {
            "success": False,
            "reason": "goto_returned_false",
            "elapsed_s": wd["elapsed_s"],
        })
        print("Return-to-start go_to returned False")
        return False

    sleep(dwell_s)
    verification = verify_pose_reached(
        logger=logger,
        goal_xyz=goal,
        goal_yaw=yaw,
        pos_tolerance_m=pos_tolerance_m,
        yaw_tolerance_deg=yaw_tolerance_deg,
    )

    logger.event("return_to_start_done", {
        "success": bool(verification["verified"]),
        "elapsed_s": wd["elapsed_s"],
        "verification": verification,
    })
    print(f"Return-to-start verified: {verification['verified']}")
    return bool(verification["verified"])


def plan_local_path(
    *,
    logger: RunLogger,
    scenario: dict,
    current_pose: Dict[str, float],
    target_vp: Dict[str, Any],
    local_planner: str,
    grid_resolution: float,
    obstacle_inflation_m: float,
    grid_margin_m: float,
    path_stride: int,
    prebuilt_grid=None,
) -> Dict[str, Any]:
    start_xyz = [float(current_pose["x"]), float(current_pose["y"]), float(current_pose["z"])]
    goal_xyz = [float(target_vp["x"]), float(target_vp["y"]), float(target_vp["z"])]
    goal_yaw = float(target_vp["w"])

    if local_planner == "straight":
        return {
            "planner": "straight",
            "subgoals": [{
                "x": goal_xyz[0],
                "y": goal_xyz[1],
                "z": goal_xyz[2],
                "yaw": goal_yaw,
            }],
            "debug": {
                "num_raw_points": 2,
                "num_simplified_points": 2,
            },
        }

    if local_planner != "astar":
        raise ValueError(f"Unknown local planner: {local_planner}")

    flight_z = max(start_xyz[2], goal_xyz[2], TAKE_OFF_HEIGHT)
    if prebuilt_grid is not None:
        grid = prebuilt_grid
    else:
        grid = build_occupancy_grid_from_scenario(
            scenario,
            start_xy=(start_xyz[0], start_xyz[1]),
            flight_z=flight_z,
            config=GridConfig(
                resolution_m=grid_resolution,
                inflation_m=obstacle_inflation_m,
                bounds_margin_m=grid_margin_m,
            ),
        )

    start_cell = grid.world_to_grid(start_xyz[0], start_xyz[1])
    goal_cell = grid.world_to_grid(goal_xyz[0], goal_xyz[1])

    cells = astar_search(grid, start_cell, goal_cell)
    if cells is None:
        return {
            "planner": "astar",
            "subgoals": [],
            "debug": {
                "failure": "no_path_found",
                "start_cell": start_cell,
                "goal_cell": goal_cell,
            },
        }

    raw_xy = cells_to_world_path(grid, cells)
    simplified_xy = simplify_world_path(grid, raw_xy)
    simplified_xy = densify_stride(simplified_xy, stride=path_stride)

    if not simplified_xy:
        simplified_xy = [(goal_xyz[0], goal_xyz[1])]

    if simplified_xy[-1] != (goal_xyz[0], goal_xyz[1]):
        simplified_xy[-1] = (goal_xyz[0], goal_xyz[1])

    subgoals = build_subgoals_from_xy_path(
        path_xy=simplified_xy,
        start_z=start_xyz[2],
        goal_z=goal_xyz[2],
        final_yaw=goal_yaw,
    )

    return {
        "planner": "astar",
        "subgoals": subgoals,
        "debug": {
            "num_raw_points": len(raw_xy),
            "num_simplified_points": len(simplified_xy),
            "start_cell": start_cell,
            "goal_cell": goal_cell,
            "grid_width": grid.width,
            "grid_height": grid.height,
            "num_occupied": len(grid.occupied),
            "grid_resolution": grid_resolution,
            "obstacle_inflation_m": obstacle_inflation_m,
        },
    }


def expected_marker_id_for_viewpoint(
    viewpoint_id: str,
    *,
    explicit_offset: int,
) -> int:
    # Markers are assigned by generate_world_from_scenario.py as aruco_id{(key%8)+1}4_marker,
    # which are DICT_5X5_250 markers with IDs 14,24,34,...,84 cycling over viewpoints.
    # Formula: ((viewpoint_id % 8) + 1) * 10 + 4
    try:
        return ((int(viewpoint_id) % 8) + 1) * 10 + 4 + explicit_offset
    except Exception:
        raise ValueError(f"Invalid viewpoint ID: {viewpoint_id}, type: {type(viewpoint_id)}")


def execute_subgoals(
    *,
    drone_interface: DroneInterface,
    logger: RunLogger,
    aruco_tracker: Optional[ArucoTracker],
    viewpoint_index: int,
    viewpoint_id: str,
    expected_marker_id: Optional[int],
    subgoals: list[Dict[str, float]],
    goto_timeout_s: float,
    stuck_timeout_s: float,
    progress_epsilon_m: float,
    subgoal_pos_tolerance_m: float,
    final_pos_tolerance_m: float,
    yaw_tolerance_deg: float,
    dwell_s: float,
    require_pose_verified: bool,
    require_aruco_verified: bool,
    inspection_timeout_s: float,
    inspection_recent_window_s: float,
    inspection_min_count: int,
) -> bool:
    final_verification: Optional[Dict[str, Any]] = None

    for sub_idx, sg in enumerate(subgoals):
        is_final = sub_idx == len(subgoals) - 1
        goal = [float(sg["x"]), float(sg["y"]), float(sg["z"])]
        yaw = float(sg["yaw"])

        logger.event("subgoal_start", {
            "viewpoint_index": viewpoint_index,
            "viewpoint_id": viewpoint_id,
            "subgoal_index": sub_idx,
            "is_final": is_final,
            "goal": goal,
            "yaw": yaw,
            "speed": SPEED,
        })

        wd = monitored_goto(
            drone_interface=drone_interface,
            logger=logger,
            action_name=f"vp{viewpoint_index}_sg{sub_idx}",
            goal_xyz=goal,
            goal_yaw=yaw,
            speed=SPEED,
            timeout_s=goto_timeout_s,
            stuck_timeout_s=stuck_timeout_s,
            progress_epsilon_m=progress_epsilon_m,
        )

        if not wd["thread_returned"]:
            logger.event("subgoal_done", {
                "viewpoint_index": viewpoint_index,
                "viewpoint_id": viewpoint_id,
                "subgoal_index": sub_idx,
                "success": False,
                "reason": wd["watchdog_reason"],
                "duration_s": wd["elapsed_s"],
            })
            return False

        logger.event("subgoal_done", {
            "viewpoint_index": viewpoint_index,
            "viewpoint_id": viewpoint_id,
            "subgoal_index": sub_idx,
            "success": bool(wd["thread_success"]),
            "duration_s": wd["elapsed_s"],
        })

        if not wd["thread_success"]:
            return False

        sleep(dwell_s if is_final else 0.1)

        verification = verify_pose_reached(
            logger=logger,
            goal_xyz=goal,
            goal_yaw=yaw,
            pos_tolerance_m=(final_pos_tolerance_m if is_final else subgoal_pos_tolerance_m),
            yaw_tolerance_deg=(yaw_tolerance_deg if is_final else 180.0),
        )
        verification["viewpoint_index"] = viewpoint_index
        verification["viewpoint_id"] = viewpoint_id
        verification["subgoal_index"] = sub_idx
        verification["is_final"] = is_final

        logger.event("subgoal_verification", verification)

        if require_pose_verified and not verification["verified"]:
            return False

        if is_final:
            final_verification = dict(verification)

    if final_verification is None:
        return False

    aruco_result = {
        "verified": False,
        "reason": "aruco_not_requested",
        "expected_marker_id": expected_marker_id,
    }

    if expected_marker_id is not None and aruco_tracker is not None:
        logger.event("aruco_wait_start", {
            "viewpoint_index": viewpoint_index,
            "viewpoint_id": viewpoint_id,
            "expected_marker_id": expected_marker_id,
            "inspection_timeout_s": inspection_timeout_s,
            "inspection_recent_window_s": inspection_recent_window_s,
            "inspection_min_count": inspection_min_count,
        })

        aruco_result = aruco_tracker.wait_for_marker(
            expected_marker_id,
            timeout_s=inspection_timeout_s,
            recent_window_s=inspection_recent_window_s,
            min_count=inspection_min_count,
        )

        logger.event("aruco_wait_done", {
            "viewpoint_index": viewpoint_index,
            "viewpoint_id": viewpoint_id,
            **aruco_result,
        })

        if aruco_result["verified"]:
            aruco_result["reason"] = "marker_seen"
        else:
            aruco_result["reason"] = "marker_not_seen_in_time"

    elif expected_marker_id is None:
        aruco_result = {
            "verified": False,
            "reason": "no_expected_marker_id",
            "expected_marker_id": None,
        }
    elif aruco_tracker is None:
        aruco_result = {
            "verified": False,
            "reason": "aruco_tracker_unavailable",
            "expected_marker_id": expected_marker_id,
        }

    viewpoint_verification = {
        **final_verification,
        "aruco_verified": bool(aruco_result["verified"]),
        "aruco_reason": aruco_result["reason"],
        "expected_marker_id": aruco_result.get("expected_marker_id"),
        "aruco_recent_ids": aruco_result.get("recent_ids"),
        "aruco_count_in_window": aruco_result.get("count_in_window"),
    }
    logger.event("viewpoint_verification", viewpoint_verification)

    if require_aruco_verified and not bool(aruco_result["verified"]):
        return False

    return True


def drone_run(
    *,
    drone_interface: DroneInterface,
    scenario: dict,
    ordered_ids: list[str],
    logger: RunLogger,
    aruco_tracker: Optional[ArucoTracker],
    marker_id_offset: int,
    local_planner: str,
    grid_resolution: float,
    obstacle_inflation_m: float,
    grid_margin_m: float,
    path_stride: int,
    goto_timeout_s: float,
    stuck_timeout_s: float,
    progress_epsilon_m: float,
    subgoal_pos_tolerance_m: float,
    final_pos_tolerance_m: float,
    yaw_tolerance_deg: float,
    dwell_s: float,
    require_pose_verified: bool,
    require_aruco_verified: bool,
    inspection_timeout_s: float,
    inspection_recent_window_s: float,
    inspection_min_count: int,
) -> bool:
    print("Run mission")

    viewpoints = scenario["viewpoint_poses"]
    logger.event("mission_waypoints_loaded", {
        "num_waypoints": len(ordered_ids),
        "ordered_ids": ordered_ids,
        "local_planner": local_planner,
        "require_aruco_verified": require_aruco_verified,
    })

    verified_count = 0


    # Build occupancy grid once for the whole mission (reused per leg)
    _global_grid = None
    if local_planner == "astar":
        from planners.grid_map import GridConfig, build_occupancy_grid_from_scenario as _build_grid
        import math as _math
        _all_vp = list(viewpoints.values())
        _max_z = max((float(v["z"]) for v in _all_vp), default=TAKE_OFF_HEIGHT)
        _flight_z_global = max(_max_z, TAKE_OFF_HEIGHT)
        logger.event("global_grid_build_start", {"flight_z": _flight_z_global})
        _global_grid = _build_grid(
            scenario,
            start_xy=(0.0, 0.0),
            flight_z=_flight_z_global,
            config=GridConfig(
                resolution_m=grid_resolution,
                inflation_m=obstacle_inflation_m,
                bounds_margin_m=grid_margin_m + 2.0,
            ),
        )
        logger.event("global_grid_build_done", {
            "width": _global_grid.width,
            "height": _global_grid.height,
            "num_occupied": len(_global_grid.occupied),
        })

    for idx, vpid in enumerate(ordered_ids):
        vp = viewpoints[vpid]
        current_pose = get_latest_pose(logger)
        if current_pose is None:
            logger.event("planning_failed", {
                "index": idx,
                "viewpoint_id": vpid,
                "reason": "no_current_pose",
            })
            return False

        expected_marker_id = expected_marker_id_for_viewpoint(
            vpid,
            explicit_offset=marker_id_offset,
        )

        logger.event("local_plan_start", {
            "index": idx,
            "viewpoint_id": vpid,
            "expected_marker_id": expected_marker_id,
            "from_pose": current_pose,
            "target": vp,
            "local_planner": local_planner,
        })

        local_plan = plan_local_path(
            logger=logger,
            scenario=scenario,
            current_pose=current_pose,
            target_vp=vp,
            local_planner=local_planner,
            grid_resolution=grid_resolution,
            obstacle_inflation_m=obstacle_inflation_m,
            grid_margin_m=grid_margin_m,
            path_stride=path_stride,
            prebuilt_grid=_global_grid,
        )

        logger.event("local_plan_done", {
            "index": idx,
            "viewpoint_id": vpid,
            "planner": local_plan["planner"],
            "num_subgoals": len(local_plan["subgoals"]),
            "debug": local_plan["debug"],
        })

        if not local_plan["subgoals"]:
            return False

        print(f"Go to {vpid} using {local_planner} with {len(local_plan['subgoals'])} subgoals")

        ok = execute_subgoals(
            drone_interface=drone_interface,
            logger=logger,
            aruco_tracker=aruco_tracker,
            viewpoint_index=idx,
            viewpoint_id=vpid,
            expected_marker_id=expected_marker_id,
            subgoals=local_plan["subgoals"],
            goto_timeout_s=goto_timeout_s,
            stuck_timeout_s=stuck_timeout_s,
            progress_epsilon_m=progress_epsilon_m,
            subgoal_pos_tolerance_m=subgoal_pos_tolerance_m,
            final_pos_tolerance_m=final_pos_tolerance_m,
            yaw_tolerance_deg=yaw_tolerance_deg,
            dwell_s=dwell_s,
            require_pose_verified=require_pose_verified,
            require_aruco_verified=require_aruco_verified,
            inspection_timeout_s=inspection_timeout_s,
            inspection_recent_window_s=inspection_recent_window_s,
            inspection_min_count=inspection_min_count,
        )
        if not ok:
            return False

        verified_count += 1
        print("Viewpoint done")
        sleep(SLEEP_TIME)

    logger.event("mission_verification_summary", {
        "verified_count": verified_count,
        "total_waypoints": len(ordered_ids),
        "require_pose_verified": require_pose_verified,
        "require_aruco_verified": require_aruco_verified,
    })
    return True


def drone_end(
    drone_interface: DroneInterface,
    logger: RunLogger,
    *,
    land_timeout_s: float = 30.0,
) -> bool:
    print("End mission")

    logger.event("land_start", {"speed": LAND_SPEED})
    print("Land")
    land_success = False

    def land_action():
        return drone_interface.land(speed=LAND_SPEED)

    land_wd = run_blocking_action_with_watchdog(
        action_fn=land_action,
        logger=logger,
        action_name="land",
        goal_xyz=[0.0, 0.0, 0.0],
        goal_yaw=0.0,
        timeout_s=land_timeout_s,
        stuck_timeout_s=10.0,
        progress_epsilon_m=0.05,
        initial_grace_s=1.0,
    )

    if land_wd["thread_returned"]:
        land_success = bool(land_wd["thread_success"])

    logger.event("land_done", {
        "success": land_success,
        "duration_s": land_wd["elapsed_s"],
        "reason": land_wd["watchdog_reason"],
    })
    print(f"Land success: {land_success}")

    manual_success = safe_manual(drone_interface, logger)
    disarm_success = safe_disarm(drone_interface, logger)

    return land_success and manual_success and disarm_success


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single drone mission')

    parser.add_argument('scenario', type=str, help='Scenario file to attempt to execute')
    parser.add_argument('-n', '--namespace', type=str, default='drone0',
                        help='ID of the drone to be used in the mission')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Enable verbose output')
    parser.add_argument('-s', '--use_sim_time', action='store_true', default=True,
                        help='Use simulation time')

    parser.add_argument('--planner', type=str, default='baseline',
                        help='Planner label to store in logs')
    parser.add_argument('--runs_root', type=str, default='runs',
                        help='Root directory for run artifacts')
    parser.add_argument('--odom_topic', type=str, default=None,
                        help='Telemetry topic override. Recommended: /<namespace>/self_localization/pose')
    parser.add_argument('--sample_hz', type=float, default=10.0,
                        help='Trajectory sample rate (Hz)')

    parser.add_argument('--speed', type=float, default=None,
                        help='Override flight speed in m/s (overrides SPEED constant)')

    parser.add_argument('--ordering', type=str, default='input',
                        choices=['input', 'nn', 'nn_2opt'],
                        help='Viewpoint ordering strategy')
    parser.add_argument('--start_x', type=float, default=0.0,
                        help='Fallback start x for ordering if live pose is unavailable')
    parser.add_argument('--start_y', type=float, default=0.0,
                        help='Fallback start y for ordering if live pose is unavailable')
    parser.add_argument('--start_z', type=float, default=TAKE_OFF_HEIGHT,
                        help='Fallback start z for ordering if live pose is unavailable')

    parser.add_argument('--local_planner', type=str, default='straight',
                        choices=['straight', 'astar'],
                        help='Local planner between viewpoints')
    parser.add_argument('--grid_resolution', type=float, default=0.5,
                        help='XY occupancy grid resolution in meters')
    parser.add_argument('--obstacle_inflation_m', type=float, default=0.8,
                        help='Obstacle inflation radius in meters')
    parser.add_argument('--grid_margin_m', type=float, default=2.0,
                        help='Additional grid bounds margin in meters')
    parser.add_argument('--path_stride', type=int, default=1,
                        help='Keep every Nth simplified path point (>=1)')

    parser.add_argument('--goto_timeout_s', type=float, default=45.0,
                        help='Hard timeout per go-to command')
    parser.add_argument('--stuck_timeout_s', type=float, default=6.0,
                        help='Fail if no meaningful progress for this long')
    parser.add_argument('--progress_epsilon_m', type=float, default=0.15,
                        help='Movement threshold counted as progress')
    parser.add_argument('--subgoal_pos_tolerance_m', type=float, default=1.0,
                        help='Position tolerance for intermediate subgoal verification')
    parser.add_argument('--final_pos_tolerance_m', type=float, default=0.75,
                        help='Position tolerance for final viewpoint verification')
    parser.add_argument('--yaw_tolerance_deg', type=float, default=30.0,
                        help='Yaw tolerance for final viewpoint verification')
    parser.add_argument('--dwell_s', type=float, default=0.5,
                        help='Hover time after each final viewpoint before verification')
    parser.add_argument('--require_pose_verified', action='store_true', default=False,
                        help='Fail mission if post-goal pose verification fails')

    # ArUco inspection
    parser.add_argument('--aruco_topic', type=str, default=None,
                        help='Marker detection topic to subscribe to')
    parser.add_argument('--require_aruco_verified', action='store_true', default=False,
                        help='Fail mission if the expected marker was not detected at each viewpoint')
    parser.add_argument('--marker_id_offset', type=int, default=0,
                        help='Offset added to viewpoint ID when deriving expected marker ID')
    parser.add_argument('--inspection_timeout_s', type=float, default=5.0,
                        help='How long to wait for marker detection after reaching a viewpoint')
    parser.add_argument('--inspection_recent_window_s', type=float, default=2.0,
                        help='Detection recency window used for marker verification')
    parser.add_argument('--inspection_min_count', type=int, default=1,
                        help='Minimum detections required in the recent window')

    parser.add_argument('--return_to_start', action='store_true', default=False,
                        help='Return to the captured start hover pose on successful missions')
    parser.add_argument('--return_to_start_on_failure', action='store_true', default=False,
                        help='Also attempt return-to-start on failed/interrupted runs')

    parser.add_argument('--no_respawn_world', action='store_true', default=False,
                        help='Skip Gazebo world respawn (use when not running in simulation)')

    args = parser.parse_args()

    drone_namespace = args.namespace
    verbosity = args.verbose
    use_sim_time = args.use_sim_time
    global SPEED
    if args.speed is not None:
        SPEED = args.speed

    print(f'Running mission for drone {drone_namespace}')
    print(f'Reading scenario {args.scenario}')
    scenario = read_scenario(args.scenario)

    if not args.no_respawn_world:
        print('Respawning Gazebo world objects for scenario...')
        respawn_scenario_objects(args.scenario)

    rclpy.init()

    logger = RunLogger(
        scenario_path=args.scenario,
        drone_namespace=drone_namespace,
        planner=args.planner,
        use_sim_time=use_sim_time,
        verbose=verbosity,
        runs_root=args.runs_root,
        odom_topic=args.odom_topic,
        sample_hz=args.sample_hz,
        repo_root=".",
        extra_meta={
            "script": "mission_scenario.py",
            "ordering": args.ordering,
            "local_planner": args.local_planner,
            "fallback_ordering_start": {
                "x": args.start_x,
                "y": args.start_y,
                "z": args.start_z,
            },
            "grid_resolution": args.grid_resolution,
            "obstacle_inflation_m": args.obstacle_inflation_m,
            "grid_margin_m": args.grid_margin_m,
            "path_stride": args.path_stride,
            "goto_timeout_s": args.goto_timeout_s,
            "stuck_timeout_s": args.stuck_timeout_s,
            "progress_epsilon_m": args.progress_epsilon_m,
            "subgoal_pos_tolerance_m": args.subgoal_pos_tolerance_m,
            "final_pos_tolerance_m": args.final_pos_tolerance_m,
            "yaw_tolerance_deg": args.yaw_tolerance_deg,
            "dwell_s": args.dwell_s,
            "require_pose_verified": args.require_pose_verified,
            "aruco_topic": args.aruco_topic,
            "require_aruco_verified": args.require_aruco_verified,
            "marker_id_offset": args.marker_id_offset,
            "inspection_timeout_s": args.inspection_timeout_s,
            "inspection_recent_window_s": args.inspection_recent_window_s,
            "inspection_min_count": args.inspection_min_count,
        },
    )

    uav = DroneInterface(
        drone_id=drone_namespace,
        use_sim_time=use_sim_time,
        verbose=verbosity,
    )

    logger_node = rclpy.create_node("run_logger", namespace=drone_namespace)
    logger.attach_ros(logger_node)

    aruco_tracker: Optional[ArucoTracker] = None
    if args.aruco_topic:
        tmp_tracker = ArucoTracker(image_topic=args.aruco_topic, dictionary=cv2.aruco.DICT_5X5_250)
        attached = tmp_tracker.attach_ros(logger_node)
        if attached is False:
            raise RuntimeError("Failed to attach ArucoTracker to ROS node")
        aruco_tracker = tmp_tracker

    stop_flag = threading.Event()
    sampler_thread = _start_sampling_thread(logger_node, logger, stop_flag, hz=args.sample_hz)

    overall_success = False
    ended_reason = "normal"
    mission_success = False
    start_pose = None

    try:
        logger.event("scenario_loaded", {"keys": list(scenario.keys())})

        success = drone_start(uav, logger)

        if success:
            sleep(0.5)

            start_pose = wait_for_pose(logger, timeout_s=5.0)
            if start_pose is None:
                start_pose = {
                    "x": args.start_x,
                    "y": args.start_y,
                    "z": args.start_z,
                    "yaw": 0.0,
                }
                logger.event("start_pose_fallback_used", {"start_pose": start_pose})
            else:
                logger.event("start_pose_captured", {"start_pose": start_pose})

            ordering_start = (
                float(start_pose["x"]),
                float(start_pose["y"]),
                float(start_pose["z"]),
            )
            ordered_mission = build_ordered_mission(
                scenario,
                strategy=args.ordering,
                start=ordering_start,
            )

            logger.event("ordering_computed", {
                "strategy": args.ordering,
                "ordering_start": {
                    "x": ordering_start[0],
                    "y": ordering_start[1],
                    "z": ordering_start[2],
                },
                "original_ids": ordered_mission.original_ids,
                "ordered_ids": ordered_mission.ordered_ids,
                "estimated_length_before_m": ordered_mission.estimated_length_before_m,
                "estimated_length_after_m": ordered_mission.estimated_length_after_m,
                "estimated_improvement_m": (
                    ordered_mission.estimated_length_before_m -
                    ordered_mission.estimated_length_after_m
                ),
            })

            start_time = time.time()
            mission_success = drone_run(
                drone_interface=uav,
                scenario=scenario,
                ordered_ids=ordered_mission.ordered_ids,
                logger=logger,
                aruco_tracker=aruco_tracker,
                marker_id_offset=args.marker_id_offset,
                local_planner=args.local_planner,
                grid_resolution=args.grid_resolution,
                obstacle_inflation_m=args.obstacle_inflation_m,
                grid_margin_m=args.grid_margin_m,
                path_stride=args.path_stride,
                goto_timeout_s=args.goto_timeout_s,
                stuck_timeout_s=args.stuck_timeout_s,
                progress_epsilon_m=args.progress_epsilon_m,
                subgoal_pos_tolerance_m=args.subgoal_pos_tolerance_m,
                final_pos_tolerance_m=args.final_pos_tolerance_m,
                yaw_tolerance_deg=args.yaw_tolerance_deg,
                dwell_s=args.dwell_s,
                require_pose_verified=args.require_pose_verified,
                require_aruco_verified=args.require_aruco_verified,
                inspection_timeout_s=args.inspection_timeout_s,
                inspection_recent_window_s=args.inspection_recent_window_s,
                inspection_min_count=args.inspection_min_count,
            )
            duration = time.time() - start_time

            print("---------------------------------")
            print(f"Tour of {args.scenario} took {duration} seconds")
            print("---------------------------------")
            logger.event("mission_duration", {
                "duration_s": duration,
                "success": mission_success,
            })

            overall_success = mission_success

    except KeyboardInterrupt:
        ended_reason = "keyboard_interrupt"
        logger.event("keyboard_interrupt")
    except Exception as e:
        ended_reason = "exception"
        logger.event("exception", {"type": type(e).__name__, "msg": str(e)})
        raise
    finally:
        should_return_to_start = (
            start_pose is not None and
            args.return_to_start and
            (
                mission_success or
                (args.return_to_start_on_failure and ended_reason in {"normal", "keyboard_interrupt", "exception"})
            )
        )

        if should_return_to_start:
            try:
                _ = drone_return_to_start(
                    drone_interface=uav,
                    start_pose=start_pose,
                    logger=logger,
                    goto_timeout_s=args.goto_timeout_s,
                    stuck_timeout_s=args.stuck_timeout_s,
                    progress_epsilon_m=args.progress_epsilon_m,
                    pos_tolerance_m=args.final_pos_tolerance_m,
                    yaw_tolerance_deg=args.yaw_tolerance_deg,
                    dwell_s=args.dwell_s,
                )
                sleep(1.0)
            except Exception as e:
                logger.event("return_to_start_failed", {"type": type(e).__name__, "msg": str(e)})

        try:
            _ = drone_end(uav, logger)
        except Exception as e:
            logger.event("end_failed", {"type": type(e).__name__, "msg": str(e)})

        stop_flag.set()
        try:
            sampler_thread.join(timeout=1.0)
        except Exception:
            pass

        try:
            uav.shutdown()
        except Exception:
            pass

        try:
            logger_node.destroy_node()
        except Exception:
            pass

        try:
            rclpy.shutdown()
        except Exception:
            pass

        logger.finalize(success=bool(overall_success), ended_reason=ended_reason)

    print('Clean exit')
    exit(0)
