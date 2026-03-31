#!/usr/bin/env python3
"""Simple mission for a single drone (instrumented)."""

__authors__ = 'Rafael Perez-Segui'
__copyright__ = 'Copyright (c) 2024 Universidad Politécnica de Madrid'
__license__ = 'BSD-3-Clause'

import argparse
from time import sleep
import time
import yaml
import threading

from as2_python_api.drone_interface import DroneInterface
import rclpy

from mission_logging.run_logger import RunLogger
from planners.ordering import build_ordered_mission

TAKE_OFF_HEIGHT = 1.0  # Height in meters
TAKE_OFF_SPEED = 1.0  # Max speed in m/s
SLEEP_TIME = 0.5  # Sleep time between behaviors in seconds
SPEED = 1.0  # Max speed in m/s
LAND_SPEED = 0.5  # Max speed in m/s


def drone_start(drone_interface: DroneInterface, logger: RunLogger) -> bool:
    print('Start mission')

    logger.event("arm_start")
    print('Arm')
    success = drone_interface.arm()
    logger.event("arm_done", {"success": success})
    print(f'Arm success: {success}')

    logger.event("offboard_start")
    print('Offboard')
    success = drone_interface.offboard()
    logger.event("offboard_done", {"success": success})
    print(f'Offboard success: {success}')

    logger.event("takeoff_start", {"height": TAKE_OFF_HEIGHT, "speed": TAKE_OFF_SPEED})
    print('Take Off')
    success = drone_interface.takeoff(height=TAKE_OFF_HEIGHT, speed=TAKE_OFF_SPEED)
    logger.event("takeoff_done", {"success": success})
    print(f'Take Off success: {success}')

    return success

def drone_return_to_start(
    drone_interface: DroneInterface,
    start_pose: dict,
    logger: RunLogger,
) -> bool:
    logger.event("return_to_start_start", {"start_pose": start_pose})

    goal = [start_pose["x"], start_pose["y"], start_pose["z"]]
    yaw = start_pose.get("yaw", 0.0)

    print(f"Returning to start pose: {goal}, yaw={yaw}")
    t0 = time.time()
    success = drone_interface.go_to.go_to_point_with_yaw(goal, angle=yaw, speed=SPEED)
    dt = time.time() - t0

    logger.event("return_to_start_done", {
        "success": success,
        "duration_s": dt,
        "goal": goal,
        "yaw": yaw,
    })
    print(f"Return-to-start success: {success}")
    return success

def drone_run(
    drone_interface: DroneInterface,
    scenario: dict,
    ordered_ids: list[str],
    logger: RunLogger,
) -> bool:
    print('Run mission')

    viewpoints = scenario["viewpoint_poses"]
    logger.event("mission_waypoints_loaded", {
        "num_waypoints": len(ordered_ids),
        "ordered_ids": ordered_ids,
    })

    for vpid in ordered_ids:
        vp = viewpoints[vpid]
        goal = [vp["x"], vp["y"], vp["z"]]
        yaw = vp["w"]

        logger.event("goto_start", {
            "viewpoint_id": vpid,
            "goal": goal,
            "yaw": yaw,
            "speed": SPEED,
        })
        print(f'Go to {vpid} with path facing {vp}')
        t0 = time.time()
        success = drone_interface.go_to.go_to_point_with_yaw(goal, angle=yaw, speed=SPEED)
        dt = time.time() - t0
        logger.event("goto_done", {
            "viewpoint_id": vpid,
            "success": success,
            "duration_s": dt,
        })
        print(f'Go to success: {success}')

        if not success:
            return False

        print('Go to done')
        sleep(SLEEP_TIME)

    return True


def drone_end(drone_interface: DroneInterface, logger: RunLogger) -> bool:
    print('End mission')

    logger.event("land_start", {"speed": LAND_SPEED})
    print('Land')
    success = drone_interface.land(speed=LAND_SPEED)
    logger.event("land_done", {"success": success})
    print(f'Land success: {success}')
    if not success:
        return False

    logger.event("manual_start")
    print('Manual')
    success = drone_interface.manual()
    logger.event("manual_done", {"success": success})
    print(f'Manual success: {success}')

    return success


def read_scenario(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        scenario = yaml.safe_load(file)
    return scenario


def _start_sampling_thread(logger: RunLogger, stop_flag: threading.Event, hz: float = 10.0) -> threading.Thread:
    period = 1.0 / max(1e-6, hz)

    def loop():
        while not stop_flag.is_set():
            # let rclpy service subscriptions
            rclpy.spin_once(logger._node, timeout_sec=0.0) if getattr(logger, "_node", None) else None
            logger.tick()
            time.sleep(period)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single drone mission')

    parser.add_argument('scenario', type=str, help="scenario file to attempt to execute")
    parser.add_argument('-n', '--namespace', type=str, default='drone0', help='ID of the drone to be used in the mission')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Enable verbose output')
    parser.add_argument('-s', '--use_sim_time', action='store_true', default=True, help='Use simulation time')

    # instrumentation args
    parser.add_argument('--planner', type=str, default='baseline', help='Planner label to store in logs')
    parser.add_argument('--runs_root', type=str, default='runs', help='Root directory for run artifacts')
    parser.add_argument('--odom_topic', type=str, default=None,
                        help='Override odom topic. Default: /<namespace>/self_localization/odom')
    parser.add_argument('--sample_hz', type=float, default=10.0, help='Trajectory sample rate (Hz)')
    parser.add_argument(
        '--ordering',
        type=str,
        default='input',
        choices=['input', 'nn', 'nn_2opt'],
        help='Viewpoint ordering strategy'
    )
    parser.add_argument(
        '--start_x',
        type=float,
        default=0.0,
        help='Assumed start x for ordering-cost estimation'
    )
    parser.add_argument(
        '--start_y',
        type=float,
        default=0.0,
        help='Assumed start y for ordering-cost estimation'
    )
    parser.add_argument(
        '--start_z',
        type=float,
        default=1.0,
        help='Assumed start z for ordering-cost estimation'
    )

    args = parser.parse_args()

    drone_namespace = args.namespace
    verbosity = args.verbose
    use_sim_time = args.use_sim_time

    print(f'Running mission for drone {drone_namespace}')
    print(f"Reading scenario {args.scenario}")
    scenario = read_scenario(args.scenario)

    ordering_start = (args.start_x, args.start_y, args.start_z)
    ordered_mission = build_ordered_mission(
        scenario,
        strategy=args.ordering,
        start=ordering_start,
    )

    rclpy.init()

    # Create logger early to capture stdout from startup
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
            "ordering_start": {
                "x": args.start_x,
                "y": args.start_y,
                "z": args.start_z,
            },
            "estimated_length_before_m": ordered_mission.estimated_length_before_m,
            "estimated_length_after_m": ordered_mission.estimated_length_after_m,
        },
    )

    logger.event("ordering_computed", {
        "strategy": args.ordering,
        "original_ids": ordered_mission.original_ids,
        "ordered_ids": ordered_mission.ordered_ids,
        "estimated_length_before_m": ordered_mission.estimated_length_before_m,
        "estimated_length_after_m": ordered_mission.estimated_length_after_m,
        "estimated_improvement_m": (
             ordered_mission.estimated_length_before_m -
             ordered_mission.estimated_length_after_m
        ),
    })

    uav = DroneInterface(
        drone_id=drone_namespace,
        use_sim_time=use_sim_time,
        verbose=verbosity)

    # Attach telemetry subscriber using the DroneInterface's internal node if available;
    # otherwise create our own node.
    node = getattr(uav, "node", None)
    if node is None:
        node = rclpy.create_node(f"{drone_namespace}_mission_logger")
    logger.attach_ros(node)

    stop_flag = threading.Event()
    sampler_thread = _start_sampling_thread(logger, stop_flag, hz=args.sample_hz)

    overall_success = False
    ended_reason = "normal"

    try:
        logger.event("scenario_loaded", {"keys": list(scenario.keys())})

        success = drone_start(uav, logger)

        if success:
            sleep(2.0)  # let pose settle after takeoff
            start_pose = logger.wait_for_first_pose(timeout_s=5.0)
            logger.event("start_pose_captured", {"start_pose": start_pose})

            if start_pose is None:
                start_pose = {"x": 0.0, "y": 0.0, "z": TAKE_OFF_HEIGHT, "yaw": 0.0}
                logger.event("start_pose_fallback_used", {"start_pose": start_pose})

        start_time = time.time()
        if success:
            overall_success = drone_run(uav, scenario, ordered_mission.ordered_ids, logger)
        duration = time.time() - start_time

        print("---------------------------------")
        print(f"Tour of {args.scenario} took {duration} seconds")
        print("---------------------------------")
        logger.event("mission_duration", {"duration_s": duration, "success": overall_success})

    except KeyboardInterrupt:
        ended_reason = "keyboard_interrupt"
        logger.event("keyboard_interrupt")
    except Exception as e:
        ended_reason = "exception"
        logger.event("exception", {"type": type(e).__name__, "msg": str(e)})
        raise
    finally:
        stop_flag.set()
        try:
            sampler_thread.join(timeout=1.0)
        except Exception:
            pass

        try:
            if start_pose is not None:
                _ = drone_return_to_start(uav, start_pose, logger)
                sleep(1.0)
        except Exception as e:
            logger.event("return_to_start_failed", {"type": type(e).__name__, "msg": str(e)})

        try:
            _ = drone_end(uav, logger)
        except Exception as e:
            logger.event("end_failed", {"type": type(e).__name__, "msg": str(e)})

        try:
            uav.shutdown()
        except Exception:
            pass

        try:
            rclpy.shutdown()
        except Exception:
            pass

        # If we never set overall_success but takeoff failed, treat as failure.
        logger.finalize(success=bool(overall_success), ended_reason=ended_reason)

    print('Clean exit')
    exit(0)
