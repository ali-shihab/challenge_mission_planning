#!/bin/python3

from time import sleep
import rclpy
from python_interface.drone_interface import DroneInterface


def drone_run(drone_interface: DroneInterface):

    speed = 1.0
    takeoff_height = 2.0
    height = 3.0

    sleep_time = 1.0

    dim = 5.0
    path = [
        [0.0, 0.0, takeoff_height],
        [dim, dim, height],
        [dim, -dim, height],
        [-dim, dim, height],
        [-dim, -dim, height],
        [0.0, 0.0, takeoff_height],
    ]

    ignore_yaw_ = False

    print("Start mission")

    drone_interface.offboard()
    drone_interface.arm()

    print("Take Off")
    drone_interface.takeoff(takeoff_height, speed=1.0)
    print("Take Off done")
    sleep(sleep_time)

    # ##### GOTO #####
    for goal in path:
        print(f"Go to {goal}")
        drone_interface.go_to_point(goal, speed=speed, ignore_yaw=ignore_yaw_)
        print("Go to done")
        sleep(sleep_time)

    ##### FOLLOW PATH #####
    # sleep(1.0)
    # print(f"Follow path: [{goal0}, {goal1}, {goal2}]")
    # path = [goal0, goal1, goal2]
    # drone_interface.follow_path(path, speed=speed)
    # print(f"Follow path done")

    sleep(1.0)
    print("Landing")
    drone_interface.land(speed=0.5)
    print("Land done")

    print("Clean exit")


if __name__ == '__main__':
    rclpy.init()
    uav = DroneInterface("drone_sim_0", verbose=False,
                         use_gps=False, use_sim_time=True)

    drone_run(uav)

    uav.shutdown()
    rclpy.shutdown()
    exit(0)
