#!/bin/python3

import rclpy
from time import sleep
from python_interface.drone_interface import DroneInterface


def drone_run(drone_interface, n_uav):

    speed = 1.0
    takeoff_height = 2.0
    height = 3.0

    size = 5.0
    goal0 = [size, 0.0, height]
    goal1 = [size, size, height]
    goal2 = [0.0, 0.0, height]

    ignore_yaw_ = False

    print(f"Start mission")

    drone_interface.offboard()
    drone_interface.arm()

    # drone_interface.send_motion_reference_pose(goal0)

    print(f"Take Off")
    drone_interface.takeoff(takeoff_height, speed=1.0)
    print(f"Take Off done")

    # ##### GOTO #####
    sleep(1.0)
    print(f"Go to: {goal0}")
    drone_interface.go_to_point(goal0, speed=speed, ignore_yaw=ignore_yaw_)
    print(f"Go to done")

    sleep(1.0)
    print(f"Go to: {goal1}")
    drone_interface.go_to_point(goal1, speed=speed, ignore_yaw=ignore_yaw_)
    print(f"Go to done")

    sleep(1.0)
    print(f"Go to: {goal2}")
    drone_interface.go_to_point(goal2, speed=speed, ignore_yaw=ignore_yaw_)
    print(f"Go to done")

    ##### FOLLOW PATH #####
    # sleep(1.0)
    # print(f"Follow path: [{goal0}, {goal1}, {goal2}]")
    # path = [goal0, goal1, goal2]
    # drone_interface.follow_path(path, speed=speed)
    # print(f"Follow path done")

    # sleep(1.0)
    print(f"Land: [{goal2}]")
    drone_interface.land(speed=0.5)
    print(f"Land done")

    print("Clean exit")


if __name__ == '__main__':
    rclpy.init()
    uav = DroneInterface("drone_sim_0", verbose=False, use_gps=False)

    drone_run(uav, 0)

    uav.shutdown()
    rclpy.shutdown()
    exit(0)
