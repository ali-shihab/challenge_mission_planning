#!/bin/python3

import rclpy
import sys
import threading
from typing import List
from as2_python_api.drone_interface import DroneInterface

drones_ns = [
    'drone_sim_0',
    'drone_sim_1',
    'drone_sim_2'
    ]

speed = 0.5
ingore_yaw = True

h1 = 1.0
h2 = 1.5
h3 = 2.0


v0 = [-2, -1, h1]
v1 = [0, 2, h1]
v2 = [2, -1, h1]

v3 = [2, 1, h2]
v4 = [-2, 1, h2]
v5 = [0, -2, h2]


v6 = v0
v6[2] = h3
v7 = v1
v7[2] = h3
v8 = v2
v8[2] = h3


l0 = [2, 0, 1.0]
l1 = [-2, 0, 1.0]
l2 = [0, 0, 1.0]


# pos0= [[1,1,1],[1,-1,1.5],[-1,-1,1.0],[-1,0,1]]
# pos1= [[-1,-1,1],[-1,1,1.5],[1,1,2.0],[1,0,1]]
# pos2= [[-1,-1,1],[-1,1,1.5],[1,1,2.0],[1,0,1]]

# pos0= [[1,1,1],[1,-1,1.5],[-1,-1,1.0],[-1,0,1]]

pos0 = [v2, v1, v0, v2, v5, v4, v3, v5, v8, v7, v6, v8, v2, l0]
pos1 = [v0, v2, v1, v0, v4, v3, v5, v4, v6, v8, v7, v6, v0, l1]
pos2 = [v1, v0, v2, v1, v3, v5, v4, v3, v7, v6, v8, v7, v1, l2]

# drones_ns=['cf0','cf1']
# drones_ns=['cf1']

n_point_0 = 0
n_point_1 = 0
n_point_2 = 0


def reset_point():
    global n_point_0, n_point_1, n_point_2
    n_point_0 = 0
    n_point_1 = 0
    n_point_2 = 0


def pose_generator(uav: DroneInterface):
    global n_point_0, n_point_1, n_point_2
    if uav.get_namespace()[-1] == '0':
        ret = pos0[n_point_0]
        n_point_0 += 1
    elif uav.get_namespace()[-1] == '1':
        ret = pos1[n_point_1]
        n_point_1 += 1
    elif uav.get_namespace()[-1] == '2':
        ret = pos2[n_point_2]
        n_point_2 += 1
    return ret


def shutdown_all(uavs):
    print("Exiting...")
    for uav in uavs:
        uav.shutdown()
    sys.exit(1)

# create decorator for creating a thread for each drone


def takeoff(uav: DroneInterface):
    uav.arm()
    uav.offboard()
    uav.takeoff(1, 0.7)
    # sleep(1)


def land(drone_interface: DroneInterface):
    drone_interface.land(0.4)


def go_to(drone_interface: DroneInterface):
    drone_interface.go_to.go_to_point(pose_generator(
        drone_interface), speed=speed)


def confirm(uavs: List[DroneInterface], msg: str = 'Continue') -> bool:
    confirmation = input(f"{msg}? (y/n): ")
    if confirmation == "y":
        return True
    elif confirmation == "n":
        return False
    else:
        shutdown_all(uavs)


def run_func(uavs: List[DroneInterface], func, *args):
    threads = []
    for uav in uavs:
        t = threading.Thread(target=func, args=(uav, *args))
        threads.append(t)
        t.start()
    print("Waiting for threads to finish...")
    for t in threads:
        t.join()
    print("all done")


def move_uavs(uavs):
    reset_point()
    for i in range(len(pos0)):
        run_func(uavs, go_to)
    return


if __name__ == '__main__':

    rclpy.init()
    uavs = []
    for i in range(len(drones_ns)):
        uavs.append(DroneInterface(drones_ns[i], use_sim_time=True, verbose=False))

    print("Takeoff")
    confirm(uavs, "Takeoff")
    run_func(uavs, takeoff)

    print("Go to")
    confirm(uavs, "Go to")
    move_uavs(uavs)
    while confirm(uavs, "Replay"):
        move_uavs(uavs)

    print("Land")
    confirm(uavs, "Land")
    run_func(uavs, land)

    print("Shutdown")
    shutdown_all(uavs)
    rclpy.shutdown()

    exit(0)
