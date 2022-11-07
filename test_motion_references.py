#!/bin/python3


import time
import rclpy
from python_interface.drone_interface import DroneInterface

import motion_reference_handlers.utils as mh_utils
from geometry_msgs.msg import PoseStamped, TwistStamped


def get_position_output(drone_interface: DroneInterface,
                        pose_list: list,
                        frame_id: str,
                        yaw_angle: float = 0.0):

    pose_frame_id = mh_utils.get_tf_name(drone_interface, frame_id)

    pose_msg = PoseStamped()
    pose_msg.header.frame_id = pose_frame_id
    pose_msg.pose.position.x = pose_list[0]
    pose_msg.pose.position.y = pose_list[1]
    pose_msg.pose.position.z = pose_list[2]

    pose_msg.pose.orientation = mh_utils.get_quaternion_from_yaw_angle(
        yaw_angle)
    return pose_msg


def get_speed_output(drone_interface: DroneInterface,
                     twist_list: list,
                     frame_id: str,
                     yaw_speed: float = 0.0):
    speed_frame_id = mh_utils.get_tf_name(drone_interface, frame_id)
    twist_msg = TwistStamped()
    twist_msg.header.frame_id = speed_frame_id
    twist_msg.twist.linear.x = twist_list[0]
    twist_msg.twist.linear.y = twist_list[1]
    twist_msg.twist.linear.z = twist_list[2]

    twist_msg.twist.angular.z = yaw_speed
    return twist_msg


def test_position_motion(drone_interface: DroneInterface):
    # ##### Motion Handler Position #####

    size = 0.0
    height = 3.0
    frame_id = '/earth'
    yaw_angle = 45.0 * 3.14 / 180.0
    yaw_speed = 0.2
    pose_list = [size, size, height]
    speed_limit = 1.0
    speed_limit_list = [speed_limit, speed_limit, speed_limit]

    pose_msg = get_position_output(
        drone_interface, pose_list, frame_id, yaw_angle)
    twist_limit_msg = get_speed_output(
        drone_interface, speed_limit_list, frame_id, yaw_speed)

    limit = 4
    time_sleep = 3.0

    # Test position motion using yaw angle
    i = 0
    while i < limit and rclpy.ok():
        print(
            f"{i} - Send position motion with yaw angle using list: [{pose_list[0]}, {pose_list[1]}, {pose_list[2]}]")
        pose_frame_id = mh_utils.get_tf_name(drone_interface, frame_id)
        drone_interface.position_motion_handler.send_position_command_with_yaw_angle(
            pose_list, yaw_angle=yaw_angle, pose_frame_id=pose_frame_id)
        time.sleep(time_sleep)
        i += 1

    i = 0
    while i < limit and rclpy.ok():
        print(
            f"{i} - Send position motion with yaw angle using msg: [{pose_msg.pose.position.x}, {pose_msg.pose.position.y}, {pose_msg.pose.position.z}]")
        drone_interface.position_motion_handler.send_position_command_with_yaw_angle(
            pose_msg)
        time.sleep(1)
        i += 1

    # Test position motion using yaw speed
    i = 0
    while i < limit and rclpy.ok():
        print(
            f"{i} - Send position motion with yaw speed using pose list and speed float: [{pose_list[0]}, {pose_list[1]}, {pose_list[2]}]")
        pose_frame_id = mh_utils.get_tf_name(drone_interface, frame_id)
        drone_interface.position_motion_handler.send_position_command_with_yaw_speed(
            pose_list, yaw_speed=yaw_speed, pose_frame_id=pose_frame_id)
        time.sleep(time_sleep)
        i += 1

    i = 0
    while i < limit and rclpy.ok():
        print(
            f"{i} - Send position motion with yaw speed using pose msg and speed float: [{pose_msg.pose.position.x}, {pose_msg.pose.position.y}, {pose_msg.pose.position.z}]")
        drone_interface.position_motion_handler.send_position_command_with_yaw_speed(
            pose_msg, yaw_speed=yaw_speed)
        time.sleep(1)
        i += 1

    i = 0
    while i < limit and rclpy.ok():
        print(
            f"{i} - Send position motion with yaw speed using pose msg and speed msg: [{pose_msg.pose.position.x}, {pose_msg.pose.position.y}, {pose_msg.pose.position.z}]")
        drone_interface.position_motion_handler.send_position_command_with_yaw_speed(
            pose_msg, twist_limit_msg)
        time.sleep(1)
        i += 1


def test_speed_motion(drone_interface: DroneInterface):
    # ##### Motion Handler Speed #####

    frame_id = '/earth'
    yaw_angle = 45.0 * 3.14 / 180.0
    yaw_speed = 0.2
    speed = 1.0
    speed_list = [speed, speed, 0.0]

    pose_msg = get_position_output(
        drone_interface, [0.0, 0.0, 0.0], frame_id, yaw_angle)
    twist_msg = get_speed_output(
        drone_interface, speed_list, frame_id, yaw_speed)

    limit = 4
    time_sleep = 3.0

    # Test speed motion using yaw angle
    i = 0
    while i < limit and rclpy.ok():
        print(
            f"{i} - Send speed motion with yaw angle using list: [{speed_list[0]}, {speed_list[1]}, {speed_list[2]}]")
        twist_frame_id = mh_utils.get_tf_name(drone_interface, frame_id)
        drone_interface.speed_motion_handler.send_speed_command_with_yaw_angle(
            speed_list, twist_frame_id=twist_frame_id, yaw_angle=yaw_angle)
        time.sleep(time_sleep)
        i += 1

    i = 0
    while i < limit and rclpy.ok():
        print(
            f"{i} - Send speed motion with yaw angle using msg and float: [{twist_msg.twist.linear.x}, {twist_msg.twist.linear.y}, {twist_msg.twist.linear.z}]")
        drone_interface.speed_motion_handler.send_speed_command_with_yaw_angle(
            twist_msg, yaw_angle=yaw_angle)
        time.sleep(time_sleep)
        i += 1

    i = 0
    while i < limit and rclpy.ok():
        print(
            f"{i} - Send speed motion with yaw angle using msg and msg: [{twist_msg.twist.linear.x}, {twist_msg.twist.linear.y}, {twist_msg.twist.linear.z}]")
        drone_interface.speed_motion_handler.send_speed_command_with_yaw_angle(
            twist_msg, pose_msg)
        time.sleep(time_sleep)
        i += 1

    # Test position motion using yaw speed
    i = 0
    while i < limit and rclpy.ok():
        print(
            f"{i} - Send speed motion with yaw speed using list: [{speed_list[0]}, {speed_list[1]}, {speed_list[2]}]")
        twist_frame_id = mh_utils.get_tf_name(drone_interface, frame_id)
        drone_interface.speed_motion_handler.send_speed_command_with_yaw_speed(
            speed_list, twist_frame_id=twist_frame_id, yaw_speed=yaw_speed)
        time.sleep(time_sleep)
        i += 1

    i = 0
    while i < limit and rclpy.ok():
        print(
            f"{i} - Send speed motion with yaw speed using msg: [{twist_msg.twist.linear.x}, {twist_msg.twist.linear.y}, {twist_msg.twist.linear.z}]")
        drone_interface.speed_motion_handler.send_speed_command_with_yaw_speed(
            twist_msg)
        time.sleep(time_sleep)
        i += 1


def test_speed_in_a_plane_motion(drone_interface: DroneInterface):
    # ##### Motion Handler Speed In A Plane#####

    frame_id = '/earth'
    yaw_angle = 45.0 * 3.14 / 180.0
    yaw_speed = 0.2
    speed = 1.0
    speed_list = [speed, speed, 0.0]

    height = 1.0
    pose_frame_id = mh_utils.get_tf_name(drone_interface, frame_id)
    pose_msg = get_position_output(
        drone_interface, [0.0, 0.0, height], frame_id, yaw_angle)
    pose_msg.header.frame_id = pose_frame_id
    twist_frame_id = mh_utils.get_tf_name(drone_interface, frame_id)
    twist_msg = get_speed_output(
        drone_interface, speed_list, frame_id, yaw_speed)
    twist_msg.header.frame_id = twist_frame_id

    limit = 4
    time_sleep = 3.0

    # Test speed motion using yaw angle
    i = 0
    while i < limit and rclpy.ok():
        print(f"{i} - Send speed motion with yaw angle {yaw_angle} using list:")
        print(f"{i} - speed: [{speed_list[0]}, {speed_list[1]}]")
        print(f"{i} - height: {height}")
        drone_interface.speed_in_a_plane_motion_handler.send_speed_in_a_plane_command_with_yaw_angle(
            speed_list, height, twist_frame_id=twist_frame_id, pose_frame_id=pose_frame_id, yaw_angle=yaw_angle)
        time.sleep(time_sleep)
        i += 1

    i = 0
    while i < limit and rclpy.ok():
        print(f"{i} - Send speed motion with yaw angle {yaw_angle} using msg:")
        print(
            f"{i} - speed: [{twist_msg.twist.linear.x}, {twist_msg.twist.linear.y}]")
        print(f"{i} - height: [{pose_msg.pose.position.z}]")
        drone_interface.speed_in_a_plane_motion_handler.send_speed_in_a_plane_command_with_yaw_angle(
            twist_msg, pose_msg)
        time.sleep(time_sleep)
        i += 1

    # Test speed motion using yaw speed
    i = 0
    while i < limit and rclpy.ok():
        print(f"{i} - Send speed motion with yaw speed {yaw_speed} using list:")
        print(f"{i} - speed: [{speed_list[0]}, {speed_list[1]}]")
        print(f"{i} - height: {height}")
        drone_interface.speed_in_a_plane_motion_handler.send_speed_in_a_plane_command_with_yaw_speed(
            speed_list, height, twist_frame_id=twist_frame_id, pose_frame_id=pose_frame_id, yaw_speed=yaw_speed)
        time.sleep(time_sleep)
        i += 1

    i = 0
    while i < limit and rclpy.ok():
        print(f"{i} - Send speed motion with yaw speed {yaw_speed} using msg:")
        print(
            f"{i} - speed: [{twist_msg.twist.linear.x}, {twist_msg.twist.linear.y}]")
        print(f"{i} - height: [{pose_msg.pose.position.z}]")
        drone_interface.speed_in_a_plane_motion_handler.send_speed_in_a_plane_command_with_yaw_speed(
            twist_msg, pose_msg)
        time.sleep(time_sleep)
        i += 1


def drone_run(drone_interface: DroneInterface):

    takeoff_height = 2.0
    takeoff_speed = 0.5

    print("Start mission")

    drone_interface.offboard()
    drone_interface.arm()

    print("Take Off")
    drone_interface.takeoff(takeoff_height, speed=takeoff_speed)
    print("Take Off done")

    time.sleep(1.0)

    print("Test position motion")
    drone_interface.hover_motion_handler.send_hover()
    print("Test position motion done")

    time.sleep(1.0)

    print("Test position motion")
    test_position_motion(drone_interface)
    print("Test position motion done")

    time.sleep(1.0)

    print("Test speed motion")
    test_speed_motion(drone_interface)
    print("Test speed motion done")

    time.sleep(1.0)

    print("Test speed in a plane motion")
    test_speed_in_a_plane_motion(drone_interface)
    print("Test speed in a plane motion done")

    print("Clean exit")
    return


if __name__ == '__main__':
    rclpy.init()
    uav = DroneInterface("drone_sim_0", verbose=False,
                         use_gps=False, use_sim_time=True)

    drone_run(uav)

    uav.shutdown()
    rclpy.shutdown()
    exit(0)
