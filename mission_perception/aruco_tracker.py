from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data


@dataclass
class MarkerDetection:
    marker_id: int
    t: float


class ArucoTracker:
    """
    Camera-based ArUco detector + temporal buffer.
    """

    def __init__(
        self,
        *,
        image_topic: str,
        dictionary=cv2.aruco.DICT_5X5_250,
        max_history_s: float = 10.0,
    ):
        self.image_topic = image_topic
        self.max_history_s = max_history_s

        self.bridge = CvBridge()

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        self._detections: Deque[MarkerDetection] = deque()
        self._total_frames: int = 0  # every callback invocation

        self._node = None
        self._attached = False

    # -------------------------
    # ROS
    # -------------------------
    def attach_ros(self, node):
        self._node = node

        node.create_subscription(
            Image,
            self.image_topic,
            self._img_callback,
            qos_profile_sensor_data,
        )

        self._attached = True

    # -------------------------
    # Detection
    # -------------------------
    def _img_callback(self, msg):
        self._total_frames += 1
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        corners, ids, _ = cv2.aruco.detectMarkers(
            frame,
            self.aruco_dict,
            parameters=self.aruco_params,
        )

        if ids is not None:
            ids = ids.flatten().tolist()
            print(f"Detected IDs: {ids}")
            now = time.time()

            for mid in ids:
                self._detections.append(MarkerDetection(int(mid), now))

        self._prune()

    def _prune(self):
        cutoff = time.time() - self.max_history_s
        while self._detections and self._detections[0].t < cutoff:
            self._detections.popleft()

    # -------------------------
    # Queries
    # -------------------------
    def recent_ids(self, window_s: float) -> List[int]:
        cutoff = time.time() - window_s
        return [d.marker_id for d in self._detections if d.t >= cutoff]

    def count_recent(self, marker_id: int, window_s: float) -> int:
        cutoff = time.time() - window_s
        return sum(1 for d in self._detections if d.t >= cutoff and d.marker_id == marker_id)

    def wait_for_marker(
        self,
        marker_id: int,
        *,
        timeout_s: float,
        recent_window_s: float,
        min_count: int = 1,
    ):
        t0 = time.time()
        frames_at_start = self._total_frames

        def _count_since_start():
            return sum(1 for d in self._detections if d.t >= t0 and d.marker_id == marker_id)

        def _recent_ids_since_start():
            return [d.marker_id for d in self._detections if d.t >= t0]

        while time.time() - t0 < timeout_s:
            count = _count_since_start()
            if count >= min_count:
                return {
                    "verified": True,
                    "marker_id": marker_id,
                    "count_in_window": count,
                    "recent_ids": _recent_ids_since_start(),
                    "frames_during_wait": self._total_frames - frames_at_start,
                }

            time.sleep(0.1)

        count = _count_since_start()
        verified = count >= min_count
        return {
            "verified": verified,
            "marker_id": marker_id,
            "count_in_window": count,
            "recent_ids": _recent_ids_since_start(),
            "frames_during_wait": self._total_frames - frames_at_start,
        }
