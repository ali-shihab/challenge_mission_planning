from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, asdict
from typing import Any, Deque, Dict, List, Optional


@dataclass
class MarkerDetection:
    marker_id: int
    t_wall: float
    source_topic: str
    source_type: str


class ArucoTracker:
    """
    Tracks recent ArUco detections from a ROS topic.

    Supported topic types:
      - ros2_aruco_interfaces/msg/ArucoMarkers
      - std_msgs/msg/Int32MultiArray
      - visualization_msgs/msg/MarkerArray

    This is intentionally lightweight: it only stores marker IDs + timestamps.
    """

    def __init__(self, *, max_history_s: float = 10.0) -> None:
        self.max_history_s = float(max_history_s)
        self._detections: Deque[MarkerDetection] = deque()
        self._node = None
        self._topic: Optional[str] = None
        self._topic_type: Optional[str] = None
        self._logger = None
        self._attached = False

    def _log(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if self._logger is not None:
            try:
                self._logger.event(event, payload or {})
            except Exception:
                pass

    def _prune(self) -> None:
        cutoff = time.time() - self.max_history_s
        while self._detections and self._detections[0].t_wall < cutoff:
            self._detections.popleft()

    def _record_ids(self, ids: List[int], source_type: str) -> None:
        now = time.time()
        self._prune()
        for mid in ids:
            self._detections.append(
                MarkerDetection(
                    marker_id=int(mid),
                    t_wall=now,
                    source_topic=str(self._topic),
                    source_type=source_type,
                )
            )
        if ids:
            self._log("aruco_detections_received", {
                "ids": [int(x) for x in ids],
                "count": len(ids),
                "topic": self._topic,
                "topic_type": source_type,
            })

    def attach_ros(self, node, topic: str, logger=None) -> bool:
        """
        Attach to an ArUco detection topic by discovering its message type.
        """
        self._node = node
        self._topic = topic
        self._logger = logger

        topic_map = dict(node.get_topic_names_and_types())
        topic_types = topic_map.get(topic, [])

        self._log("aruco_topic_types", {
            "topic": topic,
            "types": topic_types,
        })

        if not topic_types:
            self._log("aruco_attach_failed", {
                "reason": "topic_not_found",
                "topic": topic,
            })
            return False

        # 1) ros2_aruco_interfaces/msg/ArucoMarkers
        if "ros2_aruco_interfaces/msg/ArucoMarkers" in topic_types:
            try:
                from ros2_aruco_interfaces.msg import ArucoMarkers  # type: ignore
            except Exception as e:
                self._log("aruco_attach_failed", {
                    "reason": "import_failed",
                    "topic": topic,
                    "msg_type": "ros2_aruco_interfaces/msg/ArucoMarkers",
                    "error": str(e),
                })
                return False

            def cb(msg):
                ids = []
                if hasattr(msg, "marker_ids"):
                    ids = [int(x) for x in msg.marker_ids]
                self._record_ids(ids, "ros2_aruco_interfaces/msg/ArucoMarkers")

            node.create_subscription(ArucoMarkers, topic, cb, 10)
            self._topic_type = "ros2_aruco_interfaces/msg/ArucoMarkers"
            self._attached = True
            self._log("aruco_attached", {
                "topic": topic,
                "msg_type": self._topic_type,
            })
            return True

        # 2) std_msgs/msg/Int32MultiArray
        if "std_msgs/msg/Int32MultiArray" in topic_types:
            try:
                from std_msgs.msg import Int32MultiArray  # type: ignore
            except Exception as e:
                self._log("aruco_attach_failed", {
                    "reason": "import_failed",
                    "topic": topic,
                    "msg_type": "std_msgs/msg/Int32MultiArray",
                    "error": str(e),
                })
                return False

            def cb(msg):
                ids = []
                if hasattr(msg, "data"):
                    ids = [int(x) for x in msg.data]
                self._record_ids(ids, "std_msgs/msg/Int32MultiArray")

            node.create_subscription(Int32MultiArray, topic, cb, 10)
            self._topic_type = "std_msgs/msg/Int32MultiArray"
            self._attached = True
            self._log("aruco_attached", {
                "topic": topic,
                "msg_type": self._topic_type,
            })
            return True

        # 3) visualization_msgs/msg/MarkerArray
        if "visualization_msgs/msg/MarkerArray" in topic_types:
            try:
                from visualization_msgs.msg import MarkerArray  # type: ignore
            except Exception as e:
                self._log("aruco_attach_failed", {
                    "reason": "import_failed",
                    "topic": topic,
                    "msg_type": "visualization_msgs/msg/MarkerArray",
                    "error": str(e),
                })
                return False

            def cb(msg):
                ids = []
                if hasattr(msg, "markers"):
                    for m in msg.markers:
                        if hasattr(m, "id"):
                            ids.append(int(m.id))
                self._record_ids(ids, "visualization_msgs/msg/MarkerArray")

            node.create_subscription(MarkerArray, topic, cb, 10)
            self._topic_type = "visualization_msgs/msg/MarkerArray"
            self._attached = True
            self._log("aruco_attached", {
                "topic": topic,
                "msg_type": self._topic_type,
            })
            return True

        self._log("aruco_attach_failed", {
            "reason": "unsupported_topic_type",
            "topic": topic,
            "types": topic_types,
        })
        return False

    def attached(self) -> bool:
        return self._attached

    def recent_ids(self, window_s: float) -> List[int]:
        self._prune()
        cutoff = time.time() - float(window_s)
        return [d.marker_id for d in self._detections if d.t_wall >= cutoff]

    def count_recent(self, marker_id: int, window_s: float) -> int:
        self._prune()
        cutoff = time.time() - float(window_s)
        return sum(1 for d in self._detections if d.t_wall >= cutoff and d.marker_id == int(marker_id))

    def seen_recently(self, marker_id: int, window_s: float, min_count: int = 1) -> bool:
        return self.count_recent(marker_id, window_s) >= int(min_count)

    def wait_for_marker(
        self,
        marker_id: int,
        *,
        timeout_s: float,
        recent_window_s: float,
        min_count: int = 1,
        poll_s: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Wait until marker_id has been observed min_count times within recent_window_s.
        """
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            count = self.count_recent(marker_id, recent_window_s)
            if count >= min_count:
                return {
                    "verified": True,
                    "marker_id": int(marker_id),
                    "count_in_window": int(count),
                    "recent_window_s": float(recent_window_s),
                    "timeout_s": float(timeout_s),
                    "elapsed_s": time.time() - t0,
                    "recent_ids": self.recent_ids(recent_window_s),
                }
            time.sleep(poll_s)

        return {
            "verified": False,
            "marker_id": int(marker_id),
            "count_in_window": int(self.count_recent(marker_id, recent_window_s)),
            "recent_window_s": float(recent_window_s),
            "timeout_s": float(timeout_s),
            "elapsed_s": time.time() - t0,
            "recent_ids": self.recent_ids(recent_window_s),
        }
