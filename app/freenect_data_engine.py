from datetime import datetime
import threading
import time
import ipdb

import os
import sys
import struct
import ctypes
import asyncio
from queue import Queue, Full, Empty

import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
import sensor_msgs.point_cloud2 as pc2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.ros_utils import get_points_and_colors

from data_engine import DataEngineInterface
from dto import PointCloudDTO


class FreenectDataEngine(DataEngineInterface):
    def __init__(self, fps=2):
        self.min_wait_time = 1 / fps

        self.exit_event = threading.Event()

        self.pc2_queue = Queue(1)
        self.ee_pose_queue = Queue(1)
        self.ready_queue = Queue(1)

        self._dto_thread = threading.Thread(target=self._dto_generation_thread)
        self._dto_thread.start()

    def get(self) -> PointCloudDTO:
        try:
            return self.ready_queue.get(timeout=1)
        except Empty:
            return None

    def run(self):
        rospy.init_node("depth_registered_recorder")
        rospy.Subscriber("/camera/depth_registered/points", PointCloud2, self._callback)
        rospy.Subscriber("/robot/ee_pose", PoseStamped, self._ee_pose_callback)

    def _dto_generation_thread(self):
        while not self.exit_event.is_set():
            start = time.time()
            data = self.pc2_queue.get()
            if data is None:
                return

            ee2base_data = None
            try:
                raw = self.ee_pose_queue.get_nowait()
                ee2base_data = np.array([
                    raw.pose.position.x,
                    raw.pose.position.y,
                    raw.pose.position.z,
                    raw.pose.orientation.w,
                    raw.pose.orientation.x,
                    raw.pose.orientation.y,
                    raw.pose.orientation.z
                ])
            except Empty:
                pass

            points, colors = get_points_and_colors(data)

            new_dto = PointCloudDTO(
                points=points,
                rgb=colors / 255,
                timestamp=datetime.utcnow(),
                ee2base_pose=ee2base_data
            )

            try:
                self.ready_queue.get_nowait()
            except Empty:
                pass

            if not self.exit_event.is_set():
                self.ready_queue.put(new_dto)

            duration = time.time() - start

            # print("dur", duration, "sleep", max(0, self.min_wait_time - duration))
            time.sleep(max(0, self.min_wait_time - duration))

    def _callback(self, data):
        try:
            if not self.exit_event.is_set():
                self.pc2_queue.put_nowait(data)
        except Full:
            pass

    def _ee_pose_callback(self, data):
        try:
            if not self.exit_event.is_set():
                self.ee_pose_queue.put_nowait(data)
        except Full:
            pass

    def exit(self):
        self.exit_event.set()
        time.sleep(self.min_wait_time)
        self.pc2_queue.put(None)
        self.ee_pose_queue.put(None)
        time.sleep(0.2)
        self._dto_thread.join()
