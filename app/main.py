from audioop import mul
import multiprocessing
import ipdb

import os
import sys
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import time
import threading

import data_engine


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualization import get_frame_from_pose
from utils import config, logger, utils

_config = config.Config()
_logger = logger.Logger().get()


class MainApp:
    def __init__(self, data_source) -> None:
        if os.path.isfile(str(data_source)):
            self._data_source = data_engine.PickleDataEngine(data_source)
        else:
            self._data_source = data_engine.FreenectDataEngine()

        self.stop_event = multiprocessing.Event()

        self._data_source = data_engine.PickleDataEngine('/Users/bugra.sefercik/workspace/repos/unknown_object_segmentation/dataset/alive_test_v2_splits.json')
        self.rot_mat = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ])

        self.window = gui.Application.instance.create_window(
            "Open3D - Video Example", 1000, 500)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)

        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.widget3d)

        self.lit = rendering.MaterialRecord()
        self.lit.shader = "defaultLit"

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
        kinect_frame = get_frame_from_pose(frame, [0] * 7)
        # kinect_frame.rotate(self.rot_mat)
        self.widget3d.scene.add_geometry("kinect_frame", kinect_frame, self.lit)

        data = self._data_source.get_frame()
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(data.points)
        self.pcd.colors = o3d.utility.Vector3dVector(data.rgb)
        self.pcd.rotate(np.array([
            [0, 0, -1],
            [0, 1, 0],
            [1, 0, 0]
        ]))
        self.widget3d.scene.add_geometry("pcd", self.pcd, self.lit)

        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(20.0, bounds, bounds.get_center())
        self.widget3d.scene.show_axes(False)
        # self.widget3d.add_3d_label(np.array([0,0,0]), "ALIVE")

        em = self.window.theme.font_size
        margin = 0.5 * em
        self.panel = gui.Vert(0.5 * em, gui.Margins(margin))

        # ipdb.set_trace()
        # self.panel.add_child(gui.Label("Color image"))
        # # self.rgb_widget = gui.ImageWidget(self.rgb_images[0])
        # self.panel.add_child(self.rgb_widget)
        # self.panel.add_child(gui.Label("Depth image (normalized)"))
        # # self.depth_widget = gui.ImageWidget(self.depth_images[0])
        # # self.panel.add_child(self.depth_widget)
        # self.window.add_child(self.panel)

        self.is_done = False
        threading.Thread(target=self._update_thread).start()

    def _on_layout(self, layout_context):
        contentRect = self.window.content_rect
        panel_width = 15 * layout_context.theme.font_size  # 15 ems wide
        self.widget3d.frame = gui.Rect(contentRect.x, contentRect.y,
                                       contentRect.width - panel_width,
                                       contentRect.height)
        self.panel.frame = gui.Rect(self.widget3d.frame.get_right(),
                                    contentRect.y, panel_width,
                                    contentRect.height)

    def _on_close(self):
        self.stop_event.set()
        print("Closing.")
        return True  # False would cancel the close

    def _update_thread(self):
        # This is NOT the UI thread, need to call post_to_main_thread() to update
        # the scene or any part of the UI.
        def update():
            data = self._data_source.get_frame()
            self.pcd.points = o3d.utility.Vector3dVector(data.points)
            self.pcd.colors = o3d.utility.Vector3dVector(data.rgb)
            # self.pcd.rotate(self.rot_mat)

            self.widget3d.scene.remove_geometry("pcd")
            self.widget3d.scene.add_geometry("pcd", self.pcd, self.lit)

        while not self.stop_event.is_set():
            time.sleep(0.2)

            if not self.stop_event.is_set():
                # Update the images. This must be done on the UI thread.
                gui.Application.instance.post_to_main_thread(
                    self.window, update)


def main():
    app = o3d.visualization.gui.Application.instance
    app.initialize()

    win = MainApp(_config.INFERENCE.data_source)

    app.run()


if __name__ == "__main__":
    main()
