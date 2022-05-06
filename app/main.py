import os
import sys
import multiprocessing
import ipdb
import time
import threading
import queue
import copy

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualization import get_frame_from_pose
from utils import config, logger, utils

import data_engine
from inference_engine import InferenceEngine

import torch
import MinkowskiEngine as ME

_config = config.Config()
_logger = logger.Logger().get()

_use_cuda = torch.cuda.is_available()
_device = torch.device("cuda" if _use_cuda else "cpu")


class MainApp:
    def __init__(self, data_source) -> None:
        if os.path.isfile(str(data_source)):
            self._data_source = data_engine.PickleDataEngine(data_source)
        else:
            self._data_source = data_engine.FreenectDataEngine()
        self.dummy = self._data_source.get_frame()
        self._inference_engine = InferenceEngine()

        self.stop_event = multiprocessing.Event()
        self._seg_event = multiprocessing.Event()
        self._calibration_event = multiprocessing.Event()
        self._inference_queue = queue.Queue(1)

        self.rot_mat = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ])

        self.window = gui.Application.instance.create_window(
            "Alive Robot Calibration Tool", 1000, 500)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)

        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.widget3d)

        self.lit = rendering.MaterialRecord()
        self.lit.shader = "defaultLitTransparency"

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
        self.panel = gui.Vert(0.5 * em, gui.Margins(left=margin, top=margin, right=margin))

        # ipdb.set_trace()

        self._seg_button = gui.Button("Show Segmentation")
        self._seg_button.horizontal_padding_em = 0.5
        self._seg_button.vertical_padding_em = 0.5
        self._seg_button.set_on_clicked(self._toggle_segmentation)
        self.panel.add_child(self._seg_button)

        self._calibrate_button = gui.Button("Calibrate")
        self._calibrate_button.horizontal_padding_em = 0.5
        self._calibrate_button.vertical_padding_em = 0.5
        self._calibrate_button.set_on_clicked(self._calibrate)
        self.panel.add_child(self._calibrate_button)

        self._results_label = gui.Label("Calibration Results")
        self.panel.add_child(self._results_label)

        self.window.add_child(self.panel)

        threading.Thread(target=self._update_thread).start()
        threading.Thread(target=self._calibration_thread).start()
        threading.Thread(target=self._inference_thread).start()

    def _inference_thread(self):
        while True:
            data = self._inference_queue.get()

            if self.stop_event.is_set():
                return

            rgb = torch.from_numpy(data.rgb).to(dtype=torch.float32)
            points = torch.from_numpy(data.points).to(dtype=torch.float32)

            in_field = ME.TensorField(
                features=rgb,
                coordinates=ME.utils.batched_coordinates([points / 0.02], dtype=torch.float32),
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                device=_device,
            )
            sinput = in_field.sparse()
            soutput = self._inference_engine.segmentation_model(sinput)
            soutput1 = self._inference_engine.rotation_model(sinput)
            soutput2 = self._inference_engine.translation_model(sinput)
            out_field = soutput.slice(in_field)

            print('inference', soutput2.F[0], data.timestamp)

    def _calibrate(self):
        self._calibrate_button.enabled = False
        self._calibration_event.set()

    def _calibration_thread(self):
        while True:
            self._calibration_event.wait()
            if self.stop_event.is_set():
                return

            self._results_label.text = "calibration yo!"

            time.sleep(5)

            self._results_label.text = "x: \ny: \nz: \nq_w: \nq_x: \nq_y: \nq_z: \n"
            self._calibrate_button.enabled = True
            self._calibration_event.clear()

    def _toggle_segmentation(self):
        # self._segmentation_enabled = not self._segmentation_enabled

        if self._seg_event.is_set():
            self._seg_event.clear()
            self._seg_button.text = "Show Segmentation"
        else:
            self._seg_event.set()
            self._seg_button.text = "Hide Segmentation"

        return True

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
        self.stop_event.set()  # set before all
        time.sleep(0.1)
        self._calibration_event.set()  # need to clear loop
        self._inference_queue.put(dict())  # needs to clear inference loop
        time.sleep(5)

        _logger.info("Closing.")

        return True  # False would cancel the close

    def _update_thread(self):
        # This is NOT the UI thread, need to call post_to_main_thread() to update
        # the scene or any part of the UI.
        def update():
            data = self._data_source.get_frame()

            try:
                self._inference_queue.put_nowait(copy.deepcopy(data))
            except queue.Full:
                pass

            self.pcd.points = o3d.utility.Vector3dVector(data.points)

            if self._seg_event.is_set():
                data.rgb *= 0
            self.pcd.colors = o3d.utility.Vector3dVector(data.rgb)
            # self.pcd.rotate(self.rot_mat)

            print('update gui', data.timestamp)

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
