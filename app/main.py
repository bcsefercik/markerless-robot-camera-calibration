import copy
import multiprocessing
import os
import queue
import sys
import threading
import time
import typing

import ipdb
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import MinkowskiEngine as ME
import torch
from utils import config, logger, utils
from utils.visualization import (
    get_frame_from_pose,
    generate_colors,
    create_coordinate_frame,
    generate_key_point_shapes
)

import data_engine
from inference_engine import InferenceEngine
from dto import ResultDTO

_config = config.Config()
_logger = logger.Logger().get()

_use_cuda = torch.cuda.is_available()
_device = torch.device("cuda" if _use_cuda else "cpu")

BASE_PATH = os.path.abspath(os.path.dirname(__file__))


class MainApp:
    def __init__(self, data_source) -> None:
        if not os.path.isabs(str(data_source)):
            data_source = os.path.join(
                os.path.dirname(BASE_PATH),
                str(data_source)
            )

        if os.path.isfile(str(data_source)):
            self._data_source = data_engine.PickleDataEngine(data_source)
        else:
            import freenect_data_engine

            self._data_source = freenect_data_engine.FreenectDataEngine()

        self._inference_engine = InferenceEngine()

        self.stop_event = multiprocessing.Event()
        self._seg_event = multiprocessing.Event()
        self._calibration_event = multiprocessing.Event()
        self._collection_event = multiprocessing.Event()
        self._inference_queue = queue.Queue(1)
        self._calibration_queue = queue.Queue(128)

        self._calibration_data: typing.List[typing.List[ResultDTO]] = list()
        self.calibration_result = None

        self.rot_mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        if _config.INFERENCE.SEGMENTATION.class_colors is not None:
            self._seg_colors = [
                (int(color[0:2], 16), int(color[2:4], 16), int(color[4:], 16))
                for color in _config.INFERENCE.SEGMENTATION.class_colors
            ]
            self._seg_colors = np.array(self._seg_colors) / 255
        else:
            np.random.seed(2)
            self._seg_colors = generate_colors(
                len(_config.INFERENCE.SEGMENTATION.classes)
            )

        self.window = gui.Application.instance.create_window(
            "Markerless Robot-Depth Camera Calibration Tool",
            # "Alive Lab Robot Calibration Tool - Koç University, Istanbul, Turkey",
            1000,
            500,
        )
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)

        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.widget3d)

        self.lit = rendering.MaterialRecord()
        self.lit.shader = "defaultUnlit"

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        self.kinect_frame = get_frame_from_pose(frame, [0] * 7)
        # kinect_frame.rotate(self.rot_mat)
        self.widget3d.scene.add_geometry("kinect_frame", self.kinect_frame, self.lit)
        self.widget3d.scene.show_geometry("kinect_frame", False)

        self.ee_frame = create_coordinate_frame([0, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        self.base_frame = create_coordinate_frame([-0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        # self.widget3d.scene.add_geometry("ee_frame", self.ee_frame, self.lit)

        self.calibrated_base_frame = create_coordinate_frame(
            [-0.2, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        )
        self.widget3d.scene.add_geometry(
            "calibrated_base_frame", self.calibrated_base_frame, self.lit
        )
        self.widget3d.scene.show_geometry("calibrated_base_frame", False)

        self.calibrated_base_label_text = 'Calibrated\n    Frame'
        self.calibrated_base_label = self.widget3d.add_3d_label(np.array([0,0,0]), '')
        self.calibrated_base_label.color = gui.Color(0, 0, 0)
        self.calibrated_base_label.scale = 1.3

        _init_points = (np.random.rand(200000, 3) - 0.5) * 3
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(_init_points)
        self.pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(_init_points))
        self.widget3d.scene.add_geometry("pcd", self.pcd, self.lit)

        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(20.0, bounds, bounds.get_center())
        self.widget3d.scene.show_axes(False)
        # self.widget3d.add_3d_label(np.array([0,0,0]), "ALIVE")

        em = self.window.theme.font_size
        margin = 0.5 * em
        self.panel = gui.Vert(
            0.5 * em, gui.Margins(left=margin, top=margin, right=margin)
        )

        self._seg_check = gui.Checkbox("Segmentation")
        self._seg_check.checked = False
        self._seg_check.set_on_checked(lambda state: self._seg_event.set() if state else self._seg_event.clear())
        self.panel.add_child(self._seg_check)

        self._kinect_frame_check = gui.Checkbox("Camera Frame")
        self._kinect_frame_check.checked = False
        self._kinect_frame_check.set_on_checked(lambda state: self.widget3d.scene.show_geometry("kinect_frame", state))
        self.panel.add_child(self._kinect_frame_check)

        self._instant_pred_check = gui.Checkbox("Instant Prediction")
        self._instant_pred_check.checked = True
        self._instant_pred_check.set_on_checked(
            lambda state: self.widget3d.scene.show_geometry("ee_frame", state)
        )
        self.panel.add_child(self._instant_pred_check)

        self._toggle_pred = gui.ToggleSwitch("from Key Points")
        self.panel.add_child(self._toggle_pred)

        self._kp_check = gui.Checkbox("Key Point Prediction")
        self._kp_check.checked = True
        self._kp_check.set_on_checked(
            lambda state: self.widget3d.scene.show_geometry("key_points", state)
        )
        self.panel.add_child(self._kp_check)

        self._calibrated_pred_check = gui.Checkbox("Latest Calibrated Prediction")
        self._calibrated_pred_check.checked = False
        self._calibrated_pred_check.enabled = False
        self._calibrated_pred_check.set_on_checked(self._show_calib_pred)
        self.panel.add_child(self._calibrated_pred_check)

        self._collect_data_button = gui.Button("Collect Data")
        self._collect_data_button.vertical_padding_em = 0.5
        self._collect_data_button.set_on_clicked(self._collect_data)
        self.panel.add_child(self._collect_data_button)

        self._calibrate_button = gui.Button("Calibrate")
        self._calibrate_button.vertical_padding_em = 0.5
        self._calibrate_button.set_on_clicked(self._calibrate)
        self._calibrate_button.enabled = False
        self.panel.add_child(self._calibrate_button)

        self._results_label = gui.Label("")
        self.panel.add_child(self._results_label)

        self.window.add_child(self.panel)

        self.logo_panel = gui.Vert(
            0.5 * em, gui.Margins(left=margin, bottom=margin)
        )
        self.logo_img = o3d.io.read_image(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
            'logo.png'
        ))
        self.logo_rgb_widget = gui.ImageWidget(self.logo_img)
        self.logo_panel.add_child(self.logo_rgb_widget)
        # self.window.add_child(self.logo_panel)

        self.warning_panel = gui.Horiz(
            0.5 * em, gui.Margins(left=margin, right=margin, top=0.68 * em, bottom=margin)
        )
        self._warning_label = gui.Label("Please, move the end effector to a more visible position!")
        self._warning_label.text_color = gui.Color(1, 1, 1, 1)
        self._warning_label.font_id = 1
        self.warning_panel.add_child(self._warning_label)
        self.window.add_child(self.warning_panel)

        self.notification_panel = gui.Horiz(
            0.5 * em, gui.Margins(left=margin, right=margin, top=0.68 * em, bottom=margin)
        )
        self._notification_label = gui.Label("Collecting data. Please, do not move the end effector.")
        self._notification_label.text_color = gui.Color(1, 1, 1, 1)
        self._notification_label.font_id = 1
        self.notification_panel.add_child(self._notification_label)
        self.window.add_child(self.notification_panel)

        threading.Thread(target=self._update_thread).start()
        threading.Thread(target=self._calibration_thread).start()
        threading.Thread(target=self._collection_thread).start()

        self._data_source.run()

    def _show_calib_pred(self, state):
        self.calibrated_base_label.text = self.calibrated_base_label_text if state else ''
        self.calibrated_base_label.position = self.calibration_result.pose_camera_link[:3]
        self.widget3d.scene.show_geometry(
                "calibrated_base_frame", state
            )

    def _collect_data(self):
        self._calibration_data.append(list())

        self._collection_event.set()
        self._calibrate_button.enabled = False
        self._collect_data_button.enabled = False

        self._notification_label.text = "Collecting data. Please, do not move the end effector. \n"
        self.notification_panel.visible = True

    def _collection_thread(self):
        # needs to stay alive
        while True:
            self._collection_event.wait()
            if self.stop_event.is_set():
                break

            result = self._calibration_queue.get()

            self._calibration_data[-1].append(result)

            self._notification_label.text = "Collecting data. Please, do not move the end effector. \n"

            self._notification_label.text += f"Position: #{len(self._calibration_data)}, Frame: {len(self._calibration_data[-1])}/{_config.INFERENCE.CALIBRATION.num_of_frames}"
            # print(len(self._calibration_data[-1]), result)

            if len(self._calibration_data[-1]) >= _config.INFERENCE.CALIBRATION.num_of_frames:
                time.sleep(0.1)
                self._collect_data_button.enabled = True

                self._collection_event.clear()
                with self._calibration_queue.mutex:
                    self._calibration_queue.queue.clear()
                self.notification_panel.visible = False

                self._calibrate_button.enabled = len(self._calibration_data) >= _config.INFERENCE.CALIBRATION.min_num_of_positions

        return True

    def _calibrate(self):
        self._calibrate_button.enabled = False

        self._calibrated_pred_check.checked = False
        self._calibrated_pred_check.enabled = False
        self.widget3d.scene.show_geometry("calibrated_base_frame", False)

        self._calibration_event.set()

    def _calibration_thread(self):
        # needs to stay alive
        while True:
            self._calibration_event.wait()
            if self.stop_event.is_set():
                break

            self.notification_panel.visible = True

            self._notification_label.text = "Calibration in progress."
            calibration_input = {f'p{i}': v for i, v in enumerate(self._calibration_data)}

            self.calibration_result = self._inference_engine.calibrate(calibration_input)
            if self.calibration_result.pose_camera_link is not None:
                cr = self.calibration_result.pose_camera_link
                title = "camera_rgb_optical_frame\npanda_link0:\n\n"
                self._results_label.text = title + f"x:\t{cr[0]:.4f}\ny:\t{cr[1]:.4f}\nz:\t{cr[2]:.4f}\nq_w:\t{cr[3]:.4f}\nq_x:\t{cr[4]:.4f}\nq_y:\t{cr[5]:.4f}\nq_z:\t{cr[6]:.4f}\n"

                # self.calibrated_base_frame = "nanananan"

                self._calibrate_button.enabled = False
                self._calibrated_pred_check.checked = True
                self._show_calib_pred(True)
                self._calibrated_pred_check.enabled = True
                self.calibrated_base_frame = create_coordinate_frame(
                    self.calibration_result.pose_camera_link,
                    length=0.24,
                    radius=0.012,
                    switch_w=False
                )
                self.widget3d.scene.remove_geometry("calibrated_base_frame")
                self.widget3d.scene.add_geometry("calibrated_base_frame", self.calibrated_base_frame, self.lit)
                self.widget3d.scene.show_geometry("calibrated_base_frame", self._calibrated_pred_check.checked)
            else:
                self._results_label.text = "No calibration,\ntry again."

            self.notification_panel.visible = False
            self._calibration_event.clear()
            self._collection_event.clear()
            self._calibration_data.clear()

        return True

    def _on_layout(self, layout_context):
        contentRect = self.window.content_rect
        panel_width = 15 * layout_context.theme.font_size  # 15 ems wide
        self.widget3d.frame = gui.Rect(
            contentRect.x,
            contentRect.y,
            contentRect.width - panel_width,
            contentRect.height,
        )
        self.panel.frame = gui.Rect(
            self.widget3d.frame.get_right(),
            contentRect.y,
            panel_width,
            contentRect.height,
        )
        self.warning_panel.frame = gui.Rect(
            self.widget3d.frame.get_left(),
            contentRect.y,
            contentRect.width - panel_width,
            46,
        )
        self.warning_panel.background_color = gui.Color(0.9, 0.3, 0.3, 0.96)
        self.warning_panel.visible = False

        self.notification_panel.frame = gui.Rect(
            self.widget3d.frame.get_left(),
            contentRect.y,
            contentRect.width - panel_width,
            72,
        )
        self.notification_panel.background_color = gui.Color(0.5, 0.5, 0.5, 0.96)
        self.notification_panel.visible = False

        logo_panel_height = 46
        logo_panel_width = 159
        self.logo_panel.frame = gui.Rect(
            self.widget3d.frame.get_left(),
            contentRect.height - logo_panel_height,
            logo_panel_width,
            logo_panel_height,
        )
        self.logo_panel.background_color = gui.Color(0.3, 0.3, 0.3, 0)
        # ipdb.set_trace()

    def _on_close(self):
        self.stop_event.set()  # set before all
        time.sleep(0.1)
        self._calibration_event.set()  # need to clear loop
        self._collection_event.set()  # need to clear loop
        self._data_source.exit()
        time.sleep(4)

        _logger.info("Closing.")

        return True  # False would cancel the close

    def _update_thread(self):
        # This is NOT the UI thread, need to call post_to_main_thread() to update
        # the scene or any part of the UI.

        def update(data, result):
            try:
                self.warning_panel.visible = not self.notification_panel.visible and not result.is_confident

                if self._seg_event.is_set() and result.segmentation is not None:
                    rgb = self._seg_colors[result.segmentation]
                else:
                    rgb = data.rgb

                self.pcd.points = o3d.utility.Vector3dVector(data.points)
                self.pcd.colors = o3d.utility.Vector3dVector(rgb)

                # self.pcd.rotate(self.rot_mat)

                self.widget3d.scene.remove_geometry("pcd")
                self.widget3d.scene.add_geometry("pcd", self.pcd, self.lit)

                self.widget3d.scene.remove_geometry("ee_frame")
                self.widget3d.scene.remove_geometry("base_frame")
                self.ee_frame = None
                self.base_frame = None
                if self._toggle_pred.is_on:
                    if result.key_points_pose is not None:
                        self.ee_frame = create_coordinate_frame(result.key_points_pose, switch_w=False)

                    if result.key_points_base_pose is not None:
                        self.base_frame = create_coordinate_frame(result.key_points_base_pose, switch_w=False)
                else:
                    if result.ee_pose is not None:
                        self.ee_frame = create_coordinate_frame(result.ee_pose, switch_w=False)

                    if result.base_pose is not None:
                        self.base_frame = create_coordinate_frame(result.base_pose, switch_w=False)

                if self.ee_frame is not None:
                    self.widget3d.scene.add_geometry("ee_frame", self.ee_frame, self.lit)
                    self.widget3d.scene.show_geometry("ee_frame", self._instant_pred_check.checked)

                if self.base_frame is not None and not self._calibrated_pred_check.checked:
                    self.widget3d.scene.add_geometry("base_frame", self.base_frame, self.lit)
                    self.widget3d.scene.show_geometry("base_frame", self._instant_pred_check.checked)

                self.widget3d.scene.remove_geometry("key_points")
                if result.key_points is not None and len(result.key_points) > 0:
                    self.widget3d.scene.add_geometry(
                        "key_points",
                        generate_key_point_shapes(result.key_points, radius=0.008),
                        self.lit
                    )
                    self.widget3d.scene.show_geometry("key_points", self._kp_check.checked)
            except:  # noqa
                _logger.exception("GUI update failed.")

        while not self.stop_event.is_set():
            l_start = time.time()
            data = self._data_source.get()
            # print(data.points.shape)
            # print(data.rgb.shape)

            if data is not None:
                result = self._inference_engine.predict(data)

                # save results for calibration
                if self._collection_event.is_set():
                    try:
                        self._calibration_queue.put_nowait(result)
                    except queue.Full:
                        _logger.exception("Calibration queue is full.")

                if not self.stop_event.is_set():
                    # Update the images. This must be done on the UI thread.
                    gui.Application.instance.post_to_main_thread(
                        self.window, lambda: update(data, result)
                    )
            l_end = time.time()
            duration = l_end - l_start
            # print(f"FPS: {(1/(duration)):.2f}")
            time.sleep(max(0.8 - duration, 0.05))



def main():
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    app.add_font(gui.FontDescription(style=gui.FontStyle.BOLD_ITALIC, point_size=22))

    win = MainApp(_config.INFERENCE.data_source)

    app.run()


if __name__ == "__main__":
    main()
