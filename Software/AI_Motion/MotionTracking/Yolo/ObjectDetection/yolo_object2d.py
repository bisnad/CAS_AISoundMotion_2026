import cv2
import argparse
import numpy as np
import time
import sys
import os
import shutil
import threading
import torch
from ultralytics import YOLOWorld

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QSizePolicy,
    QLineEdit, QDoubleSpinBox, QSpinBox
)
from PyQt5.QtCore import QThread, pyqtSlot, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

import motion_sender

# ==============================================================================
# 1. Settings
# ==============================================================================

DEFAULT_OSC_IP = "127.0.0.1"
DEFAULT_OSC_PORT = 9007
classes_list = None


# ==============================================================================
# 2. Latest-frame reader for live cameras
# ==============================================================================

class LatestFrameReader:
    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.running = True
        self.latest_frame = None
        self.latest_id = 0
        self.read_fail = False
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            success, frame = self.cap.read()
            if not success:
                with self.lock:
                    self.read_fail = True
                    self.cond.notify_all()
                break

            with self.lock:
                self.latest_frame = frame
                self.latest_id += 1
                self.cond.notify_all()

    def get_latest(self, last_id=None, block=True, timeout=1.0):
        with self.lock:
            if block:
                end_time = time.perf_counter() + timeout
                while self.running and not self.read_fail:
                    if self.latest_frame is not None and (last_id is None or self.latest_id != last_id):
                        break
                    remaining = end_time - time.perf_counter()
                    if remaining <= 0:
                        return False, None, last_id
                    self.cond.wait(timeout=remaining)

            if self.latest_frame is None:
                return False, None, last_id

            if last_id is not None and self.latest_id == last_id:
                return False, None, last_id

            return True, self.latest_frame.copy(), self.latest_id

    def stop(self):
        self.running = False
        with self.lock:
            self.cond.notify_all()
        self.thread.join(timeout=1.0)


# ==============================================================================
# 3. THREADING & INFERENCE
# ==============================================================================

class VideoThread(QThread):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._run_flag = True

        self._new_classes = None
        self._new_osc_settings = None
        self.conf_threshold = args.conf

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"Using {self.device.upper()} device")

        model_name = f"yolov8{self.args.model}-worldv2.pt"
        model_dir = "models"
        model_path = os.path.join(model_dir, model_name)

        os.makedirs(model_dir, exist_ok=True)

        if self.args.local_model:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"--local_model flag used, but {model_path} not found.")
            print(f"Loading {model_path}...")
            self.model = YOLOWorld(model_path).to(self.device)
        else:
            if not os.path.exists(model_path):
                print(f"Model not found in {model_dir}/. Downloading {model_name}...")
                self.model = YOLOWorld(model_name).to(self.device)
                if os.path.exists(model_name):
                    shutil.move(model_name, model_path)
                    print(f"Successfully moved {model_name} into {model_dir}/ directory.")
            else:
                print(f"Loading existing {model_path}...")
                self.model = YOLOWorld(model_path).to(self.device)

        motion_sender.config["ip"] = DEFAULT_OSC_IP
        motion_sender.config["port"] = DEFAULT_OSC_PORT
        self.osc_sender = motion_sender.OscSender(motion_sender.config)

        if classes_list is not None:
            self.update_classes(classes_list)

        self.max_visual_fps = max(1.0, float(args.display_fps))
        self.frame_mode = args.frame_mode

        self._frame_lock = threading.Lock()
        self._stats_lock = threading.Lock()

        self._latest_frame_rgb = None
        self._latest_frame_seq = 0
        self._processing_fps = 0.0
        self._source_fps = 0.0
        self._skipped_frames = 0

    def update_classes(self, new_classes_list):
        self._new_classes = new_classes_list

    def update_osc_settings(self, ip, port):
        self._new_osc_settings = (ip, port)

    def set_conf_threshold(self, conf):
        self.conf_threshold = conf

    def get_latest_frame(self, last_seq=-1):
        with self._frame_lock:
            if self._latest_frame_rgb is None or self._latest_frame_seq == last_seq:
                return None, last_seq
            return self._latest_frame_rgb.copy(), self._latest_frame_seq

    def get_stats(self):
        with self._stats_lock:
            return self._processing_fps, self._source_fps, self._skipped_frames

    def _store_latest_frame(self, frame_rgb):
        with self._frame_lock:
            self._latest_frame_rgb = frame_rgb.copy()
            self._latest_frame_seq += 1

    def _set_stats(self, processing_fps=None, source_fps=None, skipped_inc=0):
        with self._stats_lock:
            if processing_fps is not None:
                self._processing_fps = processing_fps
            if source_fps is not None:
                self._source_fps = source_fps
            self._skipped_frames += skipped_inc

    def _process_frame(self, frame):
        with torch.no_grad():
            results = self.model(frame, verbose=False, conf=self.conf_threshold)[0]

        boxes = results.boxes
        if boxes is not None and len(boxes) > 0:
            class_names = results.names
            boxes_cls = boxes.cls.cpu().numpy()
            boxes_conf = boxes.conf.cpu().numpy()
            boxes_xyxyn = boxes.xyxyn.cpu().numpy()

            for box_index in range(boxes_xyxyn.shape[0]):
                osc_parameters = np.concatenate(
                    (
                        boxes_cls[box_index:box_index + 1],
                        boxes_conf[box_index:box_index + 1],
                        boxes_xyxyn[box_index]
                    ),
                    axis=0
                )
                self.osc_sender.send(f"/object/box/{box_index}/bbox", osc_parameters)

                class_id = int(boxes_cls[box_index])
                osc_class_name = class_names[class_id]
                self.osc_sender.send(f"/object/box/{box_index}/class", osc_class_name)

        return results

    def run(self):
        source = int(self.args.source) if self.args.source.isdigit() else self.args.source
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"Failed to open source: {self.args.source}")
            return

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        is_file = not str(self.args.source).isdigit()
        source_fps = 0.0

        if is_file:
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            if not source_fps or source_fps <= 1e-6:
                source_fps = 30.0
        else:
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            if not source_fps or source_fps <= 1e-6:
                source_fps = 0.0

        self._set_stats(source_fps=source_fps)

        frame_period = 1.0 / source_fps if source_fps > 1e-6 else None
        next_frame_time = time.perf_counter() if (is_file and frame_period is not None) else None

        prev_loop_time = time.perf_counter()
        fps_filter = 0.0
        last_visual_time = 0.0
        visual_period = 1.0 / self.max_visual_fps

        latest_reader = None
        last_camera_frame_id = None

        if (not is_file) and self.frame_mode == "realtime":
            latest_reader = LatestFrameReader(cap)

        try:
            while self._run_flag:
                if self._new_classes is not None:
                    print(f"Updating YOLO-World classes to: {self._new_classes}")
                    self.model.set_classes(self._new_classes)
                    self._new_classes = None

                if self._new_osc_settings is not None:
                    new_ip, new_port = self._new_osc_settings
                    print(f"Updating OSC Sender to {new_ip}:{new_port}")
                    motion_sender.config["ip"] = new_ip
                    motion_sender.config["port"] = new_port
                    self.osc_sender = motion_sender.OscSender(motion_sender.config)
                    self._new_osc_settings = None

                if is_file:
                    success, frame = cap.read()
                    if not success:
                        break
                else:
                    if self.frame_mode == "all":
                        success, frame = cap.read()
                        if not success:
                            break
                    else:
                        success, frame, new_id = latest_reader.get_latest(
                            last_id=last_camera_frame_id,
                            block=True,
                            timeout=1.0
                        )
                        if not success:
                            if latest_reader.read_fail:
                                break
                            else:
                                continue

                        if last_camera_frame_id is not None and new_id > last_camera_frame_id + 1:
                            self._set_stats(skipped_inc=(new_id - last_camera_frame_id - 1))
                        last_camera_frame_id = new_id

                results = self._process_frame(frame)

                now = time.perf_counter()
                if now - last_visual_time >= visual_period:
                    annotated_frame = results.plot()
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    self._store_latest_frame(annotated_frame_rgb)
                    last_visual_time = now

                loop_end = time.perf_counter()
                inst_fps = 1.0 / max(loop_end - prev_loop_time, 1e-6)
                prev_loop_time = loop_end
                fps_filter = inst_fps if fps_filter == 0.0 else (0.9 * fps_filter + 0.1 * inst_fps)
                self._set_stats(processing_fps=fps_filter)

                if is_file and frame_period is not None:
                    if self.frame_mode == "all":
                        pass
                    else:
                        next_frame_time += frame_period
                        delay = next_frame_time - time.perf_counter()

                        if delay > 0:
                            time.sleep(delay)
                        else:
                            frames_behind = int((-delay) / frame_period)
                            skipped_now = 0

                            for _ in range(frames_behind):
                                if cap.grab():
                                    skipped_now += 1
                                    next_frame_time += frame_period
                                else:
                                    break

                            if skipped_now > 0:
                                self._set_stats(skipped_inc=skipped_now)

                            if next_frame_time < time.perf_counter() - frame_period:
                                next_frame_time = time.perf_counter()
        finally:
            if latest_reader is not None:
                latest_reader.stop()
            cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


# ==============================================================================
# 4. GUI SETUP
# ==============================================================================

class MainWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("YOLO-World Custom Object Detection")
        self.resize(900, 680)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("background-color: black;")
        self.layout.addWidget(self.image_label, stretch=1)

        self.class_layout = QHBoxLayout()
        self.class_layout.setContentsMargins(10, 10, 10, 0)

        self.class_layout.addWidget(QLabel("Custom Classes (comma-separated):"))
        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("e.g. dancer, microphone, water bottle")
        self.class_layout.addWidget(self.class_input)

        self.btn_update_classes = QPushButton("Set Classes")
        self.btn_update_classes.clicked.connect(self.on_update_classes)
        self.class_layout.addWidget(self.btn_update_classes)

        self.layout.addLayout(self.class_layout)

        self.settings_layout = QHBoxLayout()
        self.settings_layout.setContentsMargins(10, 5, 10, 0)

        self.settings_layout.addWidget(QLabel("Confidence:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setValue(args.conf)
        self.conf_spin.valueChanged.connect(self.on_conf_changed)
        self.settings_layout.addWidget(self.conf_spin)

        self.settings_layout.addSpacing(20)

        self.settings_layout.addWidget(QLabel("OSC IP:"))
        self.ip_input = QLineEdit(DEFAULT_OSC_IP)
        self.ip_input.setFixedWidth(120)
        self.settings_layout.addWidget(self.ip_input)

        self.settings_layout.addWidget(QLabel("Port:"))
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1024, 65535)
        self.port_spin.setValue(DEFAULT_OSC_PORT)
        self.port_spin.setFixedWidth(80)
        self.settings_layout.addWidget(self.port_spin)

        self.btn_apply_osc = QPushButton("Apply OSC")
        self.btn_apply_osc.clicked.connect(self.on_apply_osc)
        self.settings_layout.addWidget(self.btn_apply_osc)

        self.settings_layout.addStretch(1)
        self.layout.addLayout(self.settings_layout)

        self.controls_layout = QHBoxLayout()
        self.controls_layout.setContentsMargins(10, 5, 10, 10)

        self.btn_exit = QPushButton("Exit")
        self.btn_exit.clicked.connect(self.close)
        self.btn_exit.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.controls_layout.addWidget(self.btn_exit)

        self.controls_layout.addStretch(1)

        self.fps_label = QLabel("Proc FPS: 0.0 | View FPS: 0.0 | Src FPS: 0.0 | Skipped: 0")
        self.fps_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.controls_layout.addWidget(self.fps_label)

        self.layout.addLayout(self.controls_layout)

        self.thread = VideoThread(args)
        self.thread.start()

        self.last_frame_seq = -1
        self.display_fps = 0.0
        self.prev_display_time = None

        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(max(1, int(1000 / max(1.0, float(args.ui_fps)))))
        self.ui_timer.timeout.connect(self.refresh_ui)
        self.ui_timer.start()

    def closeEvent(self, event):
        self.ui_timer.stop()
        self.thread.stop()
        event.accept()

    @pyqtSlot()
    def refresh_ui(self):
        frame_rgb, seq = self.thread.get_latest_frame(self.last_frame_seq)

        if frame_rgb is not None:
            self.last_frame_seq = seq

            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

            pixmap = QPixmap.fromImage(q_img).scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.FastTransformation
            )
            self.image_label.setPixmap(pixmap)

            now = time.perf_counter()
            if self.prev_display_time is not None:
                inst_display_fps = 1.0 / max(now - self.prev_display_time, 1e-6)
                self.display_fps = (
                    inst_display_fps if self.display_fps == 0.0
                    else 0.9 * self.display_fps + 0.1 * inst_display_fps
                )
            self.prev_display_time = now

        proc_fps, src_fps, skipped = self.thread.get_stats()
        self.fps_label.setText(
            f"Proc FPS: {proc_fps:.1f} | View FPS: {self.display_fps:.1f} | "
            f"Src FPS: {src_fps:.1f} | Skipped: {skipped}"
        )

    def on_update_classes(self):
        text = self.class_input.text()
        if text.strip():
            new_classes = [c.strip() for c in text.split(",") if c.strip()]
            self.thread.update_classes(new_classes)

    def on_conf_changed(self, value):
        self.thread.set_conf_threshold(value)

    def on_apply_osc(self):
        ip = self.ip_input.text().strip()
        port = self.port_spin.value()
        self.thread.update_osc_settings(ip, port)


# ==============================================================================
# 5. Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="YOLO-World Custom Classes OSC Sender")
    parser.add_argument("--source", type=str, default="0",
                        help="Webcam index (e.g. 0) or path to video file")
    parser.add_argument("--model", type=str, choices=["s", "m", "l", "x"], default="l",
                        help="YOLO-World model size (s=small, m=medium, l=large, x=extra-large)")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="Confidence threshold for detections")
    parser.add_argument("--display_fps", type=float, default=20.0,
                        help="Maximum rate for generating annotated frames for display")
    parser.add_argument("--ui_fps", type=float, default=30.0,
                        help="GUI refresh rate used to pull and display the latest frame")
    parser.add_argument("--frame_mode", type=str, choices=["all", "realtime"], default="realtime",
                        help="all = process every frame in order; realtime = drop stale frames to stay current")
    parser.add_argument("--local_model", action="store_true",
                        help="Look for model in local 'models/' directory instead of auto-downloading")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = MainWindow(args)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()