# CrowdAnalyzer.py
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import random
import time
import torch
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import markdown
from datetime import datetime
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO
from pedpy import (
    TrajectoryData, plot_trajectories, WalkableArea, MeasurementArea,
    compute_classic_density, compute_individual_voronoi_polygons, compute_voronoi_density,
    Cutoff, SpeedCalculation, compute_individual_speed, compute_mean_speed_per_frame,
    compute_voronoi_speed, PEDPY_BLUE, PEDPY_GREY, PEDPY_ORANGE, PEDPY_RED
)
from groq import Groq
from dotenv import load_dotenv
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QIcon
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget,
    QFileDialog, QInputDialog, QLineEdit, QComboBox, QDialog, QFormLayout,
    QDialogButtonBox, QMessageBox, QFrame, QTabWidget, QTextEdit
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# -----------------------------
# Track, TrackState (unchanged)
# -----------------------------
class TrackState:
    NEW = 0
    TRACKED = 1
    LOST = 2

class Track:
    """
    Minimal Kalman-based track used by Hungarian assignment in this script.
    Keeps Kalman filter state and simple life-cycle counters.
    """
    def __init__(self, box, track_id):
        self.id = track_id
        self.state = TrackState.NEW
        self.age = 0
        self.hits = 0
        self.time_since_update = 0

        # Initialize Kalman
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        x, y = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        self.kf.x = np.array([[x], [y], [0], [0]])
        self.kf.F = np.array([[1,0,1,0],
                              [0,1,0,1],
                              [0,0,1,0],
                              [0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0],
                              [0,1,0,0]])
        self.kf.P *= 10
        self.kf.R *= 1
        self.kf.Q *= 0.1

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.kf.x[:2].flatten()

    def update(self, measurement):
        # measurement must be [x, y]
        self.kf.update(np.array(measurement).reshape(2,1))
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.NEW and self.hits >= 3:
            self.state = TrackState.TRACKED

    def mark_missed(self):
        self.time_since_update += 1
        if self.state == TrackState.TRACKED and self.time_since_update > 10:
            self.state = TrackState.LOST

# -----------------------------
# CrowdDensityEstimation class
# -----------------------------
class CrowdDensityEstimation:
    """
    Single class that:
      - runs YOLO track/inference
      - handles simple Hungarian + Kalman tracking
      - stores track history (world coords if homography set)
      - computes live current_count, unique_count, density
      - exposes process_frame(frame, frame_number) used by GUI loop
    """
    def __init__(self, model_path='yolo11x.pt', conf_threshold=0.3, iou_threshold=0.5, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[CrowdDensityEstimation] Using device: {self.device}")

        # Load model (Ultralytics YOLO)
        self.model = YOLO(model_path)  # model path can be .pt or preset name if available
        try:
            # move model to device if supported (Ultralytics manages models internally)
            self.model.to(self.device)
        except Exception:
            pass

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Tracking containers
        self.tracks = []              # list of Track objects (Kalman)
        self.next_id = 0              # used to create new track IDs
        self.track_history = defaultdict(list)  # track_id -> list of (x_world, y_world)
        self.track_colors = {}        # track_id -> color tuple
        self.unique_persons = set()   # set of all seen track IDs

        # Homography and measurement area
        self.homography_matrix = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.measurement_area = None  # polygon in world coords (list of (x,y))

        # density history
        self.density_history = []

    # ---------- setters ----------
    def set_homography_matrix(self, points_image_or_matrix, points_world=None):
        """
        Use either:
         - set_homography_matrix(matrix)  -> set directly
         - set_homography_matrix(points_image_list, points_world_list) -> compute with cv2.findHomography
        """
        if points_world is None and isinstance(points_image_or_matrix, np.ndarray):
            # direct set
            self.homography_matrix = points_image_or_matrix
            return

        # otherwise compute from correspondences
        points_image = np.float32(points_image_or_matrix)
        points_world = np.float32(points_world)
        H, _ = cv2.findHomography(points_image, points_world)
        self.homography_matrix = H

    def set_camera_calibration(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def set_measurement_area(self, polygon_world):
        """
        polygon_world: list of (x,y) coordinates describing measurement area in world units (meters)
        """
        self.measurement_area = polygon_world

    # ---------- utilities ----------
    def get_color(self, track_id):
        if track_id not in self.track_colors:
            self.track_colors[track_id] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
        return self.track_colors[track_id]

    def transform_point(self, point):
        """
        image point (x_pixel, y_pixel) -> world (x, y) using homography.
        If homography not set, returns input point (as floats).
        """
        if self.homography_matrix is None:
            return float(point[0]), float(point[1])
        pt_h = np.array([point[0], point[1], 1.0]).reshape(3, 1)
        trans = self.homography_matrix @ pt_h
        trans = trans / trans[2]
        return float(trans[0][0]), float(trans[1][0])

    def world_to_image(self, point, H_inv):
        pt_h = np.array([point[0], point[1], 1.0]).reshape(3, 1)
        trans = H_inv @ pt_h
        trans = trans / trans[2]
        return float(trans[0][0]), float(trans[1][0])

    def polygon_area(self, polygon):
        """
        Shoelace formula for polygon area.
        polygon: list of (x,y) in world coords
        returns absolute area (float)
        """
        if polygon is None or len(polygon) < 3:
            return None
        arr = np.array(polygon)
        x = arr[:,0]
        y = arr[:,1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    # ---------- model + detection ----------
    def extract_tracks(self, frame):
        """
        Run YOLO track on the provided frame.
        Accepts numpy BGR frame (OpenCV).
        Returns results object and resized_frame (which we used for drawing).
        """
        # Resize to nearest multiple of 32 for YOLO
        h, w, _ = frame.shape
        new_w = (w // 32) * 32
        new_h = (h // 32) * 32
        if new_w == 0 or new_h == 0:
            new_w, new_h = w, h
        resized = cv2.resize(frame, (new_w, new_h))

        # Ultralytics accepts np.ndarray directly (they'll convert internally)
        # Avoid extra tensor conversion each frame for speed if possible
        try:
            results = self.model.track(
                resized,
                persist=True,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=[0],    # only person class
                tracker="bytetrack.yaml",
                augment=False,
                verbose=False
            )
        except Exception:
            # fallback: try calling with explicit tensor conversion
            tensor_frame = torch.tensor(resized).permute(2,0,1).unsqueeze(0).float().to(self.device) / 255.0
            results = self.model.track(
                tensor_frame,
                persist=True,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=[0],
                tracker="bytetrack.yaml",
                augment=False,
                verbose=False
            )
        return results, resized

    # ---------- converting results ----------
    def _detections_from_results(self, results):
        """
        Convert results to a list of numpy boxes [[x1,y1,x2,y2], ...]
        Handles empty cases safely.
        """
        detections = []
        try:
            if results and len(results) > 0 and getattr(results[0].boxes, "xyxy", None) is not None:
                boxes_arr = results[0].boxes.xyxy.cpu().numpy()
                for b in boxes_arr:
                    detections.append(b)  # already x1,y1,x2,y2
        except Exception:
            pass
        return detections

    # ---------- track management ----------
    def _hungarian_match(self, cost_matrix):
        """
        Hungarian matching wrapper that returns (matched, unmatched_dets, unmatched_tracks)
        """
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        matched = []
        unmatched_dets = []
        unmatched_trks = []
        for r, c in zip(row_idx, col_idx):
            if cost_matrix[r, c] > 50.0:  # distance threshold (pixels)
                unmatched_trks.append(c)
                unmatched_dets.append(r)
            else:
                matched.append((r, c))
        # tracks not in col_idx are unmatched, detections not in row_idx are unmatched
        for i in range(cost_matrix.shape[0]):
            if i not in row_idx:
                unmatched_dets.append(i)
        for j in range(cost_matrix.shape[1]):
            if j not in col_idx:
                unmatched_trks.append(j)
        return matched, unmatched_dets, unmatched_trks

    def _initiate_track(self, detection):
        """
        detection: array-like [x1,y1,x2,y2]
        """
        self.tracks.append(Track(detection, self.next_id))
        self.next_id += 1

    def update_tracks(self, detections):
        """
        Main tracker update that accepts:
          - detections: list/ndarray of boxes [x1,y1,x2,y2]
        Uses Kalman prediction + Hungarian matching to associate and update tracks.
        Also updates track_history with world coords (smoothed).
        """
        # predict existing tracks
        for t in self.tracks:
            t.predict()

        if len(detections) == 0:
            # only predict and mark missed
            for t in self.tracks:
                t.mark_missed()
            # remove lost
            self.tracks = [t for t in self.tracks if t.state != TrackState.LOST]
            return

        # build centers
        det_centers = np.array([[(d[0]+d[2])/2.0, (d[1]+d[3])/2.0] for d in detections])
        if len(self.tracks) > 0:
            track_centers = np.array([t.kf.x[:2].flatten() for t in self.tracks])
        else:
            track_centers = np.zeros((0,2))

        if track_centers.shape[0] > 0:
            # compute cost
            cost_matrix = np.linalg.norm(det_centers[:, None, :] - track_centers[None, :, :], axis=2)
            matched, unmatched_det, unmatched_trk = self._hungarian_match(cost_matrix)

            # update matched
            for d_idx, t_idx in matched:
                self.tracks[t_idx].update(det_centers[d_idx])
                # update history for that track id (world coords)
                tid = self.tracks[t_idx].id
                world_pt = self.transform_point(tuple(det_centers[d_idx]))
                # smoothing with prev if exists
                if len(self.track_history[tid]) >= 1:
                    prev = self.track_history[tid][-1]
                    world_pt = ((world_pt[0] + prev[0]) / 2.0, (world_pt[1] + prev[1]) / 2.0)
                self.track_history[tid].append(world_pt)
                if len(self.track_history[tid]) > 50:
                    self.track_history[tid].pop(0)
                self.unique_persons.add(tid)

            # mark unmatched tracks missed
            for t_idx in unmatched_trk:
                if t_idx < len(self.tracks):
                    self.tracks[t_idx].mark_missed()

            # create new tracks for unmatched detections
            for d_idx in unmatched_det:
                det = detections[d_idx]
                self._initiate_track(det)
                new_tid = self.tracks[-1].id
                world_pt = self.transform_point(((det[0]+det[2])/2.0, (det[1]+det[3])/2.0))
                self.track_history[new_tid].append(world_pt)
                self.unique_persons.add(new_tid)

        else:
            # no existing tracks -> create new ones for all detections
            for det in detections:
                self._initiate_track(det)
                tid = self.tracks[-1].id
                world_pt = self.transform_point(((det[0]+det[2])/2.0, (det[1]+det[3])/2.0))
                self.track_history[tid].append(world_pt)
                self.unique_persons.add(tid)

        # cleanup lost tracks
        self.tracks = [t for t in self.tracks if t.state != TrackState.LOST]

    # ---------- trajectories update from YOLO results (ID aware) ----------
    def update_trajectories(self, results, frame_number):
        """
        Append world coordinates for tracked IDs coming directly from YOLO tracker.
        This keeps history in sync with YOLO's IDs if available.
        """
        try:
            if results and len(results) > 0 and getattr(results[0].boxes, "id", None) is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                for b, tid in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, b)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    world_center = self.transform_point(center)
                    self.track_history[tid].append(world_center)
                    if len(self.track_history[tid]) > 50:
                        self.track_history[tid].pop(0)
                    self.unique_persons.add(tid)
        except Exception:
            pass

    # ---------- draw detections + overlays ----------
    def draw_detections(self, frame, results):
        """
        Draw bounding boxes, IDs (if present), trajectories (converted to image coords),
        and overlay current_count, unique_count, density.
        """
        draw_frame = frame.copy()

        # Draw boxes and IDs if present
        current_count = 0
        try:
            if results and len(results) > 0 and getattr(results[0].boxes, "id", None) is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                current_count = len(ids)
                self.unique_persons.update(ids)
                for b, tid in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, b)
                    color = self.get_color(tid)
                    cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(draw_frame, f"ID:{tid}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except Exception:
            # fallback: no YOLO IDs available
            current_count = 0

        # Calculate density:
        density_value = 0.0
        if self.measurement_area is not None:
            area_m2 = self.polygon_area(self.measurement_area)
            if area_m2 and area_m2 > 0:
                density_value = current_count / area_m2
        else:
            # fallback to people per 10k pixels (for display only)
            h, w = draw_frame.shape[:2]
            area_pixels = max(1, h * w)
            density_value = current_count / (area_pixels / 10000.0)

        # Save density history and annotate
        self.density_history.append(density_value)

        # Overlays
        cv2.putText(draw_frame, f"Current Count: {current_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(draw_frame, f"Unique Count: {len(self.unique_persons)}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(draw_frame, f"Density: {density_value:.3f}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        return draw_frame

    # ---------- process_frame used by GUI loop ----------
    def process_frame(self, frame, frame_number):
        """
        Full pipeline called from the GUI MainWindow.update_frame.
        Returns: (processed_frame (BGR), density_info(dict))
        density_info keys: current_count, unique_count, density
        """
        results, resized = self.extract_tracks(frame)

        # get detections as simple arrays (for Kalman/Hungarian)
        detections = self._detections_from_results(results)

        # update track-level Kalman matching (works with detections list)
        self.update_tracks(detections)

        # update trajectories using YOLO ids if available
        self.update_trajectories(results, frame_number)

        # draw detections and overlays
        processed = self.draw_detections(resized, results)

        # prepare density info
        current_count = 0
        try:
            if results and len(results) > 0 and getattr(results[0].boxes, "id", None) is not None:
                current_count = len(results[0].boxes.id)
        except Exception:
            current_count = len(detections)

        unique_count = len(self.unique_persons)
        density_value = self.density_history[-1] if len(self.density_history) > 0 else 0.0

        density_info = {
            "frame_number": frame_number,
            "current_count": int(current_count),
            "unique_count": int(unique_count),
            "density": float(density_value)
        }
        return processed, density_info

    # ---------- save trajectories ----------
    def save_trajectories(self, output_path='trajectories.csv'):
        trajectory_data = []
        for pid, pts in self.track_history.items():
            for fidx, pos in enumerate(pts):
                trajectory_data.append({'id': int(pid), 'frame': int(fidx), 'x': float(pos[0]), 'y': float(pos[1])})
        if trajectory_data:
            df = pd.DataFrame(trajectory_data)
            df.to_csv(output_path, index=False, float_format='%.6f')
            print(f"[CrowdDensityEstimation] Trajectories saved to {output_path}")
        else:
            print("[CrowdDensityEstimation] No trajectories to save.")

# -----------------------------
# GUI classes (unchanged structure, small integration edits)
# -----------------------------
class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.layout = QFormLayout(self)
        self.layout.addRow(QLabel("<b>YOLO parameters</b>"))
        self.yolo_model_combo = QComboBox(self)
        self.yolo_model_combo.addItems(["yolo11x", "yolo11l", "yolo11m", "yolo11s", "yolo11n"])
        self.yolo_model_combo.setCurrentText(self.parent().settings.get('yolo_model', 'yolo11x'))
        self.layout.addRow("YOLO Model:", self.yolo_model_combo)
        self.track_model_combo = QComboBox(self)
        self.track_model_combo.addItems(["bytetrack.yaml", "botsort.yaml"])
        self.track_model_combo.setCurrentText(self.parent().settings.get('track_model', 'bytetrack.yaml'))
        self.layout.addRow("Track Model:", self.track_model_combo)
        self.output_folder_button = QPushButton("Choose Folder", self)
        self.output_folder_button.clicked.connect(self.choose_folder)
        self.layout.addRow("Output Folder:", self.output_folder_button)
        current_output_folder = self.parent().settings.get('output_folder', '')
        if current_output_folder:
            self.output_folder = current_output_folder
            self.output_folder_button.setText(f"Folder: {current_output_folder}")
        else:
            self.output_folder = None

        self.conf_threshold_input = QLineEdit(self)
        self.conf_threshold_input.setText(str(self.parent().settings.get('conf_threshold', 0.3)))
        self.layout.addRow("Confidence Threshold:", self.conf_threshold_input)
        self.iou_threshold_input = QLineEdit(self)
        self.iou_threshold_input.setText(str(self.parent().settings.get('iou_threshold', 0.5)))
        self.layout.addRow("IoU Threshold:", self.iou_threshold_input)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        self.layout.addRow(line)
        self.layout.addRow(QLabel("<b>PedPy parameters</b>"))

        self.frame_rate_input = QLineEdit(self)
        self.frame_rate_input.setText(str(self.parent().settings.get('frame_rate', 30)))
        self.layout.addRow("Frame Rate:", self.frame_rate_input)

        self.walkable_area_input = QLineEdit(self)
        walkable_area = self.parent().settings.get('walkable_area', None)
        if walkable_area is not None:
            walkable_area_text = ' '.join([f"{x},{y}" for x, y in walkable_area])
            self.walkable_area_input.setText(walkable_area_text)
        else:
            self.walkable_area_input.setText("NA")
        self.layout.addRow("Walkable Area (x1,y1 x2,y2 ...):", self.walkable_area_input)

        self.measurement_area_input = QLineEdit(self)
        measurement_area = self.parent().settings.get('measurement_area', None)
        if measurement_area is not None:
            measurement_area_text = ' '.join([f"{x},{y}" for x, y in measurement_area])
            self.measurement_area_input.setText(measurement_area_text)
        else:
            self.measurement_area_input.setText("NA")
        self.layout.addRow("Measurement Area (x1,y1 x2,y2 ...):", self.measurement_area_input)

        self.frame_step_input = QLineEdit(self)
        self.frame_step_input.setText(str(self.parent().settings.get('frame_step', 25)))
        self.layout.addRow("Frame Step:", self.frame_step_input)

        # Add frame skip (performance)
        self.frame_skip_input = QLineEdit(self)
        self.frame_skip_input.setText(str(self.parent().settings.get('frame_skip', 1)))
        self.layout.addRow("Frame Skip (1 = every frame):", self.frame_skip_input)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder = folder
            self.output_folder_button.setText(f"Folder: {self.output_folder}")

    def get_settings(self):
        walkable_area = self.parse_polygon(self.walkable_area_input.text())
        measurement_area = self.parse_polygon(self.measurement_area_input.text())
        if walkable_area is None or measurement_area is None:
            return None

        return {
            "yolo_model": self.yolo_model_combo.currentText(),
            "track_model": self.track_model_combo.currentText(),
            "output_folder": self.output_folder,
            "conf_threshold": float(self.conf_threshold_input.text()),
            "iou_threshold": float(self.iou_threshold_input.text()),
            "frame_rate": int(self.frame_rate_input.text()),
            "walkable_area": walkable_area,
            "measurement_area": measurement_area,
            "frame_step": int(self.frame_step_input.text()),
            "frame_skip": int(self.frame_skip_input.text())
        }

    def parse_polygon(self, text):
        if text == "NA":
            return None
        try:
            points = [tuple(map(float, point.split(','))) for point in text.strip().split()]
            if len(points) < 3:
                QMessageBox.critical(self, "Error", "A polygon must have at least 3 points.")
                return None
            return points
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Invalid polygon format: {e}")
            return None

# -----------------------------
# MainWindow (kept structure, integrate frame_skip and set measurement area)
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crowd Analyzer")
        self.setGeometry(100, 100, 800, 600)
        # icons
        icon_path = os.path.join(os.path.dirname(__file__), "img", "logo.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.layout = QVBoxLayout()

        self.video_label = QLabel("No video loaded", self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.video_label)

        self.load_button = QPushButton("Load Video", self)
        self.load_button.clicked.connect(self.load_video)
        self.layout.addWidget(self.load_button)

        self.start_button = QPushButton("Start Processing", self)
        self.start_button.clicked.connect(self.start_processing)
        self.layout.addWidget(self.start_button)

        self.settings_button = QPushButton("Settings", self)
        self.settings_button.clicked.connect(self.open_settings)
        self.layout.addWidget(self.settings_button)

        self.container = QWidget()
        self.container.setLayout(self.layout)
        self.setCentralWidget(self.container)

        self.video_path = ""
        self.settings = {
            "yolo_model": "yolo11x",
            "track_model": "bytetrack.yaml",
            "output_folder": "",
            "conf_threshold": 0.3,
            "iou_threshold": 0.5,
            "frame_rate": 30,
            "walkable_area": None,
            "measurement_area": None,
            "frame_step": 25,
            "frame_skip": 1
        }

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.cap = None
        self.estimator = None
        self.frame_number = 0
        self.video_writer = None
        self.output_path = None
        self.output_filename = None

    def load_video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if self.video_path:
            self.video_label.setText(f"Loaded video: {self.video_path}")
            if not self.settings["output_folder"]:
                self.settings["output_folder"] = os.path.dirname(self.video_path)

    def start_processing(self):
        if not self.video_path:
            self.video_label.setText("Please load a video first")
            return

        if not self.settings['output_folder']:
            self.settings['output_folder'] = os.path.dirname(self.video_path)

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.video_label.setText("Error opening video stream or file")
            return

        fps = self.cap.get(cv2.CAP_PROP_FPS) or self.settings.get('frame_rate', 30)
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.output_filename = f"{base_name}_processed_{timestamp}.mp4"
        self.output_path = os.path.join(self.settings['output_folder'], self.output_filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            fps,
            (frame_width, frame_height),
            isColor=True
        )

        if not self.video_writer.isOpened():
            self.video_label.setText("Failed to create video writer")
            return

        self.video_label.setText(f"Processing video... Output will be saved to: {self.output_path}")

        ret, frame = self.cap.read()
        if not ret:
            self.video_label.setText("Error reading first frame")
            return

        # Ask user to select 4 points and distances for perspective transform
        points_image, points_world = self.select_points_and_distances(frame)
        if points_image is None or points_world is None:
            self.video_label.setText("Need 4 points and distances for perspective transform")
            self.video_writer.release()
            return

        # Initialize estimator and set homography + measurement area if available
        self.estimator = CrowdDensityEstimation(
            model_path=f"{self.settings['yolo_model']}.pt",
            conf_threshold=self.settings["conf_threshold"],
            iou_threshold=self.settings["iou_threshold"]
        )
        # set homography using the chosen points
        self.estimator.set_homography_matrix(points_image, points_world)

        # If measurement area is defined in settings, pass to estimator
        if self.settings.get("measurement_area"):
            self.estimator.set_measurement_area(self.settings["measurement_area"])

        self.frame_number = 0
        # start a timer (choose interval small e.g., 1ms to let loop run fast and use frame_skip)
        self.timer.start(1)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.cap.release()
            if self.video_writer and self.video_writer.isOpened():
                self.video_writer.release()
                print(f"Video saved to: {self.output_path}")
                self.video_label.setText(f"Processing complete.\nVideo saved to: {self.output_path}")
            # save output artifacts
            if self.estimator:
                self.estimator.save_trajectories(os.path.join(self.settings['output_folder'], 'trajectories.csv'))
            self.calculate_default_areas()
            self.show_plots()
            return

        self.frame_number += 1

        # frame skip for performance
        frame_skip = int(self.settings.get('frame_skip', 1))
        if frame_skip <= 0:
            frame_skip = 1
        if (self.frame_number % frame_skip) != 0:
            # still write raw frame to video to maintain timing if desired
            self.video_writer.write(frame)
            # update UI label with original frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, 3*w, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qimg))
            return

        # process frame via estimator
        processed_frame, density_info = self.estimator.process_frame(frame, self.frame_number)

        # ensure processed frame size matches original
        if processed_frame.shape[:2] != (frame.shape[0], frame.shape[1]):
            processed_frame = cv2.resize(processed_frame, (frame.shape[1], frame.shape[0]))

        if len(processed_frame.shape) == 2:
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)

        # write and display
        self.video_writer.write(processed_frame)
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def open_settings(self):
        dialog = SettingsDialog(self)
        if dialog.exec():
            new_settings = dialog.get_settings()
            if new_settings:
                self.settings.update(new_settings)

    def select_points_and_distances(self, frame):
        selector = PointSelector("Select 4 points")
        selector.image = frame.copy()
        cv2.namedWindow(selector.window_name)
        cv2.setMouseCallback(selector.window_name, selector.mouse_callback)

        print("Select 4 points in clockwise order")
        cv2.imshow(selector.window_name, frame)

        while len(selector.points) < 4:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()

        if len(selector.points) == 4:
            distances = []
            for i in range(4):
                distance, ok = QInputDialog.getDouble(
                    self,
                    "Enter Distance",
                    f"Enter distance between point {i+1} and point {(i+1)%4 + 1} in meters:",
                    value=1.0,
                    min=0.01,
                    max=1000.0,
                    decimals=2
                )
                if not ok:
                    return None, None
                distances.append(distance)
            # build points_world (simple rectangle walk)
            points_world = [[0,0]]
            current_x, current_y = 0, 0
            for i in range(4):
                if i == 0:
                    current_x += distances[0]
                    points_world.append([current_x, current_y])
                elif i == 1:
                    current_y += distances[1]
                    points_world.append([current_x, current_y])
                elif i == 2:
                    current_x -= distances[2]
                    points_world.append([current_x, current_y])
            return selector.points, points_world
        return None, None

    def calculate_default_areas(self):
        trajectory_file = os.path.join(self.settings['output_folder'], 'trajectories.csv')
        if not os.path.exists(trajectory_file):
            return
        df = pd.read_csv(trajectory_file)
        margin = 0.1
        minx, miny = df[['x', 'y']].min().values
        maxx, maxy = df[['x', 'y']].max().values
        minx -= margin
        miny -= margin
        maxx += margin
        maxy += margin
        self.settings["walkable_area"] = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
        center_x = (minx + maxx) / 2
        center_y = (miny + maxy) / 2
        width = (maxx - minx) * 0.25
        height = (maxy - miny) * 0.25
        self.settings["measurement_area"] = [(center_x - width / 2, center_y - height / 2),
                                            (center_x + width / 2, center_y - height / 2),
                                            (center_x + width / 2, center_y + height / 2),
                                            (center_x - width / 2, center_y + height / 2)]

    def show_plots(self):
        trajectory_file = os.path.join(self.settings['output_folder'], 'trajectories.csv')
        if not os.path.exists(trajectory_file):
            QMessageBox.information(self, "No Trajectories", "No trajectories.csv found to plot.")
            return
        df = pd.read_csv(trajectory_file)
        frame_rate = self.settings["frame_rate"]
        traj = TrajectoryData(data=df, frame_rate=frame_rate)
        fig1, ax1 = plt.subplots()
        plot_trajectories(traj=traj, ax=ax1).set_aspect("equal")
        ax1.set_title("Pedestrian Trajectories")
        ax1.set_xlabel("X Position (m)")
        ax1.set_ylabel("Y Position (m)")
        trajectory_plot_path = os.path.join(self.settings['output_folder'], 'trajectories_plot.png')
        fig1.savefig(trajectory_plot_path)

        polygon = self.settings["walkable_area"]
        walkable_area = WalkableArea(polygon=polygon)
        measurement_area_polygon = self.settings["measurement_area"]
        measurement_area = MeasurementArea(measurement_area_polygon)
        classic_density = compute_classic_density(traj_data=traj, measurement_area=measurement_area)
        individual = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=walkable_area)
        density_voronoi, intersecting = compute_voronoi_density(individual_voronoi_data=individual, measurement_area=measurement_area)
        individual_cutoff = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=walkable_area, cut_off=Cutoff(radius=12.0, quad_segments=1))
        density_voronoi_cutoff, intersecting_cutoff = compute_voronoi_density(individual_voronoi_data=individual_cutoff, measurement_area=measurement_area)
        fig2, ax2 = plt.subplots()
        ax2.set_title("Comparison of different density methods")
        ax2.plot(classic_density.reset_index().frame, classic_density.values, label="classic", color=PEDPY_BLUE)
        ax2.plot(density_voronoi.reset_index().frame, density_voronoi, label="voronoi", color=PEDPY_ORANGE)
        ax2.plot(density_voronoi_cutoff.reset_index().frame, density_voronoi_cutoff, label="voronoi with cutoff", color=PEDPY_GREY)
        ax2.set_xlabel("frame")
        ax2.set_ylabel("$\\rho$ / 1/$m^2$")
        ax2.grid()
        ax2.legend()
        density_plot_path = os.path.join(self.settings['output_folder'], 'density_plot.png')
        fig2.savefig(density_plot_path)

        frame_step = self.settings["frame_step"]
        individual_speed_single_sided = compute_individual_speed(traj_data=traj, frame_step=frame_step, compute_velocity=True, speed_calculation=SpeedCalculation.BORDER_SINGLE_SIDED)
        mean_speed = compute_mean_speed_per_frame(traj_data=traj, measurement_area=measurement_area, individual_speed=individual_speed_single_sided)
        individual_speed_direction = compute_individual_speed(traj_data=traj, frame_step=5, movement_direction=np.array([0, -1]), compute_velocity=True, speed_calculation=SpeedCalculation.BORDER_SINGLE_SIDED)
        mean_speed_direction = compute_mean_speed_per_frame(traj_data=traj, measurement_area=measurement_area, individual_speed=individual_speed_direction)
        voronoi_speed = compute_voronoi_speed(traj_data=traj, individual_voronoi_intersection=intersecting, individual_speed=individual_speed_single_sided, measurement_area=measurement_area)
        voronoi_speed_direction = compute_voronoi_speed(traj_data=traj, individual_voronoi_intersection=intersecting, individual_speed=individual_speed_direction, measurement_area=measurement_area)
        fig3, ax3 = plt.subplots()
        ax3.set_title("Comparison of different speed methods")
        ax3.plot(voronoi_speed.reset_index().frame, voronoi_speed, label="Voronoi", color=PEDPY_ORANGE)
        ax3.plot(voronoi_speed_direction.reset_index().frame, voronoi_speed_direction, label="Voronoi direction", color=PEDPY_GREY)
        ax3.plot(mean_speed.reset_index().frame, mean_speed, label="classic", color=PEDPY_BLUE)
        ax3.plot(mean_speed_direction.reset_index().frame, mean_speed_direction, label="classic direction", color=PEDPY_RED)
        ax3.set_xlabel("frame")
        ax3.set_ylabel("v / m/s")
        ax3.legend()
        ax3.grid()
        speed_plot_path = os.path.join(self.settings['output_folder'], 'speed_plot.png')
        fig3.savefig(speed_plot_path)

        plot_window = PlotWindow(density_plot_path, speed_plot_path, trajectory_plot_path, self)
        plot_window.show()

# -----------------------------
# PointSelector and PlotWindow (unchanged)
# -----------------------------
class PointSelector:
    def __init__(self, window_name):
        self.points = []
        self.window_name = window_name
        self.image = None
        self.distances = []

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append([x, y])
            cv2.circle(self.image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(self.image, str(len(self.points)), (x+10, y+10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if len(self.points) > 1:
                cv2.line(self.image, tuple(self.points[-2]), tuple(self.points[-1]), (0,255,0), 2)
            if len(self.points) == 4:
                cv2.line(self.image, tuple(self.points[-1]), tuple(self.points[0]), (0,255,0), 2)
            cv2.imshow(self.window_name, self.image)

class PlotWindow(QMainWindow):
    def __init__(self, density_plot_path, speed_plot_path, trajectory_plot_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plot Interpretations")
        self.setGeometry(100, 100, 800, 600)

        self.tab_widget = QTabWidget(self)
        self.setCentralWidget(self.tab_widget)

        # Create tabs
        self.density_tab = QWidget()
        self.speed_tab = QWidget()
        self.trajectory_tab = QWidget()

        self.tab_widget.addTab(self.density_tab, "Density Plot")
        self.tab_widget.addTab(self.speed_tab, "Speed Plot")
        self.tab_widget.addTab(self.trajectory_tab, "Trajectory Plot")

        # Set layouts for tabs
        self.density_layout = QVBoxLayout(self.density_tab)
        self.speed_layout = QVBoxLayout(self.speed_tab)
        self.trajectory_layout = QVBoxLayout(self.trajectory_tab)

        # Add plot images
        self.density_image = QLabel(self)
        self.density_image.setPixmap(QPixmap(density_plot_path))
        self.density_layout.addWidget(self.density_image)

        self.speed_image = QLabel(self)
        self.speed_image.setPixmap(QPixmap(speed_plot_path))
        self.speed_layout.addWidget(self.speed_image)

        self.trajectory_image = QLabel(self)
        self.trajectory_image.setPixmap(QPixmap(trajectory_plot_path))
        self.trajectory_layout.addWidget(self.trajectory_image)

        # Add interpretation text
        self.density_text = QTextEdit(self)
        self.density_text.setReadOnly(True)
        self.density_layout.addWidget(self.density_text)

        self.speed_text = QTextEdit(self)
        self.speed_text.setReadOnly(True)
        self.speed_layout.addWidget(self.speed_text)

        # Generate interpretations (keeps your Groq logic)
        self.generate_interpretation(density_plot_path, self.density_text, "density")
        self.generate_interpretation(speed_plot_path, self.speed_text, "speed")

    def generate_interpretation(self, image_path, text_widget, plot_type):
        base64_image = self.encode_image(image_path)
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        client = Groq(api_key=api_key)

        if plot_type == "density":
            content = """
                You are a data scientist specializing in pedestrian and crowd mobility patterns. Your task is to analyze and interpret the attached pedestrian density plot, comparing the three density estimation methods (classic, Voronoi, and Voronoi with cutoff). Discuss their differences, and trends over frames in a scientific manner.
            """
        else:
            content = """
                You are a data scientist specializing in pedestrian and crowd mobility patterns. Your task is to analyze and interpret the attached pedestrian speed plot, comparing the four speed estimation methods (classic, classic direction, Voronoi, and Voronoi direction). Discuss their differences, and trends over frames in a scientific manner.
            """

        # call Groq (may require network and API key in .env)
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": content},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                model="llama-3.2-90b-vision-preview",
            )
            html_content = markdown.markdown(chat_completion.choices[0].message.content)
            text_widget.setHtml(html_content)
        except Exception as e:
            text_widget.setPlainText(f"Failed to generate interpretation: {e}")

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


    def save_trajectories(self, path="trajectories.csv"):
        df = pd.DataFrame([{"frame":k,"count":v} for k,v in self.track_history.items()])
        df.to_csv(path,index=False)
        print(f"Trajectories saved: {path}")

    def generate_plots(self):
        counts = list(self.track_history.values())
        frames = list(self.track_history.keys())

        # People count
        plt.figure()
        plt.plot(frames, counts, label="People Count")
        plt.xlabel("Frame")
        plt.ylabel("Number of People")
        plt.title("People Count Over Time")
        plt.savefig("people_count.png")
        plt.close()

        # Density over time (same as count normalized)
        density = np.array(counts)/max(counts) if max(counts)>0 else np.array(counts)
        plt.figure()
        plt.plot(frames, density, label="Density")
        plt.xlabel("Frame")
        plt.ylabel("Density (normalized)")
        plt.title("Density Over Time")
        plt.savefig("density_over_time.png")
        plt.close()

        # Unique people over time
        cumulative = np.cumsum([1 if v>0 else 0 for v in counts])
        plt.figure()
        plt.plot(frames, cumulative, label="Unique People", color='orange')
        plt.xlabel("Frame")
        plt.ylabel("Unique People")
        plt.title("Unique People Over Time")
        plt.savefig("unique_people.png")
        plt.close()

        # Movement heatmap
        if self.movement_map is not None:
            plt.figure(figsize=(10,6))
            plt.imshow(self.movement_map, cmap='hot')
            plt.title("Movement Heatmap")
            plt.axis('off')
            plt.colorbar(label="Visits")
            plt.savefig("movement_heatmap.png")
            plt.close()

        print("Plots saved: people_count.png, density_over_time.png, unique_people.png, movement_heatmap.png")
# -----------------------------
# Run application
# -----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
