"""
camera_tracker.py - Holistic Body, Pose, Hand & Face-ID Recognition Tracker for A.R.C
Integrates:
- MediaPipe Pose, Hand & Face Mesh (468 dots) tracking
- dlib / face_recognition 128D embedding face identifier
- Persistent storage for known faces (known_faces.pkl)
- In-frame Settings panel with Layers control, Auto-Registration, and Import/Export
"""

import os
import time
import pickle
import numpy as np
import cv2

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPoint
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QInputDialog, QFileDialog, QMessageBox
)

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    face_recognition = None
    FACE_REC_AVAILABLE = False


# ----------------- Helper Connections for Drawing ----------------- #
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
    (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
    (29, 31), (30, 32), (27, 31), (28, 32)
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

def draw_landmarks(frame, landmarks, color=(0, 255, 0), radius=2):
    h, w, _ = frame.shape
    coords = []
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        coords.append((cx, cy))
        cv2.circle(frame, (cx, cy), radius, color, -1)
    return coords

def draw_connections(frame, coords, connections, color=(255, 255, 255), thickness=1):
    for start_idx, end_idx in connections:
        if start_idx < len(coords) and end_idx < len(coords):
            cv2.line(frame, coords[start_idx], coords[end_idx], color, thickness)


# ----------------- Camera Tracker Widget ----------------- #
class CameraTrackerWidget(QWidget):
    req_register_signal = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)

        self.db_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "known_faces.pkl")
        self.known_face_encodings = []
        self.known_face_names = []
        self._load_database()

        # UI & State Controls
        self.settings_open = False
        self.auto_register_toggle = False
        self.is_prompt_open = False
        self.unknown_prompt_cooldown = 0
        self.last_detected_encodings = []

        self.layers = {
            "Face Mesh": True,
            "Body Pose": True,
            "Hands": True,
            "Face Recognition": True,
            "FPS Counter": True
        }

        # On-screen UI layout coordinates (based on 1280x720 internal canvas)
        self.SETTINGS_BTN = (1120, 20, 1260, 60)
        self.SETTING_ITEMS = {
            "Auto-Register": (860, 85, 1240, 125),
            "Register New Face": (860, 135, 1240, 175),
            "Export Data": (860, 185, 1240, 225),
            "Import Data": (860, 235, 1240, 275),
            "Layer: Face Mesh": (860, 335, 1240, 370),
            "Layer: Body Pose": (860, 380, 1240, 415),
            "Layer: Hands": (860, 425, 1240, 460),
            "Layer: Face Recognition": (860, 470, 1240, 505),
            "Layer: FPS Counter": (860, 515, 1240, 550),
        }

        # MediaPipe Inferences Setup
        self._init_mediapipe_models()

        # Video Streaming Engine
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._process_frame)

        self.prev_time = 0
        self.frame_count = 0
        self.face_locations = []
        self.face_names = []

        # Build Widget UI
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        self.video_view = QLabel("Camera inactive. Open via Window -> Camera.")
        self.video_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_view.setStyleSheet("""
            background-color: #030712;
            border: 1px solid #143558;
            border-radius: 8px;
            color: #4a78a8;
            font-family: 'Segoe UI';
            font-size: 13px;
        """)
        lay.addWidget(self.video_view, 1)

        # Connect Qt registration event signal
        self.req_register_signal.connect(self._prompt_register_face)

    # ----------------- Database Persistence ----------------- #
    def _load_database(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, "rb") as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data.get("encodings", [])
                    self.known_face_names = data.get("names", [])
            except Exception as e:
                print(f"[ERROR] Could not load face database: {e}")

    def _save_database(self):
        try:
            with open(self.db_file, "wb") as f:
                pickle.dump({
                    "encodings": self.known_face_encodings,
                    "names": self.known_face_names
                }, f)
        except Exception as e:
            print(f"[ERROR] Could not save face database: {e}")

    # ----------------- MediaPipe Setup ----------------- #
    def _init_mediapipe_models(self):
        self.pose_landmarker = None
        self.face_landmarker = None
        self.hand_landmarker = None

        pose_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker.task")
        face_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
        hand_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

        try:
            if os.path.exists(pose_path):
                po = vision.PoseLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=pose_path),
                    running_mode=vision.RunningMode.VIDEO
                )
                self.pose_landmarker = vision.PoseLandmarker.create_from_options(po)

            if os.path.exists(face_path):
                fo = vision.FaceLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=face_path),
                    running_mode=vision.RunningMode.VIDEO,
                    num_faces=4
                )
                self.face_landmarker = vision.FaceLandmarker.create_from_options(fo)

            if os.path.exists(hand_path):
                ho = vision.HandLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=hand_path),
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=4
                )
                self.hand_landmarker = vision.HandLandmarker.create_from_options(ho)
        except Exception as e:
            print(f"[WARN] MediaPipe Task Init Error: {e}")

    # ----------------- Stream Lifecycle ----------------- #
    def start_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.timer.isActive():
            self.timer.start(30)

    def stop_camera(self):
        self.timer.stop()
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        self.video_view.setText("Camera stopped.")

    # ----------------- Frame Processing Pipeline ----------------- #
    def _process_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return

        success, frame = self.cap.read()
        if not success:
            return

        frame = cv2.flip(frame, 1)
        # Ensure working dimension matches UI expectations
        frame = cv2.resize(frame, (1280, 720))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frame_count += 1

        # 1. Real-time Face Recognition Inference (Every 4 frames at 0.25x scale)
        if FACE_REC_AVAILABLE and (self.frame_count % 4 == 0):
            small = cv2.resize(frame_rgb, (0, 0), fx=0.25, fy=0.25)
            self.face_locations = face_recognition.face_locations(small, model="hog")
            self.last_detected_encodings = face_recognition.face_encodings(small, self.face_locations)

            self.face_names = []
            for encoding in self.last_detected_encodings:
                name = "Unknown"
                if len(self.known_face_encodings) > 0:
                    distances = face_recognition.face_distance(self.known_face_encodings, encoding)
                    best_idx = np.argmin(distances)
                    if distances[best_idx] < 0.55:
                        name = self.known_face_names[best_idx]

                self.face_names.append(name)

                # Auto-registration modal trigger
                if name == "Unknown" and self.auto_register_toggle and not self.is_prompt_open:
                    if time.time() - self.unknown_prompt_cooldown > 4.0:
                        self.unknown_prompt_cooldown = time.time()
                        self.req_register_signal.emit(encoding)

        # 2. MediaPipe Holistic Inferences
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(time.time() * 1000)

        pose_result = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms) if self.pose_landmarker else None
        face_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms) if self.face_landmarker else None
        hand_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms) if self.hand_landmarker else None

        # --- Layer: Body Pose ---
        if self.layers["Body Pose"] and pose_result and pose_result.pose_landmarks:
            for p_lms in pose_result.pose_landmarks:
                coords = draw_landmarks(frame, p_lms, color=(0, 255, 255), radius=3)
                draw_connections(frame, coords, POSE_CONNECTIONS, color=(0, 180, 255), thickness=2)

        # --- Layer: Face Mesh (468 Dots) ---
        if self.layers["Face Mesh"] and face_result and face_result.face_landmarks:
            for f_lms in face_result.face_landmarks:
                draw_landmarks(frame, f_lms, color=(200, 200, 200), radius=1)

        # --- Layer: Hands ---
        if self.layers["Hands"] and hand_result and hand_result.hand_landmarks:
            for h_lms in hand_result.hand_landmarks:
                coords = draw_landmarks(frame, h_lms, color=(0, 255, 0), radius=3)
                draw_connections(frame, coords, HAND_CONNECTIONS, color=(50, 205, 50), thickness=2)

        # --- Layer: Face Recognition ---
        if self.layers["Face Recognition"]:
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                box_color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
                cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
                cv2.rectangle(frame, (left, top - 32), (right, top), box_color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, top - 8), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

            if "Unknown" in self.face_names:
                cv2.rectangle(frame, (450, 15), (830, 55), (0, 0, 200), -1)
                cv2.putText(frame, "UNKNOWN FACE DETECTED", (470, 42),
                            cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)

        # --- Settings Button (Top-Right) ---
        bx1, by1, bx2, by2 = self.SETTINGS_BTN
        btn_bg = (50, 50, 50) if not self.settings_open else (200, 90, 0)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), btn_bg, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "SETTINGS [S]", (bx1 + 12, by1 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # --- Settings Overlay Drawer ---
        if self.settings_open:
            overlay = frame.copy()
            cv2.rectangle(overlay, (840, 70), (1260, 570), (15, 15, 20), -1)
            frame = cv2.addWeighted(overlay, 0.88, frame, 0.12, 0)
            cv2.rectangle(frame, (840, 70), (1260, 570), (100, 100, 100), 1)

            # Actions
            actions = [
                ("Auto-Register", f"Auto-Register: {'ON' if self.auto_register_toggle else 'OFF'}", (0, 140, 0) if self.auto_register_toggle else (60, 60, 60)),
                ("Register New Face", "Register New Face", (60, 60, 60)),
                ("Export Data", "Export Data (.pkl)", (60, 60, 60)),
                ("Import Data", "Import Data (.pkl)", (60, 60, 60)),
            ]
            for key_name, label, col in actions:
                x1, y1, x2, y2 = self.SETTING_ITEMS[key_name]
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, -1, cv2.LINE_AA)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(frame, label, (x1 + 14, y1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

            cv2.putText(frame, "--- LAYERS ---", (970, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 1)

            # Layer Toggles
            layer_keys = [
                ("Layer: Face Mesh", "Face Mesh", self.layers["Face Mesh"]),
                ("Layer: Body Pose", "Body Pose", self.layers["Body Pose"]),
                ("Layer: Hands", "Hands", self.layers["Hands"]),
                ("Layer: Face Recognition", "Face Recognition", self.layers["Face Recognition"]),
                ("Layer: FPS Counter", "FPS Counter", self.layers["FPS Counter"]),
            ]
            for item_key, title, enabled in layer_keys:
                x1, y1, x2, y2 = self.SETTING_ITEMS[item_key]
                b_col = (0, 120, 0) if enabled else (40, 40, 40)
                status_box = "[x] " if enabled else "[ ] "
                cv2.rectangle(frame, (x1, y1), (x2, y2), b_col, -1, cv2.LINE_AA)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 150, 150), 1, cv2.LINE_AA)
                cv2.putText(frame, f"{status_box}{title}", (x1 + 14, y1 + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

        # --- Layer: FPS Counter ---
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time) if (curr_time - self.prev_time) > 0 else 0
        self.prev_time = curr_time

        if self.layers["FPS Counter"]:
            cv2.putText(frame, f"FPS: {int(fps)}", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)

        # Render QPixmap inside QLabel
        rgb_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_disp.shape
        qimg = QImage(rgb_disp.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_view.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_view.setPixmap(pix)

    # ----------------- Mouse & Touch Events ----------------- #
    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or self.video_view.pixmap() is None:
            super().mousePressEvent(event)
            return

        # Map screen click coordinates back to internal (1280x720) resolution
        pix_rect = self._get_pixmap_rect()
        pos = event.position().toPoint()

        if not pix_rect.contains(pos):
            super().mousePressEvent(event)
            return

        norm_x = (pos.x() - pix_rect.x()) / pix_rect.width()
        norm_y = (pos.y() - pix_rect.y()) / pix_rect.height()
        frame_x = int(norm_x * 1280)
        frame_y = int(norm_y * 720)

        # Settings button hit check
        sx1, sy1, sx2, sy2 = self.SETTINGS_BTN
        if sx1 <= frame_x <= sx2 and sy1 <= frame_y <= sy2:
            self.settings_open = not self.settings_open
            return

        # Settings panel actions hit check
        if self.settings_open:
            for action, (x1, y1, x2, y2) in self.SETTING_ITEMS.items():
                if x1 <= frame_x <= x2 and y1 <= frame_y <= y2:
                    if action == "Auto-Register":
                        self.auto_register_toggle = not self.auto_register_toggle
                    elif action == "Register New Face":
                        self._manual_register()
                    elif action == "Export Data":
                        self._export_data()
                    elif action == "Import Data":
                        self._import_data()
                    elif action == "Layer: Face Mesh":
                        self.layers["Face Mesh"] = not self.layers["Face Mesh"]
                    elif action == "Layer: Body Pose":
                        self.layers["Body Pose"] = not self.layers["Body Pose"]
                    elif action == "Layer: Hands":
                        self.layers["Hands"] = not self.layers["Hands"]
                    elif action == "Layer: Face Recognition":
                        self.layers["Face Recognition"] = not self.layers["Face Recognition"]
                    elif action == "Layer: FPS Counter":
                        self.layers["FPS Counter"] = not self.layers["FPS Counter"]
                    return

        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_S:
            self.settings_open = not self.settings_open
        elif event.key() == Qt.Key.Key_T:
            self.auto_register_toggle = not self.auto_register_toggle
        super().keyPressEvent(event)

    def _get_pixmap_rect(self):
        pix = self.video_view.pixmap()
        if not pix:
            return self.video_view.rect()
        lbl_size = self.video_view.size()
        pix_size = pix.size()
        x = (lbl_size.width() - pix_size.width()) // 2
        y = (lbl_size.height() - pix_size.height()) // 2
        from PyQt6.QtCore import QRect
        return QRect(x, y, pix_size.width(), pix_size.height())

    # ----------------- Qt Modal Dialogs ----------------- #
    def _prompt_register_face(self, encoding):
        self.is_prompt_open = True
        name, ok = QInputDialog.getText(self, "Register Face", "New face detected! Enter Name:")
        if ok and name.strip():
            self.known_face_encodings.append(encoding)
            self.known_face_names.append(name.strip())
            self._save_database()
            QMessageBox.information(self, "Face Saved", f"Registered '{name.strip()}' successfully!")
        self.is_prompt_open = False

    def _manual_register(self):
        if not self.last_detected_encodings:
            QMessageBox.warning(self, "No Face", "No face is currently visible in frame to register.")
            return
        self._prompt_register_face(self.last_detected_encodings[0])

    def _export_data(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Face Database", "known_faces_backup.pkl", "Pickle Files (*.pkl)"
        )
        if file_path:
            with open(file_path, "wb") as f:
                pickle.dump({
                    "encodings": self.known_face_encodings,
                    "names": self.known_face_names
                }, f)
            QMessageBox.information(self, "Export", "Face database exported successfully!")

    def _import_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Face Database", "", "Pickle Files (*.pkl)"
        )
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                    new_encs = data.get("encodings", [])
                    new_names = data.get("names", [])
                    self.known_face_encodings.extend(new_encs)
                    self.known_face_names.extend(new_names)
                    self._save_database()
                QMessageBox.information(self, "Import", f"Imported {len(new_names)} faces successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to import faces: {e}")