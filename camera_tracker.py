"""
camera_tracker.py - Holistic Body, Pose, Hand & Face-ID Recognition Tracker for A.R.C
Integrates:
- MediaPipe Pose, Hand & Face Mesh (468 dots + Lip/Eye/Brow wireframe contours) tracking
- Dedicated 'MESH 3D [M]' button that opens a standalone 3D HOLOGRAM window:
  every person in front of the camera gets a full connected body (head, neck, arms, hands, fingers, legs, feet)
  that moves in 3D like they do; parts outside the camera view stay visible but frozen (dimmer);
  each face is mounted on its neck and only its expression (smile, mouth, eyes, brows) animates;
  the mouse rotates / zooms the whole scene (view only - it never moves body parts)
- dlib / face_recognition 128D embedding face identifier
- Persistent storage for known faces (known_faces.pkl)
- In-frame Settings panel with Layers control, Auto-Registration, and Import/Export
"""

import os
import math
import time
import pickle
import numpy as np
import cv2

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPoint, QRect
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QInputDialog, QFileDialog, QMessageBox, QDialog, QCheckBox,
    QSizePolicy
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
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index Finger
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle Finger
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring Finger
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky Finger
    (0, 17)                                # Palm base
]

# Key MediaPipe 468 Face Mesh Contours so smiles, mouth opening, eyes, and jawline animate clearly
FACE_LIPS_OUTER = [
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314),
    (314, 405), (405, 321), (321, 375), (375, 291), (291, 409), (409, 270),
    (270, 269), (269, 267), (267, 0), (0, 37), (37, 39), (39, 40),
    (40, 185), (185, 61)
]

FACE_LIPS_INNER = [
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14), (14, 317),
    (317, 402), (402, 318), (318, 324), (324, 308), (308, 415), (415, 310),
    (310, 311), (311, 312), (312, 13), (13, 82), (82, 81), (81, 80),
    (80, 191), (191, 78)
]

FACE_LEFT_EYE = [
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
    (154, 155), (155, 133), (133, 173), (173, 157), (157, 158), (158, 159),
    (159, 160), (160, 161), (161, 246), (246, 33)
]

FACE_RIGHT_EYE = [
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381),
    (381, 382), (382, 362), (362, 398), (398, 384), (384, 385), (385, 386),
    (386, 387), (387, 388), (388, 466), (466, 263)
]

FACE_LEFT_EYEBROW = [
    (70, 63), (63, 105), (105, 66), (66, 107), (107, 55), (55, 65), (65, 52), (52, 53), (53, 46)
]

FACE_RIGHT_EYEBROW = [
    (300, 293), (293, 334), (334, 296), (296, 336), (336, 285), (285, 295), (295, 282), (282, 283), (283, 276)
]

FACE_OVAL = [
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10)
]

NOSE_BRIDGE = [(168, 6), (6, 197), (197, 195), (195, 5), (5, 4), (4, 1)]


def draw_landmarks(frame, landmarks, color=(0, 255, 0), radius=2):
    h, w, _ = frame.shape
    coords = []
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        coords.append((cx, cy))
        cv2.circle(frame, (cx, cy), radius, color, -1, cv2.LINE_AA)
    return coords


def draw_connections(frame, coords, connections, color=(255, 255, 255), thickness=1):
    for start_idx, end_idx in connections:
        if start_idx < len(coords) and end_idx < len(coords):
            cv2.line(frame, coords[start_idx], coords[end_idx], color, thickness, cv2.LINE_AA)


# ----------------- 3D Hologram Scene Constants ----------------- #
FRAME_W, FRAME_H = 1280, 720            # internal camera canvas size (matches _process_frame)
CAM_FOCAL_PX = 1000.0                   # assumed webcam focal length in pixels (1280 px wide)
DEPTH_REF_M = 2.0                       # camera distance that maps to scene depth 0
FLOOR_Y = -1.35                         # hologram floor height (metres)

MAX_PEOPLE = 4                          # how many bodies can appear in the hologram
PERSON_TTL = 2.0                        # seconds a person stays (frozen) after leaving the frame
MATCH_PX = 0.20 * FRAME_W               # max head movement between frames to count as same person
HAND_MATCH_PX = 0.15 * FRAME_W          # max wrist distance to attach a hand to a body
REAL_FACE_WIDTH_M = 0.145               # cheek-to-cheek width, used to guess distance of a face
FACE_SCALE = 1.6                        # hologram face is drawn this much bigger than life so smiles are readable (1.0 = real size)

# Landmarks that barely move when you smile / talk -> used as the face's stable anchor
FACE_RIGID = [10, 168, 6, 197, 195, 5, 4, 1, 33, 133, 362, 263, 234, 454, 127, 356]
# Left/right cheek-edge pairs used to find the face's own "right" direction
FACE_WIDTH_PAIRS = [(234, 454), (127, 356), (93, 323), (162, 389)]

# Bone colours / joint colours per person (BGR) so several bodies are easy to tell apart
BONE_COLORS = [(255, 200, 0), (0, 170, 255), (220, 90, 255), (90, 255, 130)]
JOINT_COLORS = [(0, 255, 255), (0, 220, 255), (255, 230, 140), (140, 255, 210)]

# Extra bones so the body looks like one connected figure
# (33 = synthetic mid-shoulders, 34 = synthetic mid-hips; added when the scene is built)
HEAD_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), (33, 0)]
HOLO_EXTRA_CONNECTIONS = HEAD_CONNECTIONS + [(33, 34)]

# If the hand tracker can't see a hand, fall back to the body model's 3-point hand
POSE_HAND_FALLBACK = {
    15: [(15, 17), (15, 19), (15, 21), (17, 19)],
    16: [(16, 18), (16, 20), (16, 22), (18, 20)],
}
POSE_FALLBACK_POINTS = {15: (17, 19, 21), 16: (18, 20, 22)}

# Brightness of a joint: seen live / remembered (out of frame) / never seen yet
DIM_LIVE, DIM_FROZEN, DIM_TEMPLATE = 1.0, 0.55, 0.40


def _make_body_template():
    """Neutral standing body (hip-centred, y up, z toward viewer, landmark 11 on +x).
    Used to fill in body parts that have never been visible yet."""
    p = {0: (0, .62, .05), 1: (.03, .65, .05), 2: (.06, .65, .05), 3: (.09, .64, .03),
         4: (-.03, .65, .05), 5: (-.06, .65, .05), 6: (-.09, .64, .03),
         7: (.08, .63, 0), 8: (-.08, .63, 0), 9: (.03, .56, .05), 10: (-.03, .56, .05),
         11: (.2, .45, 0), 12: (-.2, .45, 0), 13: (.28, .2, 0), 14: (-.28, .2, 0),
         15: (.3, -.05, 0), 16: (-.3, -.05, 0), 17: (.31, -.1, 0), 18: (-.31, -.1, 0),
         19: (.3, -.12, 0), 20: (-.3, -.12, 0), 21: (.28, -.09, 0), 22: (-.28, -.09, 0),
         23: (.1, 0, 0), 24: (-.1, 0, 0), 25: (.11, -.4, 0), 26: (-.11, -.4, 0),
         27: (.11, -.8, 0), 28: (-.11, -.8, 0), 29: (.12, -.85, 0), 30: (-.12, -.85, 0),
         31: (.1, -.88, .1), 32: (-.1, -.88, .1)}
    t = np.zeros((33, 3))
    for k, v in p.items():
        t[k] = v
    return t


BODY_TEMPLATE = _make_body_template()


def cam_to_scene(px, py, depth):
    """Camera pixel + distance (m) -> scene metres (x right, y up, z toward viewer)."""
    return np.array([
        (px - FRAME_W / 2) / CAM_FOCAL_PX * depth,
        -(py - FRAME_H / 2) / CAM_FOCAL_PX * depth,
        DEPTH_REF_M - depth,
    ])


def depth_from_scale(d_img_px, d_world_m, fallback=None):
    """Pinhole-camera distance estimate: how far away something is from how big it looks."""
    if d_img_px < 25 or d_world_m < 0.05:
        return fallback
    return float(np.clip(CAM_FOCAL_PX * d_world_m / d_img_px, 0.3, 6.0))


def ortho(M):
    """Nearest proper rotation matrix to M."""
    U, _, Vt = np.linalg.svd(M)
    if np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1
    return U @ Vt


def body_basis(p):
    """Body orientation from the shoulders + hips: columns = (right, up, facing) axes."""
    ex = (p[11] - p[12]) + (p[23] - p[24])
    ey = (p[11] + p[12]) / 2 - (p[23] + p[24]) / 2
    nx = np.linalg.norm(ex)
    if nx < 1e-6:
        return None
    ex = ex / nx
    ey = ey - np.dot(ey, ex) * ex
    ny = np.linalg.norm(ey)
    if ny < 1e-6:
        return None
    ey = ey / ny
    return np.stack([ex, ey, np.cross(ex, ey)], axis=1)


def face_to_local(f_lms):
    """
    Turn raw face landmarks into a 'face-only' shape:
    head position, distance to the camera, and head rotation are all removed,
    so only smiles / mouth / eyes / brows / face shape remain.
    Output axes: x = right, y = up, z = toward viewer, face width = 1.0
    """
    p = np.array([[l.x * FRAME_W, -l.y * FRAME_H, -l.z * FRAME_W] for l in f_lms], dtype=float)
    if len(p) < 468:
        return None

    centre = p[FACE_RIGID].mean(axis=0)

    ex = sum(p[b] - p[a] for a, b in FACE_WIDTH_PAIRS)
    if ex[0] < 0:
        ex = -ex
    ey = p[10] - p[152]                                   # chin -> forehead
    nx, ny = np.linalg.norm(ex), np.linalg.norm(ey)
    if nx < 1e-6 or ny < 1e-6:
        return None
    ex = ex / nx
    ey = ey - np.dot(ey, ex) * ex
    nyo = np.linalg.norm(ey)
    if nyo < 1e-6:
        return None
    ey = ey / nyo
    ez = np.cross(ex, ey)

    scale = np.linalg.norm(p[454] - p[234])
    if scale < 1e-6:
        return None
    basis = np.stack([ex, ey, ez], axis=1)                # columns = face axes
    return ((p - centre) @ basis) / scale


# ----------------- One tracked person (body + face + hands memory) ----------------- #
class TrackedPerson:
    """
    Keeps one body alive across frames.
      - joints that are visible in the camera move live
      - joints that are NOT visible stay frozen where they were last seen
        (and follow the body when it moves)
      - joints never seen yet come from a neutral standing template
      - the face is stabilised and mounted on top of the neck
    """
    _next_id = 0

    def __init__(self, head_px, now):
        self.id = TrackedPerson._next_id
        TrackedPerson._next_id += 1
        self.head_px = np.array(head_px, dtype=float)
        self.last_seen = now
        self.z = None
        self.origin = None
        self.B = np.eye(3)
        self.has_pose = False
        self.scale = 1.0
        self.local = np.zeros((33, 3))
        self.known = np.zeros(33, dtype=bool)
        self.live = np.zeros(33, dtype=bool)
        self.pose_prev = None
        self.face_local = None
        self.face_prev = None
        self.face_live = False
        self.hand_local = {15: None, 16: None}

    # --- helpers --- #
    def _smooth_depth(self, z_new):
        if z_new is None:
            return
        self.z = z_new if self.z is None else 0.7 * self.z + 0.3 * z_new

    def _display_local(self):
        return np.where(self.known[:, None], self.local, BODY_TEMPLATE * self.scale)

    # --- per-frame update --- #
    def update(self, obs, now):
        """obs = this frame's observation for this person, or None if not seen this frame."""
        if obs is not None:
            self.last_seen = now
            self.head_px = np.array(obs["head"], dtype=float)

        has_pose_obs = obs is not None and obs.get("wpt") is not None
        face_obs = obs.get("face") if obs is not None else None

        if has_pose_obs:
            self._update_from_pose(obs)
        else:
            self.live[:] = False          # nothing in the camera -> everything frozen
            if face_obs is not None:
                self._update_origin_from_face(face_obs)

        if self.origin is None:
            return None

        # ---- body points in scene space ----
        disp = self._display_local()
        pts = disp @ self.B.T + self.origin
        pts = np.vstack([pts, pts[[11, 12]].mean(0), pts[[23, 24]].mean(0)])     # idx 33, 34

        dim = np.where(self.live, DIM_LIVE, np.where(self.known, DIM_FROZEN, DIM_TEMPLATE))
        dim = np.append(dim, [min(dim[11], dim[12]), min(dim[23], dim[24])])

        if self.pose_prev is not None and self.pose_prev.shape == pts.shape:
            pts = 0.55 * pts + 0.45 * self.pose_prev
        self.pose_prev = pts

        # ---- face: stabilised, mounted on top of the neck, follows body orientation ----
        face_pts, face_dim = None, DIM_LIVE
        if face_obs is not None:
            local = face_to_local(face_obs["lms"])
            if local is not None:
                if self.face_prev is not None and self.face_prev.shape == local.shape:
                    local = 0.6 * local + 0.4 * self.face_prev
                self.face_prev = local
                self.face_local = local
                self.face_live = True
            else:
                self.face_live = False
        else:
            self.face_live = False

        if self.face_local is not None:
            fw = float(np.clip(0.38 * np.linalg.norm(pts[11] - pts[12]), 0.10, 0.22)) * FACE_SCALE
            head_c = pts[[7, 8]].mean(0) + self.B[:, 2] * 0.04
            face_pts = head_c + (self.face_local * fw) @ self.B.T
            face_dim = DIM_LIVE if self.face_live else DIM_FROZEN

        return {
            "id": self.id, "pose": pts, "dim": dim,
            "face_pts": face_pts, "face_dim": face_dim,
            "hands": [], "anchored": set(),
        }

    def _update_from_pose(self, obs):
        wpt, vis, nimg, ipx = obs["wpt"], obs["vis"], obs["nimg"], obs["ipx"]

        Bn = body_basis(wpt)
        if Bn is not None:
            self.B = Bn if not self.has_pose else ortho(0.6 * Bn + 0.4 * self.B)

        # distance from the camera: how big the torso looks vs. its real size
        sh_i, hp_i = ipx[[11, 12]].mean(0), ipx[[23, 24]].mean(0)
        sh_w, hp_w = wpt[[11, 12]].mean(0), wpt[[23, 24]].mean(0)
        z_new = depth_from_scale(np.linalg.norm(sh_i - hp_i), np.linalg.norm((sh_w - hp_w)[:2]))
        if z_new is None:       # torso too small -> use shoulder width instead
            z_new = depth_from_scale(np.linalg.norm(ipx[11] - ipx[12]),
                                     np.linalg.norm((wpt[11] - wpt[12])[:2]))
        if z_new is None and self.z is None:
            z_new = DEPTH_REF_M
        self._smooth_depth(z_new)
        self.origin = cam_to_scene(hp_i[0], hp_i[1], self.z)

        torso = np.linalg.norm(sh_w - hp_w)
        if torso > 0.1:
            s_new = torso / 0.45
            self.scale = s_new if not self.has_pose else 0.9 * self.scale + 0.1 * s_new

        # which joints are really visible in the camera right now (with hysteresis)
        inside = ((nimg[:, 0] > 0.02) & (nimg[:, 0] < 0.98) &
                  (nimg[:, 1] > 0.02) & (nimg[:, 1] < 0.98))
        thr = np.where(self.live, 0.35, 0.55)
        self.live = (vis > thr) & inside

        local_obs = wpt @ self.B
        self.local[self.live] = local_obs[self.live]
        self.known |= self.live
        self.has_pose = True

    def _update_origin_from_face(self, face_obs):
        """Only a face is visible: hang the (frozen) body below it."""
        if face_obs["width_px"] > 10:
            z_new = float(np.clip(CAM_FOCAL_PX * REAL_FACE_WIDTH_M / face_obs["width_px"], 0.3, 6.0))
        else:
            z_new = None
        if z_new is None and self.z is None:
            z_new = DEPTH_REF_M
        self._smooth_depth(z_new)
        head_target = cam_to_scene(face_obs["nose"][0], face_obs["nose"][1], self.z)
        disp = self._display_local()
        ears = (disp[7] + disp[8]) / 2
        self.origin = head_target - ears @ self.B.T


# ----------------- Software 3D Hologram Renderer ----------------- #
class HologramRenderer:
    """Tiny orbit-camera 3D renderer (numpy + OpenCV). Mouse only moves the VIEW, never the body."""

    def __init__(self):
        self.reset_view()

    def reset_view(self):
        self.yaw = 0.0
        self.pitch = 0.10
        self.dist = 3.0
        self.target = np.array([0.0, -0.15, 0.0])

    def orbit(self, dx, dy):
        self.yaw += dx * 0.008
        self.pitch = float(np.clip(self.pitch + dy * 0.008, -1.35, 1.35))

    def zoom(self, steps):
        self.dist = float(np.clip(self.dist * (0.9 ** steps), 0.6, 9.0))

    # --- projection -------------------------------------------------- #
    def project(self, pts, w, h):
        p = np.asarray(pts, dtype=float).reshape(-1, 3) - self.target
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        x1 = p[:, 0] * cy + p[:, 2] * sy
        z1 = -p[:, 0] * sy + p[:, 2] * cy
        y2 = p[:, 1] * cp - z1 * sp
        z2 = p[:, 1] * sp + z1 * cp
        depth = self.dist - z2
        ok = depth > 0.25
        d = np.where(ok, depth, 1.0)
        f = 1.1 * h
        return w / 2 + f * x1 / d, h / 2 - f * y2 / d, depth, ok

    def _shade(self, depth):
        """Near = bright, far = dim (gives the hologram its depth)."""
        return float(np.clip(1.0 - (depth - (self.dist - 1.0)) / 4.0, 0.4, 1.0))

    def _line(self, img, P, i, j, color, thick, dim=1.0):
        u, v, depth, ok = P
        if not (ok[i] and ok[j]):
            return
        s = self._shade((depth[i] + depth[j]) / 2) * dim
        col = (int(color[0] * s), int(color[1] * s), int(color[2] * s))
        cv2.line(img, (int(u[i]), int(v[i])), (int(u[j]), int(v[j])), col, thick, cv2.LINE_AA)

    def _dot(self, img, P, i, color, radius, dim=1.0):
        u, v, depth, ok = P
        if not ok[i]:
            return
        s = self._shade(depth[i]) * dim
        col = (int(color[0] * s), int(color[1] * s), int(color[2] * s))
        cv2.circle(img, (int(u[i]), int(v[i])), radius, col, -1, cv2.LINE_AA)

    # --- scene parts ------------------------------------------------- #
    def _draw_floor(self, img, w, h):
        ext, step = 2.5, 0.5
        for a in np.arange(-ext, ext + 1e-6, step):
            for p0, p1 in (((a, FLOOR_Y, -ext), (a, FLOOR_Y, ext)),
                           ((-ext, FLOOR_Y, a), (ext, FLOOR_Y, a))):
                P = self.project(np.array([p0, p1]), w, h)
                self._line(img, P, 0, 1, (70, 50, 14), 1)

    def _draw_pose(self, img, person, w, h, show_floor, hide_head):
        pts, dim, anchored = person["pose"], person["dim"], person["anchored"]
        bone_col = BONE_COLORS[person["id"] % len(BONE_COLORS)]
        joint_col = JOINT_COLORS[person["id"] % len(JOINT_COLORS)]
        P = self.project(pts, w, h)
        u, v, depth, ok = P

        conns = list(POSE_CONNECTIONS) + [c for c in HOLO_EXTRA_CONNECTIONS
                                          if not (hide_head and c in HEAD_CONNECTIONS)]
        hidden = set(range(11)) if hide_head else set()
        for wrist, idxs in POSE_FALLBACK_POINTS.items():
            if wrist in anchored:
                hidden.update(idxs)
            else:
                conns += POSE_HAND_FALLBACK[wrist]

        for i, j in conns:
            if i in hidden or j in hidden:
                continue
            self._line(img, P, i, j, bone_col, 3, dim=min(dim[i], dim[j]))

        f = 1.1 * h
        for i in range(33):
            if i in hidden:
                continue
            r = 2 if i <= 10 else int(np.clip(0.014 * f / max(depth[i], 0.3), 2, 9))
            self._dot(img, P, i, joint_col, r, dim[i])

        if show_floor:   # joint "shadows" on the floor help you read depth while rotating
            sh = pts.copy()
            sh[:, 1] = FLOOR_Y
            PS = self.project(sh, w, h)
            for i in range(11, 33):
                if i not in hidden:
                    self._dot(img, PS, i, (90, 60, 20), 2, dim[i])

    def _draw_face(self, img, person, w, h):
        pts, dim = person["face_pts"], person["face_dim"]
        P = self.project(pts, w, h)
        u, v, depth, ok = P
        width_px = math.hypot(u[234] - u[454], v[234] - v[454])
        r = 2 if width_px > 140 else 1
        thin = width_px < 90
        for i in range(len(pts)):
            self._dot(img, P, i, (255, 220, 190), r, dim)
        groups = (
            (FACE_OVAL, (180, 180, 180), 1),
            (FACE_LIPS_OUTER, (120, 255, 0), 2),
            (FACE_LIPS_INNER, (0, 230, 255), 2),
            (FACE_LEFT_EYE, (255, 255, 0), 2),
            (FACE_RIGHT_EYE, (255, 255, 0), 2),
            (FACE_LEFT_EYEBROW, (255, 180, 0), 2),
            (FACE_RIGHT_EYEBROW, (255, 180, 0), 2),
            (NOSE_BRIDGE, (200, 200, 255), 1),
        )
        for conns, col, th in groups:
            for a, b in conns:
                if a < len(pts) and b < len(pts):
                    self._line(img, P, a, b, col, 1 if thin else th, dim)

        # neck: joins the top of the shoulders to the chin
        neck = np.array([person["pose"][33], pts[152]])
        PN = self.project(neck, w, h)
        self._line(img, PN, 0, 1, BONE_COLORS[person["id"] % len(BONE_COLORS)], 3,
                   dim=min(person["dim"][11], person["dim"][12], dim))

    def _draw_hand(self, img, pts, w, h, dim=1.0):
        P = self.project(pts, w, h)
        for a, b in HAND_CONNECTIONS:
            self._line(img, P, a, b, (50, 255, 180), 2, dim)
        for i in range(len(pts)):
            self._dot(img, P, i, (0, 255, 120), 3, dim)
        u, v, depth, ok = P
        for tip in (4, 8, 12, 16, 20):
            if ok[tip]:
                s = int(255 * dim)
                cv2.circle(img, (int(u[tip]), int(v[tip])), 6,
                           (0, int(215 * dim), s), 2, cv2.LINE_AA)

    @staticmethod
    def _glow(img):
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(1, w // 3), max(1, h // 3)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 3)
        glow = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.addWeighted(img, 1.0, glow, 1.3, 0)

    # --- main entry -------------------------------------------------- #
    def render(self, scene, w, h, show_grid=True, show_face=True, show_hands=True,
               show_pose=True, fps=0.0):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (12, 6, 3)
        if show_grid:
            self._draw_floor(img, w, h)

        n_people = 0
        if scene and scene.get("people") is not None:
            n_people = len(scene["people"])
            for person in scene["people"]:
                face_on = show_face and person.get("face_pts") is not None
                if show_pose:
                    self._draw_pose(img, person, w, h, show_grid, hide_head=face_on)
                if face_on:
                    self._draw_face(img, person, w, h)
                if show_hands:
                    for hand_pts, hand_dim in person["hands"]:
                        self._draw_hand(img, hand_pts, w, h, hand_dim)
            if show_hands:
                for hand_pts in scene.get("free_hands", []):
                    self._draw_hand(img, hand_pts, w, h)

        if n_people == 0 and not (scene and scene.get("free_hands")):
            cv2.putText(img, "Step in front of the camera...", (w // 2 - 170, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 150, 180), 1, cv2.LINE_AA)

        img = self._glow(img)
        cv2.putText(img, f"HOLOGRAM TWIN  |  PEOPLE: {n_people}  |  FPS: {int(fps)}", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (231, 255, 0), 2, cv2.LINE_AA)
        return img


class ViewportLabel(QLabel):
    """Label that turns mouse drags / wheel into view-rotation signals."""
    dragged = pyqtSignal(float, float)
    zoomed = pyqtSignal(float)
    reset_requested = pyqtSignal()

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._last = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._last = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._last is not None:
            pos = event.position()
            self.dragged.emit(pos.x() - self._last.x(), pos.y() - self._last.y())
            self._last = pos
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._last = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.reset_requested.emit()

    def wheelEvent(self, event):
        self.zoomed.emit(event.angleDelta().y() / 120.0)


# ----------------- Standalone 3D Hologram Window ----------------- #
class MeshTwinWindow(QDialog):
    """
    Separate window with a 3D hologram of everyone in front of the camera:
      - each person gets a full body (head, neck, arms, hands, fingers, legs, feet)
      - parts visible in the camera move live; parts out of frame stay frozen (dimmer)
      - the face is mounted on the neck; only smiles / mouth / eyes / brows animate
      - mouse drag rotates the WHOLE scene (view only), wheel zooms, double-click resets
    """
    def __init__(self, parent=None, on_close_cb=None):
        super().__init__(parent)
        self.on_close_cb = on_close_cb
        self.setWindowTitle("A.R.C — 3D Hologram Twin (drag to rotate, wheel to zoom)")
        self.resize(1000, 680)
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        self.setStyleSheet("""
            QDialog { background-color: #030710; color: #d0e8ff; font-family: 'Segoe UI'; }
            QCheckBox { color: #00ffe7; font-weight: bold; font-size: 12px; spacing: 6px; }
            QLabel { color: #8ab4d8; font-size: 12px; }
            QPushButton { background-color: #0b2740; color: #00ffe7; border: 1px solid #143a60;
                          border-radius: 4px; padding: 4px 12px; font-weight: bold; }
            QPushButton:hover { background-color: #12406a; }
        """)

        self.show_grid = True
        self.show_face = True
        self.show_hands = True
        self.show_pose = True

        self.renderer = HologramRenderer()
        self.scene = None
        self.fps = 0.0

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        title_row = QHBoxLayout()
        lbl_title = QLabel("◈ 3D HOLOGRAM TWIN — BODIES, HANDS & FACE EXPRESSION")
        lbl_title.setStyleSheet("color: #00ffe7; font-weight: bold; font-size: 13px;")
        self.btn_reset = QPushButton("Reset View")
        self.btn_reset.clicked.connect(self._reset_view)
        title_row.addWidget(lbl_title)
        title_row.addStretch()
        title_row.addWidget(self.btn_reset)
        lay.addLayout(title_row)

        ctrl_row = QHBoxLayout()

        def make_chk(text, attr, checked=True):
            chk = QCheckBox(text)
            chk.setChecked(checked)
            chk.toggled.connect(lambda v: (setattr(self, attr, v), self._redraw()))
            ctrl_row.addWidget(chk)
            return chk

        self.chk_face = make_chk("Face & Smile Mesh", "show_face")
        self.chk_hands = make_chk("Hands & Fingers", "show_hands")
        self.chk_pose = make_chk("Body Skeleton", "show_pose")
        self.chk_grid = make_chk("Floor Grid", "show_grid")
        ctrl_row.addStretch()
        lay.addLayout(ctrl_row)

        self.canvas_lbl = ViewportLabel("Waiting for live landmarks...")
        self.canvas_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas_lbl.setMinimumSize(320, 200)
        self.canvas_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.canvas_lbl.setStyleSheet(
            "background-color: #02050c; border: 1px solid #143a60; border-radius: 8px;"
        )
        self.canvas_lbl.dragged.connect(self._on_drag)
        self.canvas_lbl.zoomed.connect(self._on_zoom)
        self.canvas_lbl.reset_requested.connect(self._reset_view)
        lay.addWidget(self.canvas_lbl, 1)

        lay.addWidget(QLabel(
            "Drag = rotate the whole scene   •   Wheel = zoom   •   Double-click = reset view   •   "
            "Dim parts are out of camera view (show them to move them)"))

    # --- interaction (changes the VIEW only) ------------------------- #
    def _on_drag(self, dx, dy):
        self.renderer.orbit(dx, dy)
        self._redraw()

    def _on_zoom(self, steps):
        self.renderer.zoom(steps)
        self._redraw()

    def _reset_view(self):
        self.renderer.reset_view()
        self._redraw()

    # --- drawing ----------------------------------------------------- #
    def set_scene(self, scene, fps=0.0):
        self.scene = scene
        self.fps = fps
        self._redraw()

    def _redraw(self):
        if not self.isVisible():
            return
        w, h = self.canvas_lbl.width() - 2, self.canvas_lbl.height() - 2
        if w < 64 or h < 64:
            return
        img = self.renderer.render(
            self.scene, w, h,
            show_grid=self.show_grid, show_face=self.show_face,
            show_hands=self.show_hands, show_pose=self.show_pose, fps=self.fps,
        )
        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.canvas_lbl.setPixmap(QPixmap.fromImage(qimg))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._redraw()

    def closeEvent(self, event):
        if self.on_close_cb:
            self.on_close_cb()
        super().closeEvent(event)



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

        # Standalone Mesh Twin Window instance
        self.mesh_window = MeshTwinWindow(self, on_close_cb=self._on_mesh_win_closed)

        # People currently tracked for the 3D hologram (one TrackedPerson per human)
        self._people = []

        # On-screen UI layout coordinates (based on 1280x720 internal canvas)
        self.MESH_BTN = (950, 20, 1105, 60)
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

        self.req_register_signal.connect(self._prompt_register_face)

    def _on_mesh_win_closed(self):
        pass

    def toggle_mesh_window(self):
        if self.mesh_window.isVisible():
            self.mesh_window.hide()
        else:
            self.mesh_window.show()
            self.mesh_window.raise_()
            self.mesh_window.activateWindow()

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
                    running_mode=vision.RunningMode.VIDEO,
                    num_poses=MAX_PEOPLE
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
        self._people = []
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        self.video_view.setText("Camera stopped.")

    # ----------------- Build the 3D Hologram Scene ----------------- #
    @staticmethod
    def _greedy_pairs(dist_matrix, max_dist):
        """Match rows to columns, closest first, each used at most once."""
        pairs, used_r, used_c = [], set(), set()
        cand = sorted(
            (d, r, c)
            for r, row in enumerate(dist_matrix) for c, d in enumerate(row) if d < max_dist
        )
        for d, r, c in cand:
            if r in used_r or c in used_c:
                continue
            used_r.add(r)
            used_c.add(c)
            pairs.append((r, c))
        return pairs

    def _build_hologram_scene(self, pose_result, face_result, hand_result):
        """
        One TrackedPerson per human in front of the camera.
          Body  : moves live in 3D where the camera sees it, stays frozen where it doesn't
          Face  : stabilised (only expression moves) and mounted on top of the neck
          Hands : real finger shapes pinned to the body's wrists
        """
        now = time.time()
        W, H = FRAME_W, FRAME_H

        # ---- 1. collect what the camera sees this frame ----
        poses = []
        if pose_result and pose_result.pose_landmarks and pose_result.pose_world_landmarks:
            for img, wld in zip(pose_result.pose_landmarks, pose_result.pose_world_landmarks):
                nimg = np.array([[l.x, l.y] for l in img])
                poses.append({
                    "nimg": nimg,
                    "ipx": nimg * [W, H],
                    "wpt": np.array([[l.x, -l.y, -l.z] for l in wld]),
                    "vis": np.array([(l.visibility if l.visibility is not None else 1.0) for l in img]),
                    "head": nimg[0] * [W, H],
                    "face": None,
                })

        faces = []
        if face_result and face_result.face_landmarks:
            for lms in face_result.face_landmarks:
                if len(lms) < 468:
                    continue
                nose = np.array([lms[1].x * W, lms[1].y * H])
                width_px = float(np.hypot((lms[234].x - lms[454].x) * W, (lms[234].y - lms[454].y) * H))
                faces.append({"lms": lms, "nose": nose, "width_px": width_px})

        hands = []
        if hand_result and hand_result.hand_landmarks and hand_result.hand_world_landmarks:
            for h_img, h_wld in zip(hand_result.hand_landmarks, hand_result.hand_world_landmarks):
                hands.append({
                    "ipx": np.array([[l.x * W, l.y * H] for l in h_img]),
                    "hw": np.array([[l.x, -l.y, -l.z] for l in h_wld]),
                })

        # ---- 2. faces -> bodies (a face belongs to the body whose head is next to it) ----
        observations = list(poses)
        if poses and faces:
            dm = [[np.linalg.norm(p["head"] - f["nose"]) for f in faces] for p in poses]
            matched_faces = set()
            for pi, fi in self._greedy_pairs(dm, 0.13 * W):
                poses[pi]["face"] = faces[fi]
                matched_faces.add(fi)
        else:
            matched_faces = set()
        for fi, f in enumerate(faces):
            if fi not in matched_faces:      # a face with no body detected -> its own person
                observations.append({"head": f["nose"], "face": f, "wpt": None})

        # ---- 3. hands -> bodies (a hand belongs to the body whose wrist is next to it) ----
        hand_for = {}                        # (pose index, wrist idx) -> hand
        used_hands = set()
        if poses and hands:
            slots = [(pi, wi) for pi in range(len(poses)) for wi in (15, 16)]
            dm = [[np.linalg.norm(hd["ipx"][0] - poses[pi]["ipx"][wi]) for (pi, wi) in slots]
                  for hd in hands]
            for hi, si in self._greedy_pairs(dm, HAND_MATCH_PX):
                hand_for[slots[si]] = hands[hi]
                used_hands.add(hi)

        # ---- 4. match observations to people we already track ----
        self._people = [p for p in self._people if now - p.last_seen <= PERSON_TTL]
        obs_for_person = {}
        if observations and self._people:
            dm = [[np.linalg.norm(o["head"] - p.head_px) for o in observations] for p in self._people]
            matched_obs = set()
            for pi, oi in self._greedy_pairs(dm, MATCH_PX):
                obs_for_person[pi] = observations[oi]
                matched_obs.add(oi)
        else:
            matched_obs = set()
        for oi, o in enumerate(observations):
            if oi not in matched_obs and len(self._people) < MAX_PEOPLE:
                self._people.append(TrackedPerson(o["head"], now))
                obs_for_person[len(self._people) - 1] = o

        # ---- 5. update every person and build the scene ----
        scene = {"people": [], "free_hands": []}
        for pi, person in enumerate(self._people):
            obs = obs_for_person.get(pi)
            entry = person.update(obs, now)
            if entry is None:
                continue

            pose_idx = next((i for i, p in enumerate(poses) if p is obs), None)
            B = person.B
            for wrist in (15, 16):
                hd = hand_for.get((pose_idx, wrist)) if pose_idx is not None else None
                wrist_pt = entry["pose"][wrist]
                if hd is not None:
                    shape = hd["hw"] - hd["hw"][0]
                    person.hand_local[wrist] = shape @ B           # remember it in body space
                    entry["hands"].append((shape + wrist_pt, DIM_LIVE))
                    entry["anchored"].add(wrist)
                elif person.hand_local[wrist] is not None:      # hand left the frame -> keep it frozen
                    entry["hands"].append((person.hand_local[wrist] @ B.T + wrist_pt, DIM_FROZEN))
                    entry["anchored"].add(wrist)
            scene["people"].append(entry)

        # ---- 6. hands that belong to nobody float where they are ----
        for hi, hd in enumerate(hands):
            if hi in used_hands:
                continue
            d_img = (np.linalg.norm(hd["ipx"][0] - hd["ipx"][9]) +
                     np.linalg.norm(hd["ipx"][5] - hd["ipx"][17]))
            d_w = (np.linalg.norm((hd["hw"][0] - hd["hw"][9])[:2]) +
                   np.linalg.norm((hd["hw"][5] - hd["hw"][17])[:2]))
            z = depth_from_scale(d_img, d_w, fallback=DEPTH_REF_M)
            wrist_scene = cam_to_scene(hd["ipx"][0][0], hd["ipx"][0][1], z)
            scene["free_hands"].append(hd["hw"] - hd["hw"][0] + wrist_scene)

        return scene


    # ----------------- Frame Processing Pipeline ----------------- #
    def _process_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return

        success, frame = self.cap.read()
        if not success:
            return

        frame = cv2.flip(frame, 1)
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

        # --- Layer: Face Mesh (468 Dots + Lip/Eye Contours) ---
        if self.layers["Face Mesh"] and face_result and face_result.face_landmarks:
            for f_lms in face_result.face_landmarks:
                f_coords = draw_landmarks(frame, f_lms, color=(200, 200, 200), radius=1)
                draw_connections(frame, f_coords, FACE_LIPS_OUTER, color=(0, 255, 180), thickness=1)
                draw_connections(frame, f_coords, FACE_LIPS_INNER, color=(0, 200, 255), thickness=1)
                draw_connections(frame, f_coords, FACE_LEFT_EYE, color=(255, 255, 0), thickness=1)
                draw_connections(frame, f_coords, FACE_RIGHT_EYE, color=(255, 255, 0), thickness=1)

        # --- Layer: Hands & Fingers ---
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

        # --- MESH 3D [M] Button (Top-Right next to Settings) ---
        mx1, my1, mx2, my2 = self.MESH_BTN
        mesh_active = self.mesh_window.isVisible()
        m_bg = (160, 120, 0) if mesh_active else (55, 45, 25)
        cv2.rectangle(frame, (mx1, my1), (mx2, my2), m_bg, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (mx1, my1), (mx2, my2), (231, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "MESH 3D [M]", (mx1 + 14, my1 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

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

        # Update the 3D Hologram Window if open
        if self.mesh_window.isVisible():
            scene = self._build_hologram_scene(pose_result, face_result, hand_result)
            self.mesh_window.set_scene(scene, fps)

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

    # ----------------- Mouse & Keyboard Events ----------------- #
    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or self.video_view.pixmap() is None:
            super().mousePressEvent(event)
            return

        pix_rect = self._get_pixmap_rect()
        pos = event.position().toPoint()

        if not pix_rect.contains(pos):
            super().mousePressEvent(event)
            return

        norm_x = (pos.x() - pix_rect.x()) / pix_rect.width()
        norm_y = (pos.y() - pix_rect.y()) / pix_rect.height()
        frame_x = int(norm_x * 1280)
        frame_y = int(norm_y * 720)

        # MESH 3D [M] button hit check -> Opens/Closes standalone Mesh Twin Window
        mx1, my1, mx2, my2 = self.MESH_BTN
        if mx1 <= frame_x <= mx2 and my1 <= frame_y <= my2:
            self.toggle_mesh_window()
            return

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
        elif event.key() == Qt.Key.Key_M:
            self.toggle_mesh_window()
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