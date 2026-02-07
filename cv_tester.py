#!/usr/bin/env python3
"""
yolo11n_person_pose_webcam.py

Requirements:
- Python 3.11
- ultralytics, opencv-python, torch

See requirements.txt included below.

What it does:
- Loads two Ultralytics YOLO models:
    * detection_model (yolo11n.pt) - used only to detect a person (class 'person').
    * pose_model (yolo11n-pose.pt) - used only on cropped person regions to estimate keypoints.
- Runs webcam, draws bbox, keypoints, skeleton, FPS.
- Prints JSON per-frame detection+pose info to stdout.
- Top-of-file global vars control thresholds and behavior.

Note: If CUDA is available, it will attempt to use it. Falls back to CPU with a warning.
"""

import sys
import time
import json
import logging
import threading
import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch

# Attempt to import Ultralytics YOLO API
try:
    from ultralytics import YOLO
except Exception as e:
    print("ERROR: Failed to import ultralytics. Install with `pip install ultralytics`.")
    raise

# -----------------------
# Top-level hyperparameters (edit these)
# -----------------------
CONFIDENCE_THRESH = 0.35   # detection confidence threshold (0.0 - 1.0)
MAX_DETECTIONS = 5         # maximum number of person detections to process (you asked for 1)
IOU_NMS = 0.45             # iou/nms threshold if you want to use in model.predict (left as var)
USE_USB_WEBCAM = True      # True = use USB webcam index, False = use built-in/default camera
WEBCAM_INDEX = 1           # USB webcam index (set to your USB device)
FRAME_WIDTH = None         # if None, use webcam default
FRAME_HEIGHT = None
TARGET_FPS = 24            # cap processing rate
DETECTION_MODEL_PATH = "yolo11n.pt"        # will download if not present via Ultralytics
POSE_MODEL_PATH = "yolo11n-pose.pt"        # will download if not present via Ultralytics
SHOW_FPS = True
PRINT_JSON_LOGS = True     # prints per-frame JSON to stdout
FOCUS_ONE_PERSON = True    # true = only process the top person detection
# Zone / control tuning
ZONE_LEFT_PCT = 0.40
ZONE_CENTER_PCT = 0.25
ZONE_ALPHA = 0.20
JUMP_FRAMES_UP = 6
JUMP_MIN_DELTA_PX = 6
JUMP_HOLD_FRAMES = 4
CENTER_VEL_WINDOW = 4
CENTER_VEL_MIN_FRAMES = 3
CENTER_MIN_UP_PX = 15
CENTER_MIN_ANGLE_DEG = 35
CENTER_KEYPOINTS = (0, 5, 6, 11, 12, 13, 14)
# -----------------------

# Keypoint indices (COCO-style ordering from user)
# 0 Nose, 1 Left Eye, 2 Right Eye, 3 Left Ear, 4 Right Ear,
# 5 Left Shoulder, 6 Right Shoulder, 7 Left Elbow, 8 Right Elbow,
# 9 Left Wrist, 10 Right Wrist, 11 Left Hip, 12 Right Hip,
# 13 Left Knee, 14 Right Knee, 15 Left Ankle, 16 Right Ankle
FACE_INDICES = {0, 1, 2, 3, 4}
BODY_INDICES = set(range(17)) - FACE_INDICES

# Skeleton connectivity for body (no face connections).
SKELETON = [
    (5, 6),          # shoulders
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10), # right arm
    (5, 11), (6, 12),# shoulders to hips
    (11, 12),        # hips
    (11, 13), (13, 15), # left leg
    (12, 14), (14, 16)  # right leg
]

# -----------------------
# Logging setup
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("yolo11n-pose")

@dataclass
class DetectionLog:
    frame_idx: int
    timestamp: float
    bbox: List[float]        # [x1, y1, x2, y2]
    confidence: float
    keypoints: List[Tuple[float, float, float]]  # list of (x, y, conf)
    frame_size: Tuple[int, int]

def choose_device_prefer_cuda():
    if torch.cuda.is_available():
        logger.info("CUDA available — using CUDA device.")
        return "cuda"
    else:
        logger.warning("CUDA NOT available — falling back to CPU. Performance will be slower.")
        return "cpu"

def scale_box_to_int(box):
    """Ensure box coords are ints and within image."""
    x1, y1, x2, y2 = box
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

def crop_with_padding(img, box):
    """Crop roi from image, ensure valid coordinates."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w - 1, x2); y2 = min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

def draw_bbox_and_label(img, box, label):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

def draw_keypoints_and_skeleton(img, keypoints, offset=(0,0)):
    # keypoints: list of (x, y, conf) in image coords
    ox, oy = offset
    for idx, (x, y, c) in enumerate(keypoints):
        if c <= 0 or idx in FACE_INDICES:
            continue
        cx, cy = int(round(x + ox)), int(round(y + oy))
        cv2.circle(img, (cx, cy), 3, (0, 255, 255), -1)

    # draw skeleton lines if SKELETON pairs are appropriate
    for (i, j) in SKELETON:
        if i < len(keypoints) and j < len(keypoints):
            xi, yi, ci = keypoints[i]
            xj, yj, cj = keypoints[j]
            if ci > 0 and cj > 0:
                p1 = (int(round(xi + ox)), int(round(yi + oy)))
                p2 = (int(round(xj + ox)), int(round(yj + oy)))
                cv2.line(img, p1, p2, (255, 0, 0), 2)

def draw_zones_overlay(frame, left_pct=0.30, center_pct=0.40, alpha=0.20):
    h, w = frame.shape[:2]
    left_w, right_start = compute_zone_edges(w, left_pct, center_pct)
    center_w = right_start - left_w

    overlay = frame.copy()
    # left and right zones (green)
    cv2.rectangle(overlay, (0, 0), (left_w, h), (0, 255, 0), -1)
    cv2.rectangle(overlay, (right_start, 0), (w, h), (0, 255, 0), -1)
    # center zone (red)
    cv2.rectangle(overlay, (left_w, 0), (right_start, h), (0, 0, 255), -1)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return left_w, right_start

def compute_zone_edges(width, left_pct, center_pct):
    left_w = int(round(width * left_pct))
    center_w = int(round(width * center_pct))
    right_start = left_w + center_w
    return left_w, right_start

def compute_center(keypoints):
    points = []
    for idx in CENTER_KEYPOINTS:
        if idx < len(keypoints):
            xk, yk, kc = keypoints[idx]
            if kc > 0:
                points.append((xk, yk))
    if not points:
        return None
    xs, ys = zip(*points)
    return (float(sum(xs) / len(xs)), float(sum(ys) / len(ys)))

@dataclass
class CVState:
    zone: Optional[str]
    jump_active: bool
    jump_vector: Optional[Tuple[float, float]]
    center: Optional[Tuple[float, float]]
    center_velocity: Optional[Tuple[float, float]]
    debug_frame: Optional[np.ndarray]
    timestamp: float
    has_person: bool

class CVController:
    def __init__(
        self,
        *,
        use_usb_webcam: bool = USE_USB_WEBCAM,
        webcam_index: int = WEBCAM_INDEX,
        target_fps: int = TARGET_FPS,
        left_pct: float = ZONE_LEFT_PCT,
        center_pct: float = ZONE_CENTER_PCT,
        debug_draw: bool = True,
        show_window: bool = False,
    ):
        self.device = choose_device_prefer_cuda()
        self.left_pct = left_pct
        self.center_pct = center_pct
        self.debug_draw = debug_draw
        self.show_window = show_window
        self.target_fps = target_fps
        self.use_usb_webcam = use_usb_webcam
        self.webcam_index = webcam_index
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._last_frame_time = 0.0

        self._jump_display_frames = 0
        self._center_history = deque(maxlen=CENTER_VEL_WINDOW)
        self._last_center_velocity = None
        self._last_jump_vector = None

        self._state = CVState(
            zone=None,
            jump_active=False,
            jump_vector=None,
            center=None,
            center_velocity=None,
            debug_frame=None,
            timestamp=time.time(),
            has_person=False,
        )

        logger.info("Loading detection model: %s", DETECTION_MODEL_PATH)
        try:
            self.det_model = YOLO(DETECTION_MODEL_PATH)
            try:
                self.det_model.to(self.device)
            except Exception:
                logger.debug("det_model.to(device) not supported by this ultralytics version; will pass device per-inference.")
        except Exception:
            logger.exception("Failed to load detection model. Ensure ultralytics can download or model path exists.")
            raise

        logger.info("Loading pose model: %s", POSE_MODEL_PATH)
        try:
            self.pose_model = YOLO(POSE_MODEL_PATH)
            try:
                self.pose_model.to(self.device)
            except Exception:
                logger.debug("pose_model.to(device) not supported by this ultralytics version; will pass device per-inference.")
        except Exception:
            logger.exception("Failed to load pose model. Ensure ultralytics can download or model path exists.")
            raise

        self.cap = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        cam_index = self.webcam_index if self.use_usb_webcam else 0
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open webcam index {cam_index}.")
        if FRAME_WIDTH and FRAME_HEIGHT:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if self.target_fps:
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.show_window:
            cv2.destroyAllWindows()

    def get_state(self) -> CVState:
        with self._lock:
            return CVState(
                zone=self._state.zone,
                jump_active=self._state.jump_active,
                jump_vector=self._state.jump_vector,
                center=self._state.center,
                center_velocity=self._state.center_velocity,
                debug_frame=None if self._state.debug_frame is None else self._state.debug_frame.copy(),
                timestamp=self._state.timestamp,
                has_person=self._state.has_person,
            )

    def _update_state(self, zone, jump_active, jump_vector, center, center_velocity, debug_frame, has_person):
        with self._lock:
            self._state = CVState(
                zone=zone,
                jump_active=jump_active,
                jump_vector=jump_vector,
                center=center,
                center_velocity=center_velocity,
                debug_frame=debug_frame,
                timestamp=time.time(),
                has_person=has_person,
            )

    def _loop(self):
        while not self._stop_event.is_set():
            if not self.cap:
                time.sleep(0.01)
                continue
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            now = time.time()
            if self._last_frame_time and self.target_fps:
                if (now - self._last_frame_time) < (1.0 / self.target_fps):
                    continue
            self._last_frame_time = now
            frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            if self.debug_draw:
                draw_zones_overlay(frame, self.left_pct, self.center_pct, ZONE_ALPHA)

            try:
                det_results = self.det_model.predict(
                    frame,
                    conf=CONFIDENCE_THRESH,
                    iou=IOU_NMS,
                    max_det=MAX_DETECTIONS,
                    verbose=False,
                )
            except TypeError:
                det_results = self.det_model(
                    frame,
                    conf=CONFIDENCE_THRESH,
                    iou=IOU_NMS,
                    max_det=MAX_DETECTIONS,
                    verbose=False,
                )
            except Exception:
                logger.exception("Detection model inference failed.")
                self._update_state(None, False, None, None, None, frame, False)
                continue

            det_res = det_results[0] if isinstance(det_results, (list, tuple)) else det_results
            boxes_xyxy = []
            scores = []
            class_ids = []
            try:
                if hasattr(det_res, "boxes"):
                    b = det_res.boxes
                    boxes_xyxy = b.xyxy.cpu().numpy() if hasattr(b, "xyxy") else b.xyxy.numpy()
                    scores = b.conf.cpu().numpy() if hasattr(b, "conf") else b.conf.numpy()
                    class_ids = b.cls.cpu().numpy().astype(int) if hasattr(b, "cls") else b.cls.numpy().astype(int)
                else:
                    boxes_xyxy = np.array(det_res.boxes.xyxy).astype(float)
                    scores = np.array(det_res.boxes.conf).astype(float)
                    class_ids = np.array(det_res.boxes.cls).astype(int)
            except Exception:
                boxes_xyxy = []
                scores = []
                class_ids = []

            person_indices = []
            try:
                names = self.det_model.model.names if hasattr(self.det_model, "model") and hasattr(self.det_model.model, "names") else None
            except Exception:
                names = None

            for i, cid in enumerate(class_ids if len(class_ids) else []):
                is_person = False
                if names is not None:
                    name = names[int(cid)]
                    if name.lower() == "person":
                        is_person = True
                else:
                    if int(cid) == 0:
                        is_person = True
                if is_person:
                    person_indices.append(i)

            person_boxes = []
            person_scores = []
            for idx in person_indices:
                person_boxes.append(boxes_xyxy[idx])
                person_scores.append(scores[idx] if len(scores) > idx else 0.0)

            if len(person_boxes) == 0:
                self._center_history.clear()
                self._last_center_velocity = None
                if self._jump_display_frames > 0:
                    self._jump_display_frames -= 1
                jump_active = self._jump_display_frames > 0
                jump_vector = self._last_jump_vector if jump_active else None
                self._update_state(None, jump_active, jump_vector, None, None, frame, False)
                if self.show_window:
                    cv2.imshow("YOLO11n Person+Pose", frame)
                    cv2.waitKey(1)
                continue

            def box_area(box):
                x1, y1, x2, y2 = box
                return max(0.0, x2 - x1) * max(0.0, y2 - y1)

            person_order = sorted(
                range(len(person_boxes)),
                key=lambda i: box_area(person_boxes[i]),
                reverse=True,
            )
            if FOCUS_ONE_PERSON:
                person_order = person_order[:1]
            else:
                person_order = person_order[:MAX_DETECTIONS]

            zone = None
            keypoints_out = []
            center = None
            center_velocity = None
            for p_i in person_order:
                box = person_boxes[p_i]
                conf = float(person_scores[p_i]) if len(person_scores) > p_i else 0.0
                x1, y1, x2, y2 = [int(round(x)) for x in box]
                crop = crop_with_padding(frame, (x1, y1, x2, y2))
                keypoints_out = []

                if crop is not None and crop.size != 0:
                    try:
                        pose_results = self.pose_model.predict(crop, conf=0.25, verbose=False)
                    except TypeError:
                        pose_results = self.pose_model(crop, conf=0.25, verbose=False)
                    except Exception:
                        pose_results = None

                    if pose_results is not None:
                        pr = pose_results[0] if isinstance(pose_results, (list, tuple)) else pose_results
                        extracted = False
                        try:
                            if hasattr(pr, "keypoints") and pr.keypoints is not None:
                                kps = pr.keypoints
                                if hasattr(kps, "xy"):
                                    kp_arr = kps.xy.cpu().numpy()
                                elif hasattr(kps, "xyxy"):
                                    kp_arr = kps.xyxy.cpu().numpy()
                                elif hasattr(kps, "data"):
                                    kp_arr = kps.data.cpu().numpy()
                                else:
                                    kp_arr = np.array(kps)
                                if kp_arr.ndim == 3:
                                    kp_arr = kp_arr[0]
                                for row in kp_arr:
                                    if len(row) >= 3:
                                        xk, yk, kc = float(row[0]), float(row[1]), float(row[2])
                                    elif len(row) == 2:
                                        xk, yk, kc = float(row[0]), float(row[1]), 1.0
                                    else:
                                        continue
                                    keypoints_out.append((xk + x1, yk + y1, kc))
                                extracted = True
                        except Exception:
                            extracted = False

                        if not extracted:
                            try:
                                if hasattr(pr, "keypoints") and hasattr(pr.keypoints, "numpy"):
                                    kp_arr = pr.keypoints.numpy()
                                    if kp_arr.ndim == 3:
                                        kp_arr = kp_arr[0]
                                    for row in kp_arr:
                                        if len(row) >= 3:
                                            xk, yk, kc = float(row[0]), float(row[1]), float(row[2])
                                        elif len(row) == 2:
                                            xk, yk, kc = float(row[0]), float(row[1]), 1.0
                                        else:
                                            continue
                                        keypoints_out.append((xk + x1, yk + y1, kc))
                                    extracted = True
                            except Exception:
                                extracted = False

                if self.debug_draw:
                    draw_bbox_and_label(frame, (x1, y1, x2, y2), f"person {conf:.2f}")
                    draw_keypoints_and_skeleton(frame, keypoints_out, offset=(0, 0))

                center = compute_center(keypoints_out)
                center_velocity = None
                jump_vector = None
                if center:
                    sample_time = time.time()
                    self._center_history.append((sample_time, center[0], center[1]))
                else:
                    self._center_history.clear()
                    self._last_center_velocity = None

                if len(self._center_history) >= CENTER_VEL_MIN_FRAMES:
                    t0, x0, y0 = self._center_history[0]
                    t1, x1, y1 = self._center_history[-1]
                    dt_window = t1 - t0
                    if dt_window > 0:
                        dx = x1 - x0
                        dy = y1 - y0
                        center_velocity = (dx / dt_window, dy / dt_window)
                        self._last_center_velocity = center_velocity
                        up_pixels = y0 - y1
                        if up_pixels > 0:
                            angle = math.degrees(math.atan2(up_pixels, max(1e-6, abs(dx))))
                            if up_pixels >= CENTER_MIN_UP_PX and angle >= CENTER_MIN_ANGLE_DEG:
                                jump_vector = center_velocity

                if jump_vector is not None:
                    self._jump_display_frames = JUMP_HOLD_FRAMES
                    self._last_jump_vector = jump_vector

            if self._jump_display_frames > 0:
                self._jump_display_frames -= 1
            jump_active = self._jump_display_frames > 0
            jump_vector = self._last_jump_vector if jump_active else None
            self._update_state(zone, jump_active, jump_vector, center, center_velocity, frame, True)

            if self.show_window:
                cv2.imshow("YOLO11n Person+Pose", frame)
                cv2.waitKey(1)

def run():
    device = choose_device_prefer_cuda()

    logger.info("Loading detection model: %s", DETECTION_MODEL_PATH)
    try:
        det_model = YOLO(DETECTION_MODEL_PATH)          # no device arg here
        # if YOLO object supports .to(), move model to device
        try:
            det_model.to(device)
        except Exception:
            logger.debug("det_model.to(device) not supported by this ultralytics version; will pass device per-inference.")
    except Exception as e:
        logger.exception("Failed to load detection model. Ensure ultralytics can download or model path exists.")
        raise

    logger.info("Loading pose model: %s", POSE_MODEL_PATH)
    try:
        pose_model = YOLO(POSE_MODEL_PATH)              # no device arg here
        try:
            pose_model.to(device)
        except Exception:
            logger.debug("pose_model.to(device) not supported by this ultralytics version; will pass device per-inference.")
    except Exception as e:
        logger.exception("Failed to load pose model. Ensure ultralytics can download or model path exists.")
        raise


    # Open webcam
    cam_index = WEBCAM_INDEX if USE_USB_WEBCAM else 0
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
    if not cap.isOpened():
        logger.error("Cannot open webcam index %s. Exiting.", cam_index)
        sys.exit(1)

    # Optionally set frame size
    if FRAME_WIDTH and FRAME_HEIGHT:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if TARGET_FPS:
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    frame_idx = 0
    t_prev = time.time()
    fps = 0.0
    last_frame_time = 0.0
    # Jump detection state
    jump_display_frames = 0
    center_history = deque(maxlen=CENTER_VEL_WINDOW)
    last_jump_vector = None
    # Jump detection state uses top-level JUMP_* constants
    logger.info("Starting webcam loop. Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from webcam. Exiting.")
                break
            now = time.time()
            if last_frame_time and (now - last_frame_time) < (1.0 / TARGET_FPS):
                continue
            last_frame_time = now
            frame = cv2.flip(frame, 1)

            frame_idx += 1
            h, w = frame.shape[:2]
            if SHOW_FPS:
                draw_zones_overlay(
                    frame,
                    left_pct=ZONE_LEFT_PCT,
                    center_pct=ZONE_CENTER_PCT,
                    alpha=ZONE_ALPHA,
                )

            # Run detector on the whole frame. Set save=False / verbose=False in predict if needed.
            # We use model.predict to allow passing conf and iou args (if API supports).
            try:
                # Use .predict / __call__ depending on the installed ultralytics version.
                det_results = det_model.predict(frame, conf=CONFIDENCE_THRESH, iou=IOU_NMS, max_det=MAX_DETECTIONS, verbose=False)
            except TypeError:
                # fallback call if signature differs
                det_results = det_model(frame, conf=CONFIDENCE_THRESH, iou=IOU_NMS, max_det=MAX_DETECTIONS, verbose=False)
            except Exception as e:
                logger.exception("Detection model inference failed on frame %d", frame_idx)
                break

            # det_results is usually a Results list; take first
            if isinstance(det_results, (list, tuple)):
                det_res = det_results[0]
            else:
                det_res = det_results

            # Extract boxes, confidences, class ids/names
            boxes_xyxy = []
            scores = []
            class_ids = []
            try:
                # Most ultralytics Results expose .boxes with .xyxy, .conf, .cls arrays
                if hasattr(det_res, "boxes"):
                    b = det_res.boxes
                    # .xyxy is tensor Nx4
                    boxes_xyxy = b.xyxy.cpu().numpy() if hasattr(b, "xyxy") else b.xyxy.numpy()
                    scores = b.conf.cpu().numpy() if hasattr(b, "conf") else b.conf.numpy()
                    class_ids = b.cls.cpu().numpy().astype(int) if hasattr(b, "cls") else b.cls.numpy().astype(int)
                else:
                    # alternative older structure: det_res.boxes.xyxy etc may live directly.
                    boxes_xyxy = np.array(det_res.boxes.xyxy).astype(float)
                    scores = np.array(det_res.boxes.conf).astype(float)
                    class_ids = np.array(det_res.boxes.cls).astype(int)
            except Exception as e:
                logger.warning("Could not parse detection outputs with standard attributes; falling back to results.boxes data if available.")
                try:
                    # try a more generic path
                    raw_boxes = getattr(det_res, "boxes", None)
                    if raw_boxes is None:
                        boxes_xyxy = []
                        scores = []
                        class_ids = []
                    else:
                        boxes_xyxy = np.array(raw_boxes.xyxy)
                        scores = np.array(raw_boxes.conf)
                        class_ids = np.array(raw_boxes.cls).astype(int)
                except Exception:
                    logger.exception("Failed to extract detection boxes/labels. Skipping frame.")
                    boxes_xyxy = []
                    scores = []
                    class_ids = []

            # Filter only 'person' class — Ultralytics uses COCO labels where person class id is usually 0.
            # Safer to check class name via model.names if available.
            person_indices = []
            try:
                names = det_model.model.names if hasattr(det_model, "model") and hasattr(det_model.model, "names") else None
            except Exception:
                names = None

            for i, cid in enumerate(class_ids if len(class_ids) else []):
                is_person = False
                if names is not None:
                    # compare name
                    name = names[int(cid)]
                    if name.lower() == "person":
                        is_person = True
                else:
                    # fallback: treat class id 0 as person
                    if int(cid) == 0:
                        is_person = True
                if is_person:
                    person_indices.append(i)

            # keep only person detections
            person_boxes = []
            person_scores = []
            for idx in person_indices:
                person_boxes.append(boxes_xyxy[idx])
                person_scores.append(scores[idx] if len(scores) > idx else 0.0)

            # If empty, nothing to do this frame
            detection_logs: List[DetectionLog] = []
            if len(person_boxes) == 0:
                center_history.clear()
                # update jump display decay even if no person
                if jump_display_frames > 0:
                    jump_display_frames -= 1
                # draw FPS only and continue
                # compute FPS
                t_now = time.time()
                dt = t_now - t_prev
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps
                t_prev = t_now
                if SHOW_FPS:
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                cv2.putText(frame, "ZONE: N/A", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(frame, f"JUMPING: {'YES' if jump_display_frames > 0 else 'NO'}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.imshow("YOLO11n Person+Pose", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quit requested by user.")
                    break
                continue

            def box_area(box):
                x1, y1, x2, y2 = box
                return max(0.0, x2 - x1) * max(0.0, y2 - y1)

            # Sort persons by area desc and respect MAX_DETECTIONS / FOCUS_ONE_PERSON
            person_order = sorted(
                range(len(person_boxes)),
                key=lambda i: box_area(person_boxes[i]),
                reverse=True,
            )
            if FOCUS_ONE_PERSON:
                person_order = person_order[:1]
            else:
                person_order = person_order[:MAX_DETECTIONS]

            zone = "N/A"
            for p_i in person_order:
                box = person_boxes[p_i]  # [x1,y1,x2,y2]
                conf = float(person_scores[p_i]) if len(person_scores) > p_i else 0.0
                x1, y1, x2, y2 = [int(round(x)) for x in box]
                # crop ROI
                crop = crop_with_padding(frame, (x1, y1, x2, y2))
                keypoints_out = []

                if crop is None or crop.size == 0:
                    logger.debug("Empty crop for bbox %s on frame %d; skipping pose.", box, frame_idx)
                else:
                    # Run pose model on the cropped image. Pose model will produce keypoints relative to the crop.
                    try:
                        pose_results = pose_model.predict(crop, conf=0.25, verbose=False)
                    except TypeError:
                        pose_results = pose_model(crop, conf=0.25, verbose=False)
                    except Exception as e:
                        logger.exception("Pose model failed on crop. Skipping pose for this bbox.")
                        pose_results = None

                    if pose_results is not None:
                        # extract keypoints from first result (if any)
                        pr = pose_results[0] if isinstance(pose_results, (list, tuple)) else pose_results
                        # try common attribute names
                        extracted = False
                        try:
                            if hasattr(pr, "keypoints") and pr.keypoints is not None:
                                # pr.keypoints likely an object with .xy or .data
                                kps = pr.keypoints
                                # Many versions: kps.xy, kps.xyxyn, or kps.data. Try several.
                                if hasattr(kps, "xy"):
                                    kp_arr = kps.xy.cpu().numpy()
                                elif hasattr(kps, "xyxy"):
                                    kp_arr = kps.xyxy.cpu().numpy()
                                elif hasattr(kps, "data"):
                                    kp_arr = kps.data.cpu().numpy()
                                else:
                                    kp_arr = np.array(kps)
                                # kp_arr expected shape (K, 3) or (N, K, 3); we take first instance if batched.
                                if kp_arr.ndim == 3:
                                    kp_arr = kp_arr[0]
                                # Many pose outputs are (num_keypoints, 3) with (x,y,confidence).
                                for row in kp_arr:
                                    if len(row) >= 3:
                                        xk, yk, kc = float(row[0]), float(row[1]), float(row[2])
                                    elif len(row) == 2:
                                        xk, yk, kc = float(row[0]), float(row[1]), 1.0
                                    else:
                                        continue
                                    # convert to original image coords by offsetting x1,y1
                                    keypoints_out.append((xk + x1, yk + y1, kc))
                                extracted = True
                        except Exception:
                            # fallback below
                            extracted = False

                        if not extracted:
                            # Try to parse common fields like pr.keypoints.xy or pr.keypoints.xyc
                            try:
                                # Some versions return pr.keypoints.xy or pr.keypoints.xyc
                                if hasattr(pr, "keypoints") and hasattr(pr.keypoints, "numpy"):
                                    kp_arr = pr.keypoints.numpy()
                                    if kp_arr.ndim == 3:
                                        kp_arr = kp_arr[0]
                                    for row in kp_arr:
                                        if len(row) >= 3:
                                            xk, yk, kc = float(row[0]), float(row[1]), float(row[2])
                                        elif len(row) == 2:
                                            xk, yk, kc = float(row[0]), float(row[1]), 1.0
                                        else:
                                            continue
                                        keypoints_out.append((xk + x1, yk + y1, kc))
                                    extracted = True
                            except Exception:
                                extracted = False

                        if not extracted:
                            # Last resort: try reading pr.masks or pr.boxes? Skip if cannot parse.
                            logger.debug("Could not extract keypoints from pose model result with standard attributes. Keypoints empty for this bbox.")
                            keypoints_out = []

                # Draw results on frame
                draw_bbox_and_label(frame, (x1, y1, x2, y2), f"person {conf:.2f}")
                draw_keypoints_and_skeleton(frame, keypoints_out, offset=(0,0))

                center = compute_center(keypoints_out)
                if center:
                    center_history.append((time.time(), center[0], center[1]))
                if len(center_history) >= CENTER_VEL_MIN_FRAMES:
                    t0, x0, y0 = center_history[0]
                    t1, x1c, y1c = center_history[-1]
                    dt_window = t1 - t0
                    if dt_window > 0:
                        dx = x1c - x0
                        dy = y1c - y0
                        up_pixels = y0 - y1c
                        if up_pixels > 0:
                            angle = math.degrees(math.atan2(up_pixels, max(1e-6, abs(dx))))
                            if up_pixels >= CENTER_MIN_UP_PX and angle >= CENTER_MIN_ANGLE_DEG:
                                jump_display_frames = JUMP_HOLD_FRAMES
                                last_jump_vector = (dx / dt_window, dy / dt_window)

                # Build detection log entry
                log_entry = DetectionLog(
                    frame_idx=frame_idx,
                    timestamp=time.time(),
                    bbox=[x1, y1, x2, y2],
                    confidence=conf,
                    keypoints=[(float(x), float(y), float(c)) for (x, y, c) in keypoints_out],
                    frame_size=(w, h),
                )
                detection_logs.append(log_entry)

            # compute FPS smoothing
            t_now = time.time()
            dt = t_now - t_prev
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps
            t_prev = t_now
            if SHOW_FPS:
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            if jump_display_frames > 0:
                jump_display_frames -= 1
            cv2.putText(frame, f"JUMPING: {'YES' if jump_display_frames > 0 else 'NO'}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # Show frame
            cv2.imshow("YOLO11n Person+Pose", frame)

            # Print logs as JSON lines
            if PRINT_JSON_LOGS and detection_logs:
                for dl in detection_logs:
                    # use asdict from dataclass
                    payload = asdict(dl)
                    print(json.dumps(payload), flush=True)

            # key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Quit requested by user.")
                break

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received — exiting.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Cleaned up and exiting.")

if __name__ == "__main__":
    run()
