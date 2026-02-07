"""
YOLO + SAM Segmentation Pipeline
---------------------------------
Uses YOLO for object detection to get bounding boxes,
then feeds those boxes into SAM for precise instance segmentation.
"""

import cv2
import math
import numpy as np
import time
from pathlib import Path
from PIL import Image as PilImage, ImageDraw, ImageFont
from ultralytics import YOLO, SAM

# Webcam selection (like cv_tester)
USE_USB_WEBCAM = True
WEBCAM_INDEX = 2

# Detect screen resolution once at module load
try:
    import tkinter as _tk
    _root = _tk.Tk()
    _root.withdraw()
    _SCREEN_W = _root.winfo_screenwidth()
    _SCREEN_H = _root.winfo_screenheight()
    _root.destroy()
except Exception:
    _SCREEN_W, _SCREEN_H = 1920, 1080

# ── Custom font (same as the game) ──────────────────────────────────
_FONT_PATH = str(
    Path(__file__).resolve().parent
    / "assets"
    / "Extraordinary Pixelvania - Free Asset Pack"
    / "Font"
    / "Extraordinary Font.ttf"
)


def _draw_text_pil(img, text, pos, font_size, color=(255, 255, 255)):
    """Draw *text* onto a BGR numpy image using the game's TTF font via PIL.

    Parameters
    ----------
    img : numpy array (BGR, uint8)  – modified **in-place**.
    text : str
    pos : (x, y) – top-left corner of the text.
    font_size : int – pixel height.
    color : (R, G, B) tuple  (note: RGB, NOT BGR).
    """
    pil_img = PilImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(_FONT_PATH, font_size)
    except OSError:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color)
    img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _get_text_size_pil(text, font_size):
    """Return (width, height) of *text* rendered at *font_size* using the game font."""
    try:
        font = ImageFont.truetype(_FONT_PATH, font_size)
    except OSError:
        font = ImageFont.load_default()
    bbox = font.getbbox(text)          # (left, top, right, bottom)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _resolve_webcam_index(webcam_index=None):
    if webcam_index is not None:
        return webcam_index
    return WEBCAM_INDEX if USE_USB_WEBCAM else 0


def capture_from_webcam(save_path: str = "capture.jpg", countdown: int = 10,
                        reference_image: str = None, pose_name: str = None,
                        ref_scale: float = 1.0, webcam_index: int | None = None) -> str:
    """
    Open the webcam, show a live preview with a countdown timer,
    then capture and save a frame.

    A YOLO detector finds the closest person each frame and the
    reference outline is scaled to fit that person's bounding box
    instead of using a fixed size.

    Args:
        save_path: Where to save the captured image.
        countdown: Seconds to wait before capturing.
        reference_image: Optional path to a reference/example image to show.
        pose_name: Optional name of the pose to display on screen.
        ref_scale: Extra scale multiplier applied on top of the bbox fit.

    Returns:
        The path to the saved image.
    """
    cap = cv2.VideoCapture(_resolve_webcam_index(webcam_index))
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    # Load reference image if provided
    ref_img = None
    if reference_image and Path(reference_image).exists():
        ref_img = cv2.imread(reference_image, cv2.IMREAD_UNCHANGED)
        print(f"[INFO] Showing reference: {reference_image}")

    # Load a lightweight YOLO model for person detection
    yolo_detect = YOLO("yolo11n.pt")

    label = f"Pose: {pose_name}" if pose_name else ""
    print(f"[INFO] Webcam opened. Capturing in {countdown} seconds...")
    start_time = time.time()

    # Create a fullscreen window
    window_name = "Webcam - Get Ready!"
    screen_w, screen_h = _SCREEN_W, _SCREEN_H
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, screen_w, screen_h)
    cv2.moveWindow(window_name, 0, 0)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to read from webcam.")

        # Mirror the frame so movements match the screen
        frame = cv2.flip(frame, 1)

        elapsed = time.time() - start_time
        remaining = max(0, countdown - elapsed)

        display = frame.copy()

        # ── Overlay reference outline onto detected person ──────────
        if ref_img is not None:
            # Detect people in the current frame
            det = yolo_detect(frame, conf=0.3, classes=[0], verbose=False)
            boxes = det[0].boxes
            bbox = None
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                idx = int(np.argmax(areas))
                bbox = xyxy[idx]  # x1, y1, x2, y2 of the closest person

            if bbox is not None:
                bx1, by1, bx2, by2 = bbox
                bbox_w = bx2 - bx1
                bbox_h = by2 - by1

                ref_h, ref_w = ref_img.shape[:2]

                # Scale outline to fit the person bbox (match height, preserve aspect ratio)
                scale = (bbox_h / max(ref_h, 1)) * ref_scale
                new_w = int(ref_w * scale)
                new_h = int(ref_h * scale)

                if new_w > 0 and new_h > 0:
                    ref_resized = cv2.resize(ref_img, (new_w, new_h))
                    if len(ref_resized.shape) == 2:
                        ref_resized = cv2.cvtColor(ref_resized, cv2.COLOR_GRAY2BGR)

                    # Find the horizontal center of the visible outline content
                    # so the overlay aligns with the person, not the image center
                    if ref_resized.shape[2] == 4:
                        vis_gray = cv2.cvtColor(ref_resized[:, :, :3], cv2.COLOR_BGR2GRAY)
                    else:
                        vis_gray = cv2.cvtColor(ref_resized, cv2.COLOR_BGR2GRAY)
                    cols = np.where(vis_gray > 128)
                    if len(cols[1]) > 0:
                        content_cx = int(np.mean(cols[1]))
                    else:
                        content_cx = new_w // 2

                    # Align the outline's content center with the bbox center,
                    # anchored at the bottom of the bbox
                    cx = int((bx1 + bx2) / 2)
                    x_off = cx - content_cx
                    y_off = int(by2) - new_h

                    # Clamp to frame bounds
                    rx1 = max(0, -x_off)
                    ry1 = max(0, -y_off)
                    fx1 = max(0, x_off)
                    fy1 = max(0, y_off)
                    rw = min(new_w - rx1, frame.shape[1] - fx1)
                    rh = min(new_h - ry1, frame.shape[0] - fy1)

                    if rw > 0 and rh > 0:
                        ref_crop = ref_resized[ry1:ry1 + rh, rx1:rx1 + rw]
                        roi = display[fy1:fy1 + rh, fx1:fx1 + rw]

                        if ref_crop.shape[2] == 4:
                            gray = cv2.cvtColor(ref_crop[:, :, :3], cv2.COLOR_BGR2GRAY)
                            rgb = ref_crop[:, :, :3]
                        else:
                            gray = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY)
                            rgb = ref_crop

                        # Build a black outline around the white region
                        white_binary = (gray > 128).astype(np.uint8)
                        # Dilate the white region then subtract to get an edge ring
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                        dilated = cv2.dilate(white_binary, kernel, iterations=1)
                        outline_mask = ((dilated - white_binary) > 0).astype(np.float32)[:, :, np.newaxis]

                        white_mask = white_binary.astype(np.float32)[:, :, np.newaxis]
                        opacity = 0.35
                        blended = (white_mask * (opacity * rgb + (1 - opacity) * roi)
                                   + (1 - white_mask) * roi).astype(np.uint8)
                        # Paint the outline ring black
                        blended = (outline_mask * np.zeros_like(roi)
                                   + (1 - outline_mask) * blended).astype(np.uint8)
                        display[fy1:fy1 + rh, fx1:fx1 + rw] = blended

        # Resize display to fill the entire screen, then draw text at screen res
        display = cv2.resize(display, (screen_w, screen_h))

        hud_size = max(3, screen_h // 180)
        if label:
            _draw_text_pil(display, label.upper(), (20, int(screen_h * 0.02)),
                           hud_size, color=(255, 176, 193))

        # Draw countdown (show whole seconds)
        secs = str(math.ceil(remaining)) if remaining > 0 else "0"
        _draw_text_pil(display, secs, (20, int(screen_h * 0.06)),
                       hud_size, color=(255, 80, 80))

        cv2.imshow(window_name, display)

        if remaining <= 0:
            # Capture this frame
            cv2.imwrite(save_path, frame)
            print(f"[INFO] Captured image saved to: {save_path}")
            break

        # Allow quitting early with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            raise RuntimeError("Capture cancelled by user.")

    cap.release()
    cv2.destroyAllWindows()
    return save_path


# Pose sequence for the 3-capture mode
_OUTLINE_DIR = Path(__file__).resolve().parent / "assets" / "outline_images"
POSES = [
    {"name": "Stand",  "save_path": "capture_stand.png", "reference": str(_OUTLINE_DIR / "stand_mask.png"), "ref_scale": 1.0},
    {"name": "Jump",   "save_path": "capture_jump.png",  "reference": str(_OUTLINE_DIR / "jump_mask.png"),  "ref_scale": 1.0},
    {"name": "Left",   "save_path": "capture_left.png",  "reference": str(_OUTLINE_DIR / "left_mask.png"),  "ref_scale": 1.0},
]


def capture_all_poses(countdown: int = 5, webcam_index: int | None = None) -> list:
    """Capture 3 images in sequence: stand, jump, and left poses.

    Uses a single fullscreen window and webcam for all poses.
    """
    results, window_name, screen_w, screen_h = _capture_poses_single_window(
        POSES, countdown, webcam_index=webcam_index
    )
    _fade_out_window(window_name, screen_w, screen_h, duration=0.4)
    return results


def _capture_poses_single_window(poses, countdown, save_path_override=None, webcam_index: int | None = None):
    """Capture multiple poses using one persistent fullscreen window.

    Args:
        poses: List of pose dicts (name, save_path, reference, ref_scale).
        countdown: Seconds per pose countdown.
        save_path_override: If a callable, called with pose dict to get save path.
                            Otherwise uses pose["save_path"].

    Returns:
        Tuple of (results, window_name, screen_w, screen_h).
        results is a list of (pose_name, saved_path) tuples.
        The window is left open so the caller can show status/fade.
    """
    cap = cv2.VideoCapture(_resolve_webcam_index(webcam_index))
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    window_name = "Webcam - Get Ready!"
    screen_w, screen_h = _SCREEN_W, _SCREEN_H
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, screen_w, screen_h)
    cv2.moveWindow(window_name, 0, 0)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    yolo_detect = YOLO("yolo11n.pt")
    screen_w, screen_h = _SCREEN_W, _SCREEN_H

    results = []

    # ── Intro screen: two-line instructional blurb ─────────────────
    intro_duration = 3.0  # seconds
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    canvas[:] = (30, 18, 15)

    line1 = "CREATE YOUR AVATAR"
    line2 = "COPY THE POSES!"
    intro_size = max(4, screen_h // 120)

    w1, h1 = _get_text_size_pil(line1, intro_size)
    w2, h2 = _get_text_size_pil(line2, intro_size)
    gap = int(screen_h * 0.04)
    total_h = h1 + gap + h2
    y1 = (screen_h - total_h) // 2
    y2 = y1 + h1 + gap

    _draw_text_pil(canvas, line1, ((screen_w - w1) // 2, y1),
                   intro_size, color=(255, 176, 193))
    _draw_text_pil(canvas, line2, ((screen_w - w2) // 2, y2),
                   intro_size, color=(255, 176, 193))

    # Hold the intro, then fade it out into the live camera
    intro_start = time.time()
    while time.time() - intro_start < intro_duration:
        cv2.imshow(window_name, canvas)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            raise RuntimeError("Capture cancelled by user.")

    # Brief fade from intro to camera
    fade_steps = 10
    for step in range(1, fade_steps + 1):
        alpha = step / fade_steps
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            cam_screen = cv2.resize(frame, (screen_w, screen_h))
            blended = cv2.addWeighted(canvas, 1 - alpha, cam_screen, alpha, 0)
            cv2.imshow(window_name, blended)
            cv2.waitKey(50)

    for pose_idx, pose in enumerate(poses):
        pose_name = pose["name"]
        ref_scale = pose.get("ref_scale", 1.0)

        if callable(save_path_override):
            save_path = save_path_override(pose)
        else:
            save_path = pose["save_path"]

        # Load reference image for this pose
        ref_img = None
        ref_path = pose.get("reference")
        if ref_path and Path(ref_path).exists():
            ref_img = cv2.imread(ref_path, cv2.IMREAD_UNCHANGED)

        label = f"Pose: {pose_name}"
        print(f"\n{'=' * 50}")
        print(f"[{pose_idx + 1}/{len(poses)}] Get ready for: {pose_name}")
        print(f"{'=' * 50}")

        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to read from webcam.")

            frame = cv2.flip(frame, 1)
            elapsed = time.time() - start_time
            remaining = max(0, countdown - elapsed)
            display = frame.copy()

            # ── Overlay reference outline onto detected person ──────
            if ref_img is not None:
                det = yolo_detect(frame, conf=0.3, classes=[0], verbose=False)
                boxes = det[0].boxes
                bbox = None
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                    idx = int(np.argmax(areas))
                    bbox = xyxy[idx]

                if bbox is not None:
                    bx1, by1, bx2, by2 = bbox
                    bbox_h = by2 - by1
                    ref_h, ref_w = ref_img.shape[:2]

                    scale = (bbox_h / max(ref_h, 1)) * ref_scale
                    new_w = int(ref_w * scale)
                    new_h = int(ref_h * scale)

                    if new_w > 0 and new_h > 0:
                        ref_resized = cv2.resize(ref_img, (new_w, new_h))
                        if len(ref_resized.shape) == 2:
                            ref_resized = cv2.cvtColor(ref_resized, cv2.COLOR_GRAY2BGR)

                        if ref_resized.shape[2] == 4:
                            vis_gray = cv2.cvtColor(ref_resized[:, :, :3], cv2.COLOR_BGR2GRAY)
                        else:
                            vis_gray = cv2.cvtColor(ref_resized, cv2.COLOR_BGR2GRAY)
                        cols = np.where(vis_gray > 128)
                        content_cx = int(np.mean(cols[1])) if len(cols[1]) > 0 else new_w // 2

                        cx = int((bx1 + bx2) / 2)
                        x_off = cx - content_cx
                        y_off = int(by2) - new_h

                        rx1 = max(0, -x_off)
                        ry1 = max(0, -y_off)
                        fx1 = max(0, x_off)
                        fy1 = max(0, y_off)
                        rw = min(new_w - rx1, frame.shape[1] - fx1)
                        rh = min(new_h - ry1, frame.shape[0] - fy1)

                        if rw > 0 and rh > 0:
                            ref_crop = ref_resized[ry1:ry1 + rh, rx1:rx1 + rw]
                            roi = display[fy1:fy1 + rh, fx1:fx1 + rw]

                            if ref_crop.shape[2] == 4:
                                gray = cv2.cvtColor(ref_crop[:, :, :3], cv2.COLOR_BGR2GRAY)
                                rgb = ref_crop[:, :, :3]
                            else:
                                gray = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY)
                                rgb = ref_crop

                            white_binary = (gray > 128).astype(np.uint8)
                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                            dilated = cv2.dilate(white_binary, kernel, iterations=1)
                            outline_mask = ((dilated - white_binary) > 0).astype(np.float32)[:, :, np.newaxis]

                            white_mask = white_binary.astype(np.float32)[:, :, np.newaxis]
                            opacity = 0.35
                            blended = (white_mask * (opacity * rgb + (1 - opacity) * roi)
                                       + (1 - white_mask) * roi).astype(np.uint8)
                            blended = (outline_mask * np.zeros_like(roi)
                                       + (1 - outline_mask) * blended).astype(np.uint8)
                            display[fy1:fy1 + rh, fx1:fx1 + rw] = blended

            # ── Resize then draw HUD text at screen resolution ─────
            display = cv2.resize(display, (screen_w, screen_h))

            hud_size = max(3, screen_h // 180)
            _draw_text_pil(display, label.upper(), (20, int(screen_h * 0.02)),
                           hud_size, color=(255, 176, 193))

            secs = str(math.ceil(remaining)) if remaining > 0 else "0"
            _draw_text_pil(display, secs, (20, int(screen_h * 0.06)),
                           hud_size, color=(255, 80, 80))

            cv2.imshow(window_name, display)

            if remaining <= 0:
                cv2.imwrite(save_path, frame)
                print(f"[INFO] Captured image saved to: {save_path}")
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                raise RuntimeError("Capture cancelled by user.")

        results.append((pose_name, save_path))

        # Brief pause between poses (stay in the window)
        if pose_idx < len(poses) - 1:
            pause_end = time.time() + 1.0
            while time.time() < pause_end:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    frame = cv2.resize(frame, (screen_w, screen_h))
                    cv2.imshow(window_name, frame)
                cv2.waitKey(30)

    cap.release()
    # Don't destroy the window yet — caller may want to show a processing screen
    return results, window_name, screen_w, screen_h


def _show_status_on_window(window_name, screen_w, screen_h, message, sub_message=""):
    """Display a status message on the fullscreen CV window."""
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    canvas[:] = (30, 18, 15)  # dark background matching game

    # Main message — use game font via PIL (uppercase — only working glyphs)
    main_size = max(4, screen_h // 120)
    msg_upper = message.upper()
    tw, th = _get_text_size_pil(msg_upper, main_size)
    x = (screen_w - tw) // 2
    y = (screen_h - th) // 2
    _draw_text_pil(canvas, msg_upper, (x, y), main_size, color=(255, 176, 193))

    # Sub-message (smaller)
    if sub_message:
        sub_size = max(3, main_size * 2 // 3)
        sub_upper = sub_message.upper()
        sw, sh = _get_text_size_pil(sub_upper, sub_size)
        _draw_text_pil(canvas, sub_upper,
                       ((screen_w - sw) // 2, y + th + 20),
                       sub_size, color=(200, 200, 200))

    cv2.imshow(window_name, canvas)
    cv2.waitKey(1)


def _fade_out_window(window_name, screen_w, screen_h, duration=0.5, destroy=True):
    """Fade the window to black, optionally destroying it afterwards."""
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    canvas[:] = (30, 18, 15)
    steps = 15
    for i in range(steps + 1):
        alpha = 1.0 - (i / steps)
        faded = (canvas * alpha).astype(np.uint8)
        cv2.imshow(window_name, faded)
        cv2.waitKey(int(duration / steps * 1000))
    if destroy:
        cv2.destroyAllWindows()


def run_yolo_sam_pipeline(
    image_path: str,
    yolo_model: str = "yolo11n.pt",
    sam_model: str = "sam2_b.pt",
    conf_threshold: float = 0.25,
    output_dir: str = "output",
    grid_size: int = 40,
):
    """
    Run YOLO detection followed by SAM segmentation.

    Args:
        image_path: Path to the input image.
        yolo_model: YOLO model weights (downloaded automatically).
        sam_model: SAM model weights (downloaded automatically).
        conf_threshold: Confidence threshold for YOLO detections.
        output_dir: Directory to save results.
        grid_size: Number of grid cells along each axis. Lower = more pixelated.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # ── 1. Load models ──────────────────────────────────────────────
    print(f"[INFO] Loading YOLO model: {yolo_model}")
    yolo = YOLO(yolo_model)

    print(f"[INFO] Loading SAM model: {sam_model}")
    sam = SAM(sam_model)

    # ── 2. Run YOLO detection (people only) ────────────────────────
    print(f"[INFO] Running YOLO detection on: {image_path}")
    det_results = yolo(image_path, conf=conf_threshold, classes=[0], verbose=False)
    det_result = det_results[0]

    boxes = det_result.boxes
    if boxes is None or len(boxes) == 0:
        print("[WARN] No objects detected by YOLO.")
        return

    # Extract bounding boxes as xyxy numpy array
    all_bboxes = boxes.xyxy.cpu().numpy()            # (N, 4) — x1, y1, x2, y2
    all_confs = boxes.conf.cpu().numpy()             # (N,)
    all_cls_ids = boxes.cls.cpu().numpy().astype(int) # (N,)
    class_names = det_result.names                   # {id: name, ...}

    # Pick the closest person (largest bounding box area)
    areas = (all_bboxes[:, 2] - all_bboxes[:, 0]) * (all_bboxes[:, 3] - all_bboxes[:, 1])
    closest_idx = int(np.argmax(areas))

    bboxes = all_bboxes[closest_idx:closest_idx + 1]   # keep (1, 4) shape
    confs = all_confs[closest_idx:closest_idx + 1]
    cls_ids = all_cls_ids[closest_idx:closest_idx + 1]

    print(f"[INFO] Detected {len(all_bboxes)} person(s), keeping closest (largest bbox):")
    box, conf, cls_id = bboxes[0], confs[0], cls_ids[0]
    print(f"  {class_names[cls_id]}  conf={conf:.2f}  "
          f"bbox=({box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f})")

    # ── 3. Run SAM segmentation with YOLO bounding boxes ───────────
    print("[INFO] Running SAM segmentation on detected bounding boxes...")
    sam_results = sam(image_path, bboxes=bboxes, verbose=False)
    sam_result = sam_results[0]

    # ── 4. Isolate & pixelate masked person ───────────────────────
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    if sam_result.masks is not None:
        masks = sam_result.masks.data.cpu().numpy()  # (N, H, W)

        # Use the first (only) mask
        binary_mask = (masks[0] > 0.5).astype(np.uint8)

        # Extract the person pixels; transparent (black) background
        isolated = cv2.bitwise_and(image, image, mask=binary_mask)

        print(f"[INFO] Pixelating with {grid_size}×{grid_size} grid "
              f"(cell size ≈ {w // grid_size}×{h // grid_size} px)")

        # Pixelate using a grid that divides the image into grid_size cells
        pixelated = np.zeros_like(image)

        # Compute cell boundaries that fully tile the image (handles uneven division)
        row_edges = np.linspace(0, h, grid_size + 1, dtype=int)
        col_edges = np.linspace(0, w, grid_size + 1, dtype=int)

        for r in range(grid_size):
            for c in range(grid_size):
                y1, y2 = row_edges[r], row_edges[r + 1]
                x1, x2 = col_edges[c], col_edges[c + 1]
                if y1 == y2 or x1 == x2:
                    continue

                # Only fill if any masked pixel is in this cell
                cell_mask = binary_mask[y1:y2, x1:x2]
                if cell_mask.any():
                    cell_pixels = isolated[y1:y2, x1:x2]
                    masked_pixels = cell_pixels[cell_mask == 1]
                    avg_colour = masked_pixels.mean(axis=0).astype(np.uint8)
                    pixelated[y1:y2, x1:x2] = avg_colour

        # Save pixelated isolated person with transparent background
        stem = Path(image_path).stem

        # Build an alpha mask: 255 for any grid cell that was filled, 0 elsewhere
        alpha = np.zeros((h, w), dtype=np.uint8)
        for r in range(grid_size):
            for c in range(grid_size):
                y1, y2 = row_edges[r], row_edges[r + 1]
                x1, x2 = col_edges[c], col_edges[c + 1]
                if y1 == y2 or x1 == x2:
                    continue
                if binary_mask[y1:y2, x1:x2].any():
                    alpha[y1:y2, x1:x2] = 255

        # Merge BGR + Alpha into a 4-channel BGRA image
        pixelated_rgba = cv2.merge([pixelated[:, :, 0],
                                     pixelated[:, :, 1],
                                     pixelated[:, :, 2],
                                     alpha])

        # Crop to bounding box of non-transparent pixels
        ys, xs = np.where(alpha > 0)
        crop_y1, crop_y2 = ys.min(), ys.max() + 1
        crop_x1, crop_x2 = xs.min(), xs.max() + 1
        pixelated_rgba = pixelated_rgba[crop_y1:crop_y2, crop_x1:crop_x2]

        out_file = output_path / f"{stem}_pixelated.png"
        cv2.imwrite(str(out_file), pixelated_rgba)
        print(f"[INFO] Saved pixelated mask to: {out_file}")

        # Also save the clean binary mask (cropped to same region)
        mask_file = output_path / f"{stem}_mask.png"
        cv2.imwrite(str(mask_file), (binary_mask * 255)[crop_y1:crop_y2, crop_x1:crop_x2])
        print(f"[INFO] Saved binary mask to:    {mask_file}")

        return pixelated
    else:
        print("[WARN] SAM produced no masks.")
        return None


# ── Mapping from pose name to the sprite filename the game expects ──
_POSE_TO_SPRITE = {
    "Stand": "stand_still.png",
    "Jump":  "jump.png",
    "Left":  "move_left.png",
}


def user_has_photos(player_dir: str) -> bool:
    """Return True if *player_dir* already contains all 3 required sprite files."""
    d = Path(player_dir)
    return all((d / fname).exists() for fname in _POSE_TO_SPRITE.values())


def capture_and_process_for_user(
    player_dir: str,
    countdown: int = 10,
    yolo_model: str = "yolo11n.pt",
    sam_model: str = "sam2_b.pt",
    conf_threshold: float = 0.25,
    grid_size: int = 40,
    webcam_index: int | None = None,
) -> bool:
    """
    Run the full capture → segment → pixelate pipeline for a player.

    Captures 3 poses via webcam, runs YOLO+SAM on each, and saves the
    resulting pixelated transparent PNGs into *player_dir* with the
    filenames the game expects (stand_still.png, jump.png, move_left.png).

    Args:
        player_dir:     Directory to save the final sprites into.
        countdown:      Seconds per pose countdown.
        yolo_model:     YOLO model weights filename.
        sam_model:      SAM model weights filename.
        conf_threshold: YOLO confidence threshold.
        grid_size:      Pixelation grid size (lower = chunkier pixels).

    Returns:
        True if all 3 sprites were created successfully, False otherwise.
    """
    import shutil

    out_dir = Path(player_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use a temporary working directory for intermediate files
    tmp_dir = out_dir / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    # Capture all 3 poses in a single fullscreen window
    captured, window_name, screen_w, screen_h = _capture_poses_single_window(
        POSES,
        countdown,
        save_path_override=lambda pose: str(tmp_dir / f"capture_{pose['name'].lower()}.png"),
        webcam_index=webcam_index,
    )

    # Show processing status on the same fullscreen window
    _show_status_on_window(window_name, screen_w, screen_h,
                           "PROCESSING...", "Creating your character")

    # Process each captured image through YOLO+SAM
    success = True
    for i, (pose_name, img_path) in enumerate(captured):
        _show_status_on_window(window_name, screen_w, screen_h,
                               "PROCESSING...",
                               f"{pose_name}")
        print(f"\n[INFO] Processing {pose_name} pose...")
        result = run_yolo_sam_pipeline(
            image_path=img_path,
            yolo_model=yolo_model,
            sam_model=sam_model,
            conf_threshold=conf_threshold,
            output_dir=str(tmp_dir),
            grid_size=grid_size,
        )
        if result is None:
            print(f"[WARN] Pipeline failed for {pose_name} pose.")
            success = False
            continue

        # The pipeline saves "<stem>_pixelated.png" in tmp_dir
        stem = Path(img_path).stem
        pixelated_file = tmp_dir / f"{stem}_pixelated.png"
        sprite_name = _POSE_TO_SPRITE.get(pose_name)
        if pixelated_file.exists() and sprite_name:
            dest = out_dir / sprite_name
            shutil.copy2(str(pixelated_file), str(dest))
            print(f"[INFO] Saved sprite: {dest}")
        else:
            print(f"[WARN] Expected pixelated file not found: {pixelated_file}")
            success = False

    # Clean up temporary files
    shutil.rmtree(str(tmp_dir), ignore_errors=True)

    if success:
        _show_status_on_window(window_name, screen_w, screen_h,
                               "DONE!", "Loading game...")
        time.sleep(0.8)
        print(f"\n[INFO] All sprites saved to {out_dir}")
    else:
        _show_status_on_window(window_name, screen_w, screen_h,
                               "DONE", "Some poses failed — using defaults")
        time.sleep(1.0)
        print(f"\n[WARN] Some sprites could not be created.")

    # Smooth fade to black — keep the window open so the caller can
    # restore pygame behind it before destroying, avoiding a desktop flash.
    _fade_out_window(window_name, screen_w, screen_h, duration=0.4, destroy=False)

    return success


# ── CLI entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="YOLO + SAM: detect objects then segment them."
    )
    parser.add_argument("image", nargs="?", default=None,
                        help="Path to the input image. Omit to use webcam.")
    parser.add_argument("--webcam", action="store_true",
                        help="Capture from webcam with a 5-second countdown.")
    parser.add_argument("--capture-poses", action="store_true",
                        help="Capture all 3 poses (stand, jump, left) with webcam.")
    parser.add_argument("--yolo-model", default="yolo11n.pt",
                        help="YOLO model name (default: yolo11n.pt)")
    parser.add_argument("--sam-model", default="sam2_b.pt",
                        help="SAM model name (default: sam2_b.pt)")
    parser.add_argument("--webcam-index", type=int, default=None,
                        help="Webcam index to use (default: 0 or WEBCAM_INDEX)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="YOLO confidence threshold (default: 0.25)")
    parser.add_argument("--output-dir", default="output",
                        help="Output directory (default: output/)")
    parser.add_argument("--grid-size", type=int, default=40,
                        help="Number of grid cells per axis, lower = more pixelated (default: 40)")
    args = parser.parse_args()

    # Determine image source
    if args.capture_poses:
        # Capture all 3 poses then process each
        captured = capture_all_poses(countdown=10, webcam_index=args.webcam_index)
        for _name, img_path in captured:
            run_yolo_sam_pipeline(
                image_path=img_path,
                yolo_model=args.yolo_model,
                sam_model=args.sam_model,
                conf_threshold=args.conf,
                output_dir=args.output_dir,
                grid_size=args.grid_size,
            )
    else:
        if args.webcam or args.image is None:
            image_path = capture_from_webcam(
                save_path="capture.jpg",
                countdown=10,
                webcam_index=args.webcam_index,
            )
        else:
            image_path = args.image

        run_yolo_sam_pipeline(
            image_path=image_path,
            yolo_model=args.yolo_model,
            sam_model=args.sam_model,
            conf_threshold=args.conf,
            output_dir=args.output_dir,
            grid_size=args.grid_size,
        )
