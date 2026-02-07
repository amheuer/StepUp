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
from ultralytics import YOLO, SAM


def capture_from_webcam(save_path: str = "capture.jpg", countdown: int = 10,
                        reference_image: str = None, pose_name: str = None,
                        ref_scale: float = 1.0) -> str:
    """
    Open the webcam, show a live preview with a countdown timer,
    then capture and save a frame.

    Args:
        save_path: Where to save the captured image.
        countdown: Seconds to wait before capturing.
        reference_image: Optional path to a reference/example image to show.
        pose_name: Optional name of the pose to display on screen.
        ref_scale: Scale multiplier for the reference image (1.0 = full frame height).

    Returns:
        The path to the saved image.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    # Load reference image if provided
    ref_img = None
    if reference_image and Path(reference_image).exists():
        ref_img = cv2.imread(reference_image, cv2.IMREAD_UNCHANGED)
        print(f"[INFO] Showing reference: {reference_image}")

    label = f"Pose: {pose_name}" if pose_name else ""
    print(f"[INFO] Webcam opened. Capturing in {countdown} seconds...")
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to read from webcam.")

        elapsed = time.time() - start_time
        remaining = max(0, countdown - elapsed)

        display = frame.copy()

        # Draw the reference image centered and translucent
        if ref_img is not None:
            ref_h, ref_w = ref_img.shape[:2]
            # Scale reference to fill the frame height, adjusted by ref_scale
            scale = frame.shape[0] / max(ref_h, 1) * ref_scale
            new_w = int(ref_w * scale)
            new_h = int(ref_h * scale)
            if new_w > 0 and new_h > 0:
                ref_resized = cv2.resize(ref_img, (new_w, new_h))
                # Convert grayscale to BGR if needed
                if len(ref_resized.shape) == 2:
                    ref_resized = cv2.cvtColor(ref_resized, cv2.COLOR_GRAY2BGR)

                # Center horizontally, anchor bottom to screen edge
                x_off = (frame.shape[1] - new_w) // 2
                y_off = frame.shape[0] - new_h

                # Clamp to frame bounds
                rx1 = max(0, -x_off)
                ry1 = max(0, -y_off)
                fx1 = max(0, x_off)
                fy1 = max(0, y_off)
                rw = min(new_w - rx1, frame.shape[1] - fx1)
                rh = min(new_h - ry1, frame.shape[0] - fy1)

                ref_crop = ref_resized[ry1:ry1 + rh, rx1:rx1 + rw]
                roi = display[fy1:fy1 + rh, fx1:fx1 + rw]

                # Build per-pixel mask: white pixels are translucent, black stays black
                if ref_crop.shape[2] == 4:
                    # Use alpha channel AND brightness
                    gray = cv2.cvtColor(ref_crop[:, :, :3], cv2.COLOR_BGR2GRAY)
                    rgb = ref_crop[:, :, :3]
                else:
                    gray = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY)
                    rgb = ref_crop

                # White regions (brightness > 128) blend translucently; black regions stay as webcam
                white_mask = (gray > 128).astype(np.float32)[:, :, np.newaxis]
                opacity = 0.35
                blended = (white_mask * (opacity * rgb + (1 - opacity) * roi)
                           + (1 - white_mask) * roi).astype(np.uint8)
                display[fy1:fy1 + rh, fx1:fx1 + rw] = blended

        # Draw pose name
        if label:
            cv2.putText(display, label, (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

        # Draw countdown (show whole seconds)
        text = str(math.ceil(remaining)) if remaining > 0 else "0"
        cv2.putText(display, text, (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 4, cv2.LINE_AA)
        cv2.imshow("Webcam - Get Ready!", display)

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
POSES = [
    {"name": "Stand",  "save_path": "capture_stand.png", "reference": "output/stand_mask.png", "ref_scale": 1.0},
    {"name": "Jump",   "save_path": "capture_jump.png",  "reference": "output/jump_mask.png",  "ref_scale": 1.0},
    {"name": "Left",   "save_path": "capture_left.png",  "reference": "output/left_mask.png",  "ref_scale": 0.75},
]


def capture_all_poses(countdown: int = 5) -> list:
    """Capture 3 images in sequence: stand, jump, and left poses."""
    captured = []
    for i, pose in enumerate(POSES):
        print(f"\n{'=' * 50}")
        print(f"[{i + 1}/{len(POSES)}] Get ready for: {pose['name']}")
        print(f"{'=' * 50}")
        path = capture_from_webcam(
            save_path=pose["save_path"],
            countdown=countdown,
            reference_image=pose["reference"],
            pose_name=pose["name"],
            ref_scale=pose.get("ref_scale", 1.0),
        )
        captured.append(path)
        # Brief pause between captures
        if i < len(POSES) - 1:
            time.sleep(1)
    return captured


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
        captured = capture_all_poses(countdown=5)
        for img_path in captured:
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
            image_path = capture_from_webcam(save_path="capture.jpg", countdown=5)
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
