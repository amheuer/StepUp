"""
YOLO + SAM Segmentation Pipeline
---------------------------------
Uses YOLO for object detection to get bounding boxes,
then feeds those boxes into SAM for precise instance segmentation.
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO, SAM


def run_yolo_sam_pipeline(
    image_path: str,
    yolo_model: str = "yolo11n.pt",
    sam_model: str = "sam2_b.pt",
    conf_threshold: float = 0.25,
    output_dir: str = "output",
    grid_size: int = 32,
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

        # Pixelate using a grid that divides the image into grid_size cells
        pixelated = np.zeros_like(image)
        cell_h = max(1, h // grid_size)
        cell_w = max(1, w // grid_size)

        for row in range(grid_size):
            for col in range(grid_size):
                y1 = row * cell_h
                x1 = col * cell_w
                y2 = min(y1 + cell_h, h)
                x2 = min(x1 + cell_w, w)

                # Only fill if any masked pixel is in this cell
                cell_mask = binary_mask[y1:y2, x1:x2]
                if cell_mask.any():
                    cell_pixels = isolated[y1:y2, x1:x2]
                    masked_pixels = cell_pixels[cell_mask == 1]
                    avg_colour = masked_pixels.mean(axis=0).astype(np.uint8)
                    pixelated[y1:y2, x1:x2] = avg_colour

        # Save pixelated isolated person
        stem = Path(image_path).stem
        out_file = output_path / f"{stem}_pixelated.png"
        cv2.imwrite(str(out_file), pixelated)
        print(f"[INFO] Saved pixelated mask to: {out_file}")

        # Also save the clean binary mask
        mask_file = output_path / f"{stem}_mask.png"
        cv2.imwrite(str(mask_file), binary_mask * 255)
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
    parser.add_argument("image", help="Path to the input image.")
    parser.add_argument("--yolo-model", default="yolo11n.pt",
                        help="YOLO model name (default: yolo11n.pt)")
    parser.add_argument("--sam-model", default="sam2_b.pt",
                        help="SAM model name (default: sam2_b.pt)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="YOLO confidence threshold (default: 0.25)")
    parser.add_argument("--output-dir", default="output",
                        help="Output directory (default: output/)")
    parser.add_argument("--grid-size", type=int, default=32,
                        help="Number of grid cells per axis, lower = more pixelated (default: 32)")
    args = parser.parse_args()

    run_yolo_sam_pipeline(
        image_path=args.image,
        yolo_model=args.yolo_model,
        sam_model=args.sam_model,
        conf_threshold=args.conf,
        output_dir=args.output_dir,
        grid_size=args.grid_size,
    )
