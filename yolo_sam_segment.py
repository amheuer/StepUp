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
):
    """
    Run YOLO detection followed by SAM segmentation.

    Args:
        image_path: Path to the input image.
        yolo_model: YOLO model weights (downloaded automatically).
        sam_model: SAM model weights (downloaded automatically).
        conf_threshold: Confidence threshold for YOLO detections.
        output_dir: Directory to save results.
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
    bboxes = boxes.xyxy.cpu().numpy()            # (N, 4) — x1, y1, x2, y2
    confs = boxes.conf.cpu().numpy()             # (N,)
    cls_ids = boxes.cls.cpu().numpy().astype(int) # (N,)
    class_names = det_result.names               # {id: name, ...}

    print(f"[INFO] Detected {len(bboxes)} object(s):")
    for i, (box, conf, cls_id) in enumerate(zip(bboxes, confs, cls_ids)):
        print(f"  [{i}] {class_names[cls_id]:>12s}  conf={conf:.2f}  "
              f"bbox=({box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f})")

    # ── 3. Run SAM segmentation with YOLO bounding boxes ───────────
    print("[INFO] Running SAM segmentation on detected bounding boxes...")
    sam_results = sam(image_path, bboxes=bboxes, verbose=False)
    sam_result = sam_results[0]

    # ── 4. Visualise & save ─────────────────────────────────────────
    image = cv2.imread(image_path)
    overlay = image.copy()

    # Generate distinct colours for each detection
    rng = np.random.default_rng(42)
    colours = rng.integers(60, 255, size=(len(bboxes), 3)).tolist()

    if sam_result.masks is not None:
        masks = sam_result.masks.data.cpu().numpy()  # (N, H, W) bool / float

        for i, (mask, box, conf, cls_id) in enumerate(
            zip(masks, bboxes, confs, cls_ids)
        ):
            colour = colours[i]
            label = f"{class_names[cls_id]} {conf:.2f}"

            # Draw filled mask on overlay
            binary_mask = (mask > 0.5).astype(np.uint8)
            coloured_mask = np.zeros_like(image)
            coloured_mask[:] = colour
            overlay[binary_mask == 1] = cv2.addWeighted(
                overlay[binary_mask == 1], 0.5,
                coloured_mask[binary_mask == 1], 0.5, 0,
            )

            # Draw bounding box
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, 2)

            # Put label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(overlay, (x1, y1 - th - 8), (x1 + tw, y1), colour, -1)
            cv2.putText(
                overlay, label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
            )

    # Save result
    stem = Path(image_path).stem
    out_file = output_path / f"{stem}_segmented.jpg"
    cv2.imwrite(str(out_file), overlay)
    print(f"[INFO] Saved segmented image to: {out_file}")

    return overlay


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
    args = parser.parse_args()

    run_yolo_sam_pipeline(
        image_path=args.image,
        yolo_model=args.yolo_model,
        sam_model=args.sam_model,
        conf_threshold=args.conf,
        output_dir=args.output_dir,
    )
