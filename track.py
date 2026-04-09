"""
Multi-Object Detection and Persistent ID Tracking Pipeline
Detector  : YOLOv8 Nano
Tracker   : DeepSORT
Author    : [Your Name]
"""

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
import os

# ─── CONFIG ───────────────────────────────────────────────────────────────────
INPUT_VIDEO      = "input_short.mp4"
OUTPUT_VIDEO     = "output_tracked.mp4"
SCREENSHOTS_DIR  = "screenshots"
CONFIDENCE       = 0.4        # minimum detection confidence
TARGET_CLASS     = 0          # 0 = person in COCO classes
MAX_AGE          = 30         # frames to keep a lost track alive
MAX_TRAIL        = 40         # how many past positions to draw as trail
SCREENSHOT_EVERY = 150        # save a screenshot every N frames
# ──────────────────────────────────────────────────────────────────────────────

# Store trajectory points per track ID
trajectory_history = defaultdict(list)

# Track counts over time (for stats)
count_over_time = []


def get_color(track_id):
    """Return a unique BGR color for each track ID."""
    np.random.seed(int(track_id) % 200)
    return tuple(int(c) for c in np.random.randint(80, 255, 3))


def draw_trajectory(frame, track_id, color):
    """Draw a fading dotted trail behind each tracked subject."""
    points = trajectory_history[track_id]
    for i in range(1, len(points)):
        alpha     = i / len(points)
        thickness = max(1, int(alpha * 3))
        cv2.line(frame, points[i - 1], points[i], color, thickness)


def draw_hud(frame, frame_num, active_count):
    """Draw a small heads-up display overlay on the frame."""
    h, w = frame.shape[:2]

    # Dark semi-transparent bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 45), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.putText(frame, f"Frame: {frame_num}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)
    cv2.putText(frame, f"Active Tracks: {active_count}",
                (220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 120), 2)
    cv2.putText(frame, "YOLOv8 + DeepSORT",
                (w - 240, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 200, 255), 2)


def run_tracker():
    # ── Setup output folder for screenshots ──────────────────────────────────
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

    # ── Load models ──────────────────────────────────────────────────────────
    print("Loading YOLOv8 model...")
    model   = YOLO("yolov8n.pt")

    print("Initializing DeepSORT tracker...")
    tracker = DeepSort(
        max_age            = MAX_AGE,
        n_init             = 3,
        nms_max_overlap    = 0.7,
        max_cosine_distance= 0.3,
    )

    # ── Open video ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"ERROR: Cannot open '{INPUT_VIDEO}'. Make sure the file exists.")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo Info: {width}x{height} @ {fps:.1f} fps | {total} total frames\n")

    # ── Setup video writer ───────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    frame_num   = 0
    total_ids   = set()

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # ── A. Detect with YOLO ───────────────────────────────────────────
        results    = model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            if cls != TARGET_CLASS or conf < CONFIDENCE:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            detections.append(([x1, y1, w, h], conf, cls))

        # ── B. Update DeepSORT tracker ────────────────────────────────────
        tracks = tracker.update_tracks(detections, frame=frame)

        active_count = 0

        # ── C. Draw results ───────────────────────────────────────────────
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            total_ids.add(track_id)

            ltrb           = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            cx, cy          = (x1 + x2) // 2, (y1 + y2) // 2
            color           = get_color(track_id)

            # Update trajectory history
            trajectory_history[track_id].append((cx, cy))
            if len(trajectory_history[track_id]) > MAX_TRAIL:
                trajectory_history[track_id].pop(0)

            # Draw trajectory trail
            draw_trajectory(frame, track_id, color)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background + text
            label           = f"ID {track_id}"
            (tw, th), _     = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw center dot
            cv2.circle(frame, (cx, cy), 4, color, -1)

            active_count += 1

        # ── D. Draw HUD overlay ───────────────────────────────────────────
        draw_hud(frame, frame_num, active_count)
        count_over_time.append(active_count)

        # ── E. Save screenshot every N frames ────────────────────────────
        if frame_num % SCREENSHOT_EVERY == 0:
            ss_path = os.path.join(SCREENSHOTS_DIR, f"frame_{frame_num:05d}.jpg")
            cv2.imwrite(ss_path, frame)
            print(f"  Screenshot saved: {ss_path}")

        # ── F. Write frame to output video ────────────────────────────────
        out.write(frame)

        print(f"  Frame {frame_num:4d}/{total} | Active: {active_count:2d} | "
              f"Total unique IDs so far: {len(total_ids)}", end="\r")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    out.release()

    # ── Print final summary ───────────────────────────────────────────────────
    print(f"\n\n{'='*50}")
    print(f"  Processing complete!")
    print(f"  Output video  : {OUTPUT_VIDEO}")
    print(f"  Total frames  : {frame_num}")
    print(f"  Unique IDs    : {len(total_ids)}")
    if count_over_time:
        print(f"  Avg tracks/frame : {np.mean(count_over_time):.1f}")
        print(f"  Max tracks/frame : {max(count_over_time)}")
    print(f"  Screenshots   : {SCREENSHOTS_DIR}/")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    run_tracker()