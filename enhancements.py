"""
Optional Enhancements Script
Run this AFTER track.py has already been run.
Generates:
  1. Object count over time graph
  2. Movement heatmap
  3. Simple evaluation metrics (printed to terminal)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
import os

# ─── CONFIG ───────────────────────────────────────────────
INPUT_VIDEO     = "input_short.mp4"
OUTPUT_DIR      = "enhancements"
CONFIDENCE      = 0.4
TARGET_CLASS    = 0
MAX_AGE         = 30
# ──────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_enhancements():
    print("Loading models...")
    model   = YOLO("yolov8n.pt")
    tracker = DeepSort(max_age=MAX_AGE, n_init=3,
                       nms_max_overlap=0.7, max_cosine_distance=0.3)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print("ERROR: Cannot open video.")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} | {total} frames | {fps:.1f} fps\n")

    # Data collectors
    count_over_time  = []        # active tracks per frame
    heatmap_acc      = np.zeros((height, width), dtype=np.float32)
    all_track_ids    = set()
    trajectory       = defaultdict(list)
    frame_num        = 0

    # ── Main loop ─────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        print(f"  Analysing frame {frame_num}/{total}", end="\r")

        results    = model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            if cls != TARGET_CLASS or conf < CONFIDENCE:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(([x1, y1, x2-x1, y2-y1], conf, cls))

        tracks = tracker.update_tracks(detections, frame=frame)

        active = 0
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid  = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            all_track_ids.add(tid)
            trajectory[tid].append((cx, cy, frame_num))

            # Add to heatmap — fill the bounding box area
            x1c = max(0, x1)
            y1c = max(0, y1)
            x2c = min(width - 1, x2)
            y2c = min(height - 1, y2)
            heatmap_acc[y1c:y2c, x1c:x2c] += 1
            active += 1

        count_over_time.append(active)

    cap.release()
    print(f"\n\nAnalysis complete. Generating outputs...\n")

    # ── 1. OBJECT COUNT OVER TIME GRAPH ───────────────────
    print("  Generating count-over-time graph...")
    frames_axis = list(range(1, len(count_over_time) + 1))
    times_axis  = [f / fps for f in frames_axis]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(times_axis, count_over_time, alpha=0.3, color="#00aaff")
    ax.plot(times_axis, count_over_time, color="#00aaff", linewidth=1.5)
    ax.axhline(y=np.mean(count_over_time), color="orange",
               linestyle="--", linewidth=1.5, label=f"Average: {np.mean(count_over_time):.1f}")
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Active Tracks", fontsize=12)
    ax.set_title("Object Count Over Time — YOLOv8 + DeepSORT", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    graph_path = os.path.join(OUTPUT_DIR, "count_over_time.png")
    plt.savefig(graph_path, dpi=150)
    plt.close()
    print(f"  Saved: {graph_path}")

    # ── 2. MOVEMENT HEATMAP ────────────────────────────────
    print("  Generating movement heatmap...")

    # Read one frame as background
    cap2 = cv2.VideoCapture(INPUT_VIDEO)
    ret, bg_frame = cap2.read()
    cap2.release()

    # Blur and normalise heatmap
    heatmap_blur = cv2.GaussianBlur(heatmap_acc, (51, 51), 0)
    heatmap_norm = cv2.normalize(heatmap_blur, None, 0, 255,
                                 cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    # Overlay on background frame
    if ret:
        overlay = cv2.addWeighted(bg_frame, 0.5, heatmap_color, 0.6, 0)
    else:
        overlay = heatmap_color

    cv2.putText(overlay, "Movement Heatmap — YOLOv8 + DeepSORT",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    heatmap_path = os.path.join(OUTPUT_DIR, "movement_heatmap.png")
    cv2.imwrite(heatmap_path, overlay)
    print(f"  Saved: {heatmap_path}")

    # ── 3. TRAJECTORY MAP ─────────────────────────────────
    print("  Generating trajectory map...")

    fig2, ax2 = plt.subplots(figsize=(10, 7))

    # Show video frame as background
    if ret:
        bg_rgb = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB)
        ax2.imshow(bg_rgb, alpha=0.5)

    np.random.seed(42)
    for tid, points in trajectory.items():
        if len(points) < 5:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        color = np.random.rand(3)
        ax2.plot(xs, ys, linewidth=1.5, color=color, alpha=0.8)
        ax2.plot(xs[-1], ys[-1], "o", color=color, markersize=5)
        ax2.text(xs[-1], ys[-1], f"ID {tid}", fontsize=7,
                 color="white",
                 bbox=dict(facecolor=color, alpha=0.7, pad=1))

    ax2.set_xlim(0, width)
    ax2.set_ylim(height, 0)
    ax2.set_title("Trajectory Map — All Tracked Subjects", fontsize=13, fontweight="bold")
    ax2.axis("off")
    plt.tight_layout()
    traj_path = os.path.join(OUTPUT_DIR, "trajectory_map.png")
    plt.savefig(traj_path, dpi=150)
    plt.close()
    print(f"  Saved: {traj_path}")

    # ── 4. SIMPLE EVALUATION METRICS (printed + saved) ────
    print("\n  Computing metrics...")
    duration_sec    = total / fps
    avg_tracks      = np.mean(count_over_time)
    max_tracks      = max(count_over_time)
    min_tracks      = min(count_over_time)
    total_ids       = len(all_track_ids)
    frames_with_det = sum(1 for c in count_over_time if c > 0)
    detection_rate  = (frames_with_det / total) * 100

    # Track stability: avg track length in frames
    track_lengths   = [len(pts) for pts in trajectory.values()]
    avg_track_len   = np.mean(track_lengths) if track_lengths else 0

    metrics_text = f"""
========================================
       EVALUATION METRICS SUMMARY
========================================
Video Duration       : {duration_sec:.1f} seconds
Total Frames         : {total}
FPS                  : {fps:.1f}

--- Detection Stats ---
Total Unique IDs     : {total_ids}
Avg Active Tracks    : {avg_tracks:.2f} per frame
Max Active Tracks    : {max_tracks}
Min Active Tracks    : {min_tracks}
Detection Rate       : {detection_rate:.1f}% of frames had detections

--- Tracking Stats ---
Avg Track Length     : {avg_track_len:.1f} frames
  (higher = more stable ID assignment)

--- Observations ---
ID switches may occur during heavy occlusion.
Subjects exiting/re-entering frame get new IDs.
No GPU used — CPU inference only.
========================================
"""
    print(metrics_text)

    metrics_path = os.path.join(OUTPUT_DIR, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(metrics_text)
    print(f"  Saved: {metrics_path}")

    # ── Final summary ──────────────────────────────────────
    print(f"\n{'='*45}")
    print(f"  All enhancements saved to: {OUTPUT_DIR}/")
    print(f"  Files generated:")
    print(f"    - count_over_time.png")
    print(f"    - movement_heatmap.png")
    print(f"    - trajectory_map.png")
    print(f"    - metrics.txt")
    print(f"{'='*45}\n")


if __name__ == "__main__":
    run_enhancements()