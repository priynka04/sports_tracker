# Multi-Object Detection and Persistent ID Tracking

## Source Video
[PASTE YOUR YOUTUBE LINK HERE]

## What This Does
Detects all players in a sports video and assigns each one a
unique persistent ID that stays stable across the full video.

## Setup

### Step 1 - Install Python 3.8+
Download from https://python.org

### Step 2 - Install dependencies
pip install -r requirements.txt

### Step 3 - Add your video
Place your video file as input_short.mp4 in the same folder

### Step 4 - Run
python track.py

### Output
- output_tracked.mp4  → annotated video with boxes and IDs
- screenshots/        → sample frames saved automatically

## Model & Tracker
- Detector : YOLOv8 Nano (ultralytics)
- Tracker  : DeepSORT (deep-sort-realtime)

## Assumptions
- Video contains people as main subjects
- Input file is named input_short.mp4
- Reasonable lighting conditions

## Limitations
- ID switches can happen during heavy occlusion
- Very distant/small players may be missed
- Processing speed depends on hardware (no GPU used)