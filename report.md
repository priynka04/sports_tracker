# Technical Report — Multi-Object Detection & ID Tracking
## Source Video
- **URL:** https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi
- **Source:** OpenCV official GitHub repository (publicly accessible)
- **Description:** A publicly available pedestrian surveillance video from the 
  official OpenCV sample dataset. It contains multiple people walking in an 
  outdoor area, making it ideal for multi-object detection and tracking evaluation.
- **License:** Publicly available open-source sample data from OpenCV

## 1. Model Used — YOLOv8 Nano
YOLOv8 (You Only Look Once v8) by Ultralytics is a real-time
object detector. The Nano variant was chosen for speed on CPU.
It detects the "person" class (class 0) from the COCO dataset.
Detections below 40% confidence are filtered out.

## 2. Tracker Used — DeepSORT
DeepSORT (Simple Online and Realtime Tracking with Deep
appearance features) maintains persistent IDs using:
- Kalman Filter: predicts where a person moves next frame
- Hungarian Algorithm: matches detections to existing tracks
- Cosine Distance: uses appearance to re-identify after occlusion

## 3. Why This Combination
YOLOv8 + DeepSORT is the most well-established baseline for
multi-person tracking. It works well out of the box, requires
no training, and handles real-world sports footage reliably.

## 4. How ID Consistency Is Maintained
Each track has an internal state (position + velocity via Kalman
Filter). When a new detection appears, DeepSORT matches it to
the nearest existing track using both spatial overlap (IoU) and
visual appearance (cosine distance on feature embeddings).
If a match is found, the same ID continues. If no match for
MAX_AGE=30 frames, the track is deleted.

## 5. Challenges Faced
- Players with similar jerseys/appearance cause ID confusion
- Overlapping players during tackles or close groupings
- Motion blur from fast movement reduces detection confidence
- Players exiting and re-entering frame get new IDs

## 6. Observed Failure Cases
- Two players crossing paths sometimes swap IDs briefly
- Goalkeepers or distant players sometimes not detected
- Players entering frame from edges trigger new ID assignment

## 7. Possible Improvements
- Use YOLOv8m (medium) for higher detection accuracy
- Use BoT-SORT or ByteTrack for more stable ID assignment
- Add team-color clustering for better re-identification
- Run on GPU for real-time processing
- Use higher resolution input for small/distant subjects