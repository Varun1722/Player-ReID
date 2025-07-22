import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchreid.reid.utils.feature_extractor import FeatureExtractor
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_distances

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
VIDEO_PATH = r"D:\Player ReID\fifteen_sec_input_720p.mp4"          
DETECT_MODEL = "best.pt"           # your custom YOLOv11 weights
REID_MODEL_NAME = "osnet_ain_x1_0"  # TorchReID model
REID_MODEL_WEIGHTS = "osnet_ain_x1_0_market1501.pth.tar"  
CONFIDENCE_THRESHOLD = 0.3         # YOLO detection threshold
MAX_COSINE_DISTANCE = 0.5          # assignment threshold
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------------------------------------------------------

def compute_cost_matrix(tracks, detections):
    """
    Build a cost matrix where cost[i,j] = cosine distance between
    track i embedding and detection j embedding.
    """
    if not tracks or not detections:
        return np.zeros((len(tracks), len(detections)), dtype=np.float32)

    track_feats = np.stack([t["feature"] for t in tracks])
    det_feats   = np.stack([d["feature"] for d in detections])
    return cosine_distances(track_feats, det_feats)

def main():
    # 1) Load Models
    det_model = YOLO(DETECT_MODEL)
    reid_extractor = FeatureExtractor(
        model_name=REID_MODEL_NAME,
        model_path=REID_MODEL_WEIGHTS,
        device=DEVICE
    )

    print(f"Opening video: {VIDEO_PATH!r}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: could not open video {VIDEO_PATH}")
        return

    tracks = []    # list of dicts: {id, bbox, feature}
    next_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 2) Detect persons in frame
        results = det_model.predict(frame, conf=CONFIDENCE_THRESHOLD)[0]
        detections = []
        for box in results.boxes:
            cls = int(box.cls.cpu().numpy()[0])
            if cls != 0:  # skip non-person classes
                continue

            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # 3) Extract appearance feature
            feat = reid_extractor(crop)
            feat = feat[0].cpu().numpy()  # (512,) vector

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "feature": feat
            })

        # 4) Associate detections to existing tracks
        if len(tracks) == 0:
            # initialize all detections as new tracks
            for det in detections:
                det["id"] = next_id
                tracks.append(det)
                next_id += 1
        else:
            cost = compute_cost_matrix(tracks, detections)
            row_idx, col_idx = linear_sum_assignment(cost)

            assigned_tracks = set()
            assigned_dets   = set()

            # match and update tracks
            for r, c in zip(row_idx, col_idx):
                if cost[r, c] < MAX_COSINE_DISTANCE:
                    # update existing track r with detection c
                    tracks[r]["bbox"]    = detections[c]["bbox"]
                    tracks[r]["feature"] = detections[c]["feature"]
                    detections[c]["id"]  = tracks[r]["id"]
                    assigned_tracks.add(r)
                    assigned_dets.add(c)

            # create new tracks for unmatched detections
            for i, det in enumerate(detections):
                if i not in assigned_dets:
                    det["id"] = next_id
                    tracks.append(det)
                    next_id += 1

        # 5) Visualization
        vis = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            tid = det["id"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(vis, f"ID {tid}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("ReID Tracking", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

