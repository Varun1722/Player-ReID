# auto_label_with_ids.py
# pip install ultralytics opencv-python deep_sort_realtime

import csv, cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ---------- config ----------
VIDEO_PATH  = "../videos/fifteen_sec_input_720p.mp4"
MODEL_PATH  = "../model/best.pt"            # YOLO weights
CONF_THRES  = 0.4
IOU_THRES   = 0.45
CLASS_LIST  = None                 # None = keep all classes
OUT_TXT     = "auto_gt.txt"        # MOT file with track IDs
# -----------------------------

# 1) Detector
yolo = YOLO(MODEL_PATH)

# 2) Very light tracker: only IoU + centroid; no appearance
tracker = DeepSort(
    max_iou_distance = 0.7,
    max_age          = 15,
    n_init           = 3,
    nn_budget        = None,
    embedder         = 'mobilenet',       # disable appearance net
)

def xyxy_to_tlwh(xyxy):
    x1,y1,x2,y2 = xyxy
    return [x1, y1, x2-x1, y2-y1]

rows = []
cap  = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    det_res = yolo.predict(
        frame,
        imgsz=1024,
        conf=CONF_THRES,
        iou=IOU_THRES,
        classes=CLASS_LIST,
        verbose=False
    )[0]

    detections = []
    for *xyxy, conf, cls in det_res.boxes.data.tolist():
        tlwh = xyxy_to_tlwh(xyxy)
        detections.append((tlwh, conf, int(cls)))

    tracks = tracker.update_tracks(detections, frame=frame)   # no frame needed (no appearance)

    for t in tracks:
        if not t.is_confirmed():          # filter tentative tracks
            continue
        tlwh = xyxy_to_tlwh(t.to_ltrb())
        rows.append([
            frame_idx,
            t.track_id,                   # <<< auto-generated ID
            round(tlwh[0],2), round(tlwh[1],2),
            round(tlwh[2],2), round(tlwh[3],2),
            1,
            2, -1, -1
        ])

cap.release()

with open(OUT_TXT, "w", newline="") as f:
    csv.writer(f).writerows(rows)
print(f"✅ Wrote {len(rows)} labelled boxes with IDs → {OUT_TXT}")
