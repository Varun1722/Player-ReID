# auto_label_yolo_to_mot.py
# pip install ultralytics opencv-python

import csv, cv2
from pathlib import Path
from ultralytics import YOLO   # works for v8 / v11

# ---------- user config ----------
VIDEO_PATH  = "../videos/fifteen_sec_input_720p.mp4"
MODEL_PATH  = "../model/best.pt"               # detector weights
CONF_THRES  = 0.4                     # detection confidence
IOU_THRES   = 0.45                    # NMS IoU
CLASS_LIST  = [0,1,2,3]                  # only keep class 0 ('person') – adjust if needed
OUT_TXT     = "det_yolo.txt"          # MOT-format output
# ----------------------------------

model = YOLO(MODEL_PATH)
cap   = cv2.VideoCapture(VIDEO_PATH)
rows  = []
frame_idx = 0

def xyxy_to_tlwh(x1,y1,x2,y2):
    """convert (x1,y1,x2,y2) -> (tl_x, tl_y, w, h)"""
    return [x1, y1, x2-x1, y2-y1]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    res = model.predict(
        frame,
        imgsz=1024,
        conf=CONF_THRES,
        iou=IOU_THRES,
        classes=CLASS_LIST,
        verbose=False
    )[0]

    for *xyxy, conf, cls in res.boxes.data.tolist():
        tlwh = xyxy_to_tlwh(*xyxy)
        rows.append([
            frame_idx,          # frame number (1-based)
            -1,                 # track_id = -1  (unassigned)
            round(tlwh[0],2), round(tlwh[1],2),
            round(tlwh[2],2), round(tlwh[3],2),
            round(conf,4),      # detection confidence
            -1, -1, -1          # dummy cols (ignored by evaluator)
        ])

cap.release()
Path(OUT_TXT).write_text("")  # create/clear file
with open(OUT_TXT, "w", newline="") as f:
    csv.writer(f).writerows(rows)

print(f"✅ Wrote {len(rows)} detections to {OUT_TXT}")
