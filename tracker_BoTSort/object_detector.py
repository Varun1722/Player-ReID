from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path, confidence=0.4):
        self.model = YOLO(model_path)
        self.keep = {"player", "ball", "referee", "goalkeeper"}
        self.conf = confidence            # keep attribute short

    def detect(self, image):
        result   = self.model.predict(image, conf=self.conf, iou=0.45)[0]
        boxes    = result.boxes
        dets     = []                     # (xywh, conf, cls_id)
        for box in boxes:
            cls_id = int(box.cls[0])
            if result.names[cls_id] not in self.keep:
                continue
            x1,y1,x2,y2 = map(float, box.xyxy[0])
            w, h        = x2 - x1, y2 - y1
            dets.append(([x1, y1, w, h], float(box.conf[0]), cls_id))
        return dets
