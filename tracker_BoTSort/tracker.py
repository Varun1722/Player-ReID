from deep_sort_realtime.deepsort_tracker import DeepSort
from pathlib import Path

class Tracker:
    def __init__(self):
        weights = Path(r"..\model\osnet_ain_ms_d_c.pth.tar")  # adjust to your tree
        self.trk = DeepSort(
            model_type="botsort",            # ← ByteTrack + OSNet gating
            max_iou_distance=0.6,
            max_age=200,
            n_init=5,
            embedder="torchreid",
            embedder_model_name="osnet_ain_x1_0",
            embedder_wts=str(weights),
            embedder_gpu=True,
            half=True,
            bgr=True
        )

    def track(self, detections, frame):
        tracks = self.trk.update_tracks(detections, frame=frame)
        ids, boxes = [], []
        for t in tracks:
            if not t.is_confirmed():
                continue
            ids.append(t.track_id)
            boxes.append(t.to_ltrb())        # (x1,y1,x2,y2)
        return ids, boxes
