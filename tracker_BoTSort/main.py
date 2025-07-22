import cv2, time
from object_detector import YoloDetector
from tracker import Tracker

MODEL_PATH = r"D:\Player ReID\model\best.pt"
VIDEO_PATH = r"D:\Player ReID\videos\fifteen_sec_input_720p.mp4"

colour_map, next_colour = {}, 0
COLOURS = [(0,255,0),(255,0,0),(0,0,255),(255,255,0),(0,255,255),
           (255,0,255),(128,0,0),(0,128,0),(0,0,128)]

def get_colour(tid):
    global next_colour
    if tid not in colour_map:
        colour_map[tid] = COLOURS[next_colour % len(COLOURS)]
        next_colour += 1
    return colour_map[tid]

def main():
    detector = YoloDetector(model_path=MODEL_PATH, confidence=0.25)
    
    tracker  = Tracker()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error opening video."); return

    while True:
        ok, frame = cap.read()
        if not ok: break
        t0 = time.perf_counter()

        results = detector.track(frame, persist=True, tracker='bytetrack.yaml')

        for tid, box in zip(track_ids, boxes):
            x1,y1,x2,y2 = map(int, box)
            colour      = get_colour(tid)
            cv2.rectangle(frame,(x1,y1),(x2,y2),colour,2)
            cv2.putText(frame,f"ID:{tid}",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,colour,2)

        fps = 1 / (time.perf_counter() - t0)
        cv2.putText(frame,f"{fps:.1f} FPS",(20,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

        cv2.imshow("BoT-SORT Tracking", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')): break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
