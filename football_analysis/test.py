from ultralytics import YOLO
import cv2 as cv

file_path = r"D:\Player ReID\model\best.pt"
video_path = r"D:\Player ReID\videos\fifteen_sec_input_720p.mp4"
model = YOLO(file_path)
# results = model.predict(video_path,save=True)
# for box in results[0]:
#     print(box)

cap = cv.VideoCapture(video_path)
fps = cap.get(cv.CAP_PROP_FPS)
print(fps)


