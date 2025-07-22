import cv2
from ultralytics import YOLO
import imutils
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

# --- Configuration ---
video_path = r"videos\fifteen_sec_input_720p.mp4"
model_path = r"model\best.pt"  # Assuming 'best.pt' is your trained YOLOv8 model for player detection

# Load YOLO model once
model = YOLO(model_path)

# Define a dictionary to store player colors for consistent visualization
player_colors = {}
next_color_idx = 0
COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255),
          (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)]

def get_player_color(track_id):
    """Assigns a consistent color to each unique player ID."""
    global next_color_idx
    if track_id not in player_colors:
        player_colors[track_id] = COLORS[next_color_idx % len(COLORS)]
        next_color_idx += 1
    return player_colors[track_id]

def processVideo():
    yolo_model = model

    # Initialize DeepSort tracker
    # Increased max_age to allow for longer absence from frame
    # Reduced max_iou_distance for stricter association in current frame
    object_tracker = DeepSort(max_iou_distance=0.3,  # Increased strictness for current frame association
                              max_age=200,             # Increased to allow more frames for re-emergence
                              n_init=10,
                              nms_max_overlap=1.0,
                              max_cosine_distance=0.2, # Cosine distance for appearance similarity
                              nn_budget=300,
                              gating_only_position=False,
                              embedder='torchreid',
                              embedder_model_name = 'osnet_ain_x1_0',   # Use osnet for appearance embedding
                              embedder_wts ='osnet_ain_ms_d_c.pth.tar',
                              embedder_gpu = True,
                              half=True,              # Use half-precision for faster inference if supported
                              bgr=True
                              )      # Leverage GPU for embedder if available

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # Optional: Setup video writer to save the output
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
    # out = cv2.VideoWriter('output_tracked_video.mp4', fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = video.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # Predict using YOLO model
        # Using persist=True can sometimes help with tracking by retaining features
        results = yolo_model(frame, stream=False, verbose=False, conf=0.5) # Set verbose to False for cleaner output

        detections = []
        if results and len(results) > 0:
            # DeepSort expects detections in a specific format:
            # (bbox_xywh, confidence, class_id)
            for res in results:
                for *xyxy, conf, cls in res.boxes.data.tolist():
                    # Convert xyxy to xywh
                    x1, y1, x2, y2 = xyxy
                    w, h = x2 - x1, y2 - y1
                    bbox_xywh = [x1, y1, w, h]
                    
                    # Filter detections to only include 'person' class if available
                    # Assuming your 'best.pt' is trained to detect 'person' or 'player'
                    # You might need to adjust the class index based on your model's classes
                    class_name = yolo_model.names[int(cls)]
                    if class_name == 'player': # or 'player' if your model is specifically for players
                        detections.append((bbox_xywh, conf, int(cls)))

        # Update tracker with current detections
        # The 'frame' argument is crucial for the embedder to extract features
        tracks = object_tracker.update_tracks(detections, frame=frame)

        # Draw bounding boxes and IDs
        for track in tracks:
            if not track.is_confirmed():
                continue # Only draw confirmed tracks

            track_id = track.track_id
            bbox_ltrb = track.to_ltrb()  # Get bounding box in (x1, y1, x2, y2) format
            
            x1, y1, x2, y2 = int(bbox_ltrb[0]), int(bbox_ltrb[1]), int(bbox_ltrb[2]), int(bbox_ltrb[3])
            
            # Get consistent color for the player
            color = get_player_color(track_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Display track ID and confidence (optional)
            text = f"ID: {track_id}" # Add confidence if desired: f"ID: {track_id} Conf: {track.confidence:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Optional: Display class name if needed
            # class_id = track.get_det_class() # DeepSort stores the class id of the last detection
            # if class_id is not None and class_id in yolo_model.names:
            #     class_name = yolo_model.names[class_id]
            #     cv2.putText(frame, class_name, (x1, y1 - 30),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)


        cv2.imshow('Player Tracking', frame)
        
        # Optional: Write frame to output video
        # out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): # Use waitKey(1) for continuous video
            break

    video.release()
    # out.release() # Release video writer if used
    cv2.destroyAllWindows()

# Removed drawBox and get_details as their functionality is integrated into processVideo
# or can be simplified directly within the loop.
# drawBox was drawing based on raw YOLO output, not tracked IDs.
# get_details was converting to a list of tuples, which is now done inline.

# --- Run the processing ---
processVideo()