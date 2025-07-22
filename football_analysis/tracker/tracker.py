from ultralytics import YOLO
import supervision as sv
import pickle
import cv2
import numpy as np
import os
import sys
sys.path.append('../')
from utils import get_center_of_bbox,get_width_of_bbox

class Tracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self,frames):
        '''
        Detect the relevant classes in a frame

        Args:
        A list of frames extracted from original video
        Returns:
        a list of frames containing detections (bounding boxes, confidence score, class)
        '''
        batch_size = 20
        detections = []
        for i in range(0,len(frames),batch_size):
            result = self.model.predict(frames[i:i+batch_size],conf=0.1)
            '''
            We could have directly used track above but our model confuses between goalkeeper and player 
            so we are going to treat goalkeeper as player by overiding the goalkeeper class
            We will use supervision library to track after detecting
            '''
            detections += result

        return detections
    
    def get_object_tracks(self,frames, read_from_stub=False,stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)

            return tracks
    
        detections = self.detect_frames(frames)

        tracks = {
            "players":[],  #[{0:{'bbox':[0,0,0,0]}, 1:{'bbox':[1,8,3,6]}}, {0:{'bbox':[0,3,4,0]}, 1:{'bbox':[1,8,3,6]}}]
            "referee":[],
            "ball":[]
        }

        '''
        To override the goalkeeper class
        '''
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            #convert the frames to supervision detection format 
            supervision_detections = sv.Detections.from_ultralytics(detection)

            #override the goalkeeper class to player class
            for index, class_id in enumerate(supervision_detections.class_id):
                if cls_names[class_id]=='goalkeeper':
                    supervision_detections.class_id[index] = cls_names_inv['player'] 

            #Track Objects
            detection_with_track = self.tracker.update_with_detections(supervision_detections)

            tracks["players"].append({})
            tracks["referee"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_track:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                
                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                if cls_id == cls_names_inv['referee']:
                    tracks["referee"][frame_num][track_id] = {"bbox":bbox}
                
            for frame_detection in detection_with_track:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    
    def draw_ellipse(self, frame,bbox, color, track_id):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox=bbox)
        width = get_width_of_bbox(bbox=bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width),int(0.35*width)),
            angle = 0.0,
            startAngle = -45,
            endAngle = 235,
            color=color,
            thickness=2,
            lineType = cv2.LINE_8
        )

        #adding rectangle in the middle
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = int(x_center-rectangle_width//2)
        x2_rect = int(x_center+rectangle_width//2)
        y1_rect = int(y2-rectangle_height//2)+15
        y2_rect = int (y2+rectangle_height//2)+15

        if track_id is not None:
            cv2.rectangle(frame,
                          (x1_rect,y1_rect),
                          (x2_rect,y2_rect),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame
    
    def draw_triangle(self,frame, bbox, color):
        x = get_center_of_bbox(bbox=bbox)
        y = int(bbox[1])

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])

        cv2.drawContours(frame,[triangle_points],0,color,cv2.FILLED)
        cv2.drawContours(frame,[triangle_points],0,(0,0,0),2)
        
        return frame
    
    def draw_annotations(self,input_frames, tracks=None):
        '''
        to draw ellipse on a player rather than bbox
        '''
        output_frames =[]
        for frame_num, frame in enumerate(input_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referee"][frame_num]

            #draw ellipse
            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player["bbox"],(0,0,255),track_id)

            #for referees
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,0,255),track_id)
            
            #for ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"],(0,255,0))
            
            output_frames.append(frame)

        return output_frames