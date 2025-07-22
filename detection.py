import cv2
from ultralytics import YOLO

video_path = "videos/fifteen_sec_input_720p.mp4"

model = YOLO("model/best.pt")

def processVideo():
    path = video_path
    yolo_model = model
    print(model.names)
    ''' object_tracker = DeepSort(max_iou_distance = 0.7,
                              max_age = 5,
                              n_init = 3,
                              nms_max_overlap =1.0 ,
                              max_cosine_distance=0.2,
                              nn_budget=None,
                              gating_only_position=False,
                              override_track_class=None,
                              embedder='mobilenet',
                              half=True,
                              bgr=True,
                              embedder_gpu=True,
                              embedder_model_name=None,
                              embedder_wts=None,
                              polygon=False,
                              today=None)
'''
    video = cv2.VideoCapture(path)
    while True:
        (ret,frame) = video.read()  #ret is a boolean variable which tells if frame was read
        if not ret and 0xFF==ord('q'):
            break
        
        #predicting using a model
        result = yolo_model(frame,stream = False)
        detection_class = result[0].names
        for res in result:
            for data in res.boxes.data.tolist(): #originally in tensor 
                # print(data)
                id = data[5]
                name = detection_class[id]
                drawBox(data=data, image=frame,name=name)

            details = get_details(result=res,image=frame)
        #   /  tracks = object_tracker.update_tracks(details,frame=frame)

            # for track in tracks:
            #     if not track.is_confirmed():
            #         break
                
            #     track_id = track.track_id
            #     bbox=track.to_ltrb()
            #     cv2.putText(frame, "id "+str(track_id),(int(bbox[0]),int(bbox[1])),
            #                 cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

        cv2.imshow('image',frame)
        cv2.waitKey(20)
        

def drawBox(data,image, name):
    x1,y1,x2,y2, conf, id = data
    p1 = (int(x1), int(y1))
    p2 = (int(x2), int(y2))
    cv2.rectangle(image,p1,p2,(0,0,255),2)
    cv2.putText(image,name,p1,cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    return image

def get_details(result,image):
    classes = result.boxes.cls.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    xywh = result.boxes.xywh.cpu().numpy()

    detections=[]
    for i,item in enumerate(xywh):
        sample = (item,conf[i],classes[i])
        detections.append(sample)

    return detections
    
    
processVideo()