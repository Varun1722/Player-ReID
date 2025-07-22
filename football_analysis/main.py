from utils import read_video,save_video
from tracker import Tracker
import cv2

video_path = r"D:\Player ReID\videos\fifteen_sec_input_720p.mp4"
model_path = r"D:\Player ReID\model\best.pt"
output_path = r"D:\Player ReID\football_analysis\output_videos\output2.avi"

def main():
    #read video
    video_frames = read_video(video_path)


    track = Tracker(model_path)
    tracks = track.get_object_tracks(video_frames,read_from_stub=True,stub_path='stubs/track_stub.pkl')

    output_video_frames = track.draw_annotations(video_frames,tracks)

    #save cropped image of a player
    for track_id,player in tracks['player'][0].items():
        bbox = player['bbox']
        frame = video_frames[0]

        #crop bbox from frame
        cropped_image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        cv2.imwrite(f'output_videos/cropped_image.jpg', cropped_image)
        break

    #save video
    save_video(output_video_frames,output_path=output_path)
if __name__ == '__main__':
    main()