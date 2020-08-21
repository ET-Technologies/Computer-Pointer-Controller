# Udacity Workspace
# Model Downloader python3 downloader.py --name face-detection-retail-0004 --precisions FP32 -o /home/workspace

# python3 computer_pointer.py --model models/face-detection-retail-0004 --video demo.mp4

import time
import argparse
import cv2
import logging as log

from face_detection import Facedetection
from input_feeder import InputFeeder

def main():
    args = build_argparser().parse_args()
    
    # Load face_detection
    facedetection = Facedetection(model_name=args.fd_model, threshold=args.threshold, device=args.device)
    print("Load class Facedetection = OK")
    print("--------")
    facedetection.load_model()
    print("Load model facedetection = OK")
    print("--------")
    
    # Get the input video stream
    print("Get input from input_feeder")
    input_stream = InputFeeder(input_type='video', input_file= args.video)
    input_stream.load_data()
    print("input_stream: " + str(input_stream))
    print("Reading video file: ", args.video)
    if not (input_stream.cap.isOpened()):
        print("Cannot find video file: " + video)
        
    # Capture information about the input video stream
    initial_w = int(input_stream.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(input_stream.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(input_stream.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(input_stream.cap.get(cv2.CAP_PROP_FPS))
    print("--------")
    print("Input video Data")
    print("initial_w: " + str(initial_w))
    print("initial_h: " + str(initial_h))
    print("video_len: " + str(video_len))
    print("fps: " + str(fps))
    print("--------")
    
    # Define output video

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter('output_video3.mp4', fourcc, fps, (initial_w, initial_h))

    try:
        while input_stream.cap.isOpened():
            result, frame = input_stream.cap.read()
            if not result:
                break
            image = facedetection.predict(frame, initial_w, initial_h)
            print("The video is writen to the output path")
            out_video.write(image)
    except Exception as e:
        print("Could not run Inference: ", e)

        input_stream.close()
        cv2.destroyAllWindows()
    
def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--extension', default=None)
    parser.add_argument('--video', default=None)
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--threshold', default=0.60)
    parser.add_argument("-fd_model", default='models/face-detection-retail-0004', required=False)

    return parser

if __name__ == '__main__':
    log.basicConfig(filename="logging.txt", level=log.INFO)
    log.info("Start logging")
    main()