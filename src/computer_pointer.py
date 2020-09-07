'''
Udacity Workspace
Model Downloader python3 downloader.py --name face-detection-retail-0004 --precisions FP32 -o /home/workspace
python3 computer_pointer.py --video demo.mp4 --input_type video --output_path demo_output.mp4
'''

import time
import argparse
import cv2
import logging as log

import openvino

from face_detection import Facedetection
from facial_landmarks_detection import Facial_Landmarks
from head_pose_estimation import Head_Pose_Estimation
from gaze_estimation import Gaze_Estimation

from input_feeder import InputFeeder

#from mouse_controller import MouseController

#import mouse_controller_original

def main():
    
    # loads argparser
    args = build_argparser().parse_args()
    input_type = args.input_type
    input_file = args.video
    output_path = args.output_path
    
    # Get Openvinoversion
    openvino_version = (openvino.__file__)
    print ("Openvino version: "+ str(openvino_version))
        
    # Load class Facedetection
    facedetection = Facedetection(model_name=args.fd_model, threshold=args.threshold, device=args.device, extension=args.extension)
    print("Load class Facedetection = OK")
    print("--------")
    # Load model Facedetection
    facedetection.load_model()
    print("Load model facedetection = Finished")
    print("--------")
    
    # Load facial landmark
    faciallandmarks = Facial_Landmarks(model_name=args.fl_model, threshold=args.threshold, device=args.device, extension=args.extension)
    print("Load class Facial_Landmarks = OK")
    print("--------")
    faciallandmarks.load_model()
    print("Load model Facial_Landmarks = Finished")
    print("--------")
    
    # Load head_pose_estimation
    headposeestimation = Head_Pose_Estimation(model_name=args.hp_model, device=args.device, extension=args.extension)
    print("Load class head_pose_estimation = OK")
    print("--------")
    headposeestimation.load_model()
    print("Load model head_pose_estimation = Finished")
    print("--------")
    
    # Load gaze_estimation
    gazeestimation = Gaze_Estimation(model_name=args.ga_model, device=args.device, extension=args.extension)
    print("Load class gaze_estimation = OK")
    print("--------")
    gazeestimation.load_model()
    print("Load model gaze_estimation = Finished")
    print("--------")
    
    ##############
    feed=InputFeeder(input_type, input_file)
    cap = feed.load_data()
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    facedetection.get_initial_w_h (initial_w, initial_h)
    #headposeestimation.get_initial_w_h (initial_w, initial_h)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, fps, (initial_w, initial_h))
    
    try:
        while cap.isOpened():
            result, frame = cap.read()
            if not result:
                break
                #image = inference.predict(frame, initial_w, initial_h)
            print ("Start facedetection (computer_pointer.py)")
            print ("Cap is feeded to the face detection!")
            face_image, frame_cropped = facedetection.predict(frame)
            print ("The video from the face detection is writen to the output path")
            out_video.write(face_image)
            print ("End facedetection (computer_pointer.py)")

            print ("Start faciallandmark (computer_pointer.py)")
            print("The cropped face image is feeded to the faciallandmarks detection.")
            faciallandmarks.get_initial_w_h(frame_cropped)
            left_eye_image, right_eye_image = faciallandmarks.predict(frame_cropped)
            print ("End faciallandmarks (computer_pointer.py)")

            print ("Start headposeestimation (computer_pointer.py)")
            print("The cropped face image is feeded to the headposeestimation.")
            headposeestimation.get_initial_w_h(frame_cropped)
            head_pose_angles = headposeestimation.predict(frame_cropped)
            print ("Head pose angeles: ", head_pose_angles)
            print ("End faciallheadposeestimationandmarks (computer_pointer.py)")

            print ("Start gazeestimation (computer_pointer.py)")
            gaze = gazeestimation.predict(left_eye_image, right_eye_image, head_pose_angles)
            print ("End gazeestimation (computer_pointer.py)")
            print (gaze)

            # TODO feed into the mouse controller
    except Exception as e:
        print("Could not run Inference: ", e)
        
        cap.release()
        cv2.destroyAllWindows()
        
    
    '''
    for batch in feed.next_batch():
        #print (batch)
        self.initial_w = int(input_file.get(cv2.CAP_PROP_FRAME_WIDTH))
        print("initial_w: " + str(self.initial_w))
        #do_something(batch)
    feed.close()
    
    
    self.initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    self.initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    self.video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    self.fps = int(cap.get(cv2.CAP_PROP_FPS))
    print("--------")
    print("Input video Data")
    print("initial_w: " + str(self.initial_w))
    print("initial_h: " + str(self.initial_h))
    print("video_len: " + str(self.video_len))
    print("fps: " + str(self.fps))
    print("--------")
    
    
    for batch in feed.next_batch():
        do_something(batch)
    feed.close()
    ##############
        # Text Mouse Controller
    #mousecontroller = MouseController('high', 'fast')
    #mousecontroller.move(10,100)
    '''
def getinputstream(input_type, input_file):
    # Get the input video stream
    print("Get input from input_feeder")
    input_stream = InputFeeder(input_type, input_file)
    input_stream.load_data()
    print("input_stream: " + str(input_stream))
    print("Reading video file: ", input_file)
    if not (input_stream.cap.isOpened()):
        print("Cannot find video file: " + input_file)
        
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
    
    return initial_w, initial_h

def test():
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
    parser.add_argument('--model', required=False)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--extension', default='/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so')
    parser.add_argument('--video', default=None)
    parser.add_argument('--output_path', default='demo_output.mp4')
    parser.add_argument('--threshold', default=0.60)
    parser.add_argument("-fd_model", default='models/face-detection-retail-0004', required=False)
    parser.add_argument("-fl_model", default='models/landmarks-regression-retail-0009', required=False)
    parser.add_argument("-hp_model", default='models/head-pose-estimation-adas-0001', required=False)
    parser.add_argument("-ga_model", default='models/gaze-estimation-adas-0002', required=False)
    parser.add_argument('--input_type', default='video', required=False)

    return parser

if __name__ == '__main__':
    log.basicConfig(filename="logging.txt", level=log.INFO)
    log.info("Start logging")
    main()