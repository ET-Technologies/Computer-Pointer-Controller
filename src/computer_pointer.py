'''
Udacity Workspace
Model Downloader python3 downloader.py --name face-detection-retail-0004 --precisions FP32 -o /home/workspace
python3 computer_pointer.py --video demo.mp4 --input_type video --output_path demo_output.mp4
'''

'''
Raspberry
python3 computer_pointer.py --video /bin/demo.mp4 --input_type video --output_path demo_output.mp4 --device MYRIAD
/home/pi/Udacity/Computer-Pointer-Controller-master/bin/demo.mp4
'''

'''
Linux
source /opt/intel/openvino/bin/setupvars.sh
python3 computer_pointer.py \
--video /home/thomas/PycharmProjects/Intel/Computer-Pointer-Controller-master/src/demo.mp4 \
--output_path /home/thomas/PycharmProjects/Intel/Computer-Pointer-Controller-master/src/demo_output.mp4 \
-fd_model /home/thomas/PycharmProjects/Intel/Computer-Pointer-Controller-master/models/face-detection-adas-0001 \
-fl_model /home/thomas/PycharmProjects/Intel/Computer-Pointer-Controller-master/models/landmarks-regression-retail-0009 \
-hp_model /home/thomas/PycharmProjects/Intel/Computer-Pointer-Controller-master/models/head-pose-estimation-adas-0001 \
-ga_model /home/thomas/PycharmProjects/Intel/Computer-Pointer-Controller-master/models/gaze-estimation-adas-0002 \
--input_type video \
--version 2020
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
from mouse_controller import MouseController


'''
Used following resources:
https://knowledge.udacity.com/questions/254779 output gaze estimation
https://knowledge.udacity.com/questions/171017
https://knowledge.udacity.com/questions/257811 (move mouse)
'''

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
    facedetection = Facedetection(model_name=args.fd_model, threshold=args.threshold, device=args.device, extension=args.extension, version=args.version)
    print("Load class Facedetection = OK")
    print("--------")
    # Load model Facedetection
    facedetection.load_model()
    print("Load model facedetection = Finished")
    print("--------")
    
    # Load facial landmark
    faciallandmarks = Facial_Landmarks(model_name=args.fl_model, threshold=args.threshold, device=args.device, extension=args.extension, version=args.version)
    print("Load class Facial_Landmarks = OK")
    print("--------")
    faciallandmarks.load_model()
    print("Load model Facial_Landmarks = Finished")
    print("--------")
    
    # Load head_pose_estimation
    headposeestimation = Head_Pose_Estimation(model_name=args.hp_model, device=args.device, extension=args.extension, version=args.version)
    print("Load class head_pose_estimation = OK")
    print("--------")
    headposeestimation.load_model()
    print("Load model head_pose_estimation = Finished")
    print("--------")
    
    # Load gaze_estimation
    gazeestimation = Gaze_Estimation(model_name=args.ga_model, device=args.device, extension=args.extension, version=args.version)
    print("Load class gaze_estimation = OK")
    print("--------")
    gazeestimation.load_model()
    print("Load model gaze_estimation = Finished")
    print("--------")
    
    ##############
    feed = InputFeeder(input_type, input_file)
    feed.load_data()
    initial_w = int(feed.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(feed.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(feed.cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, fps, (initial_w, initial_h))
    facedetection.get_initial_w_h (initial_w, initial_h)
    #headposeestimation.get_initial_w_h (initial_w, initial_h)
    for batch in feed.next_batch():
        if batch is None:
            break

        # facedetection
        print("Start facedetection (computer_pointer.py)")
        print("Cap is feeded to the face detection!")
        face_batch = batch.copy()
        face_image, face_cropped = facedetection.predict(face_batch)
        print("The video from the face detection is writen to the output path")
        out_video.write(face_image)
        print("End facedetection (computer_pointer.py)")

        # faciallandmark
        if (face_cropped is None) or (len(face_cropped)==0):
            print("No Face above threshold detected")
        else:
            print("Start faciallandmark (computer_pointer.py)")
            print("The cropped face image is feeded to the faciallandmarks detection.")
            faciallandmarks.get_initial_w_h(face_cropped)
            left_eye_image, right_eye_image = faciallandmarks.predict(face_cropped)
            print("End faciallandmarks (computer_pointer.py)")

            # headposeestimation
            print("Start headposeestimation (computer_pointer.py)")
            print("The cropped face image is feeded to the headposeestimation.")
            headposeestimation.get_initial_w_h(face_cropped)
            head_pose_angles = headposeestimation.predict(face_cropped)
            print("Head pose angeles: ", head_pose_angles)
            print("End faciallheadposeestimationandmarks (computer_pointer.py)")

            # gazeestimation
            print("Start gazeestimation (computer_pointer.py)")
            # print ("Head pose angles", head_pose_angles)
            # print ("Head pose left_eye_image", left_eye_image)
            gaze_result, tmpX, tmpY, gaze_vector02 = gazeestimation.predict(left_eye_image, right_eye_image,
                                                                            head_pose_angles)
            print("End gazeestimation (computer_pointer.py)")
            print(gaze_result)
            log.info("Gaze results: ({})".format(str(gaze_result)))

            # mouse controller
            mousecontroller = MouseController('high', 'fast')
            mousecontroller.move(tmpX, tmpY)

    input_feed.close()
    cv2.destroyAllWindows()


def old():
    try:
        while cap.isOpened():
            result, frame = cap.read()
            if not result:
                break
            print ("Start facedetection (computer_pointer.py)")
            print ("Cap is feeded to the face detection!")
            face_frame = frame.copy()
            face_image, face_cropped = facedetection.predict(face_frame)
            print ("The video from the face detection is writen to the output path")
            out_video.write(face_image)
            print ("End facedetection (computer_pointer.py)")

            # faciallandmark
            if face_cropped is None:
                print ("No Face above threshold detected")
            else:
                print ("Start faciallandmark (computer_pointer.py)")
                print("The cropped face image is feeded to the faciallandmarks detection.")
                faciallandmarks.get_initial_w_h(face_cropped)
                left_eye_image, right_eye_image = faciallandmarks.predict(face_cropped)
                print ("End faciallandmarks (computer_pointer.py)")

                # headposeestimation
                print ("Start headposeestimation (computer_pointer.py)")
                print("The cropped face image is feeded to the headposeestimation.")
                headposeestimation.get_initial_w_h(face_cropped)
                head_pose_angles = headposeestimation.predict(face_cropped)
                print ("Head pose angeles: ", head_pose_angles)
                print ("End faciallheadposeestimationandmarks (computer_pointer.py)")

                # gazeestimation
                print ("Start gazeestimation (computer_pointer.py)")
                #print ("Head pose angles", head_pose_angles)
                #print ("Head pose left_eye_image", left_eye_image)
                gaze_result, tmpX, tmpY, gaze_vector02  = gazeestimation.predict(left_eye_image, right_eye_image, head_pose_angles)
                print ("End gazeestimation (computer_pointer.py)")
                print (gaze_result)
                log.info("Gaze results: ({})".format(str(gaze_result)))

                # mouse controller
                mousecontroller = MouseController('low', 'fast')
                mousecontroller.move(tmpX,tmpY)
            
            
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
    
    '''
    
def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=False)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--extension', default='/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so')
    parser.add_argument('--video', default=None)
    parser.add_argument('--output_path', default='demo_output.mp4')
    parser.add_argument('--threshold', default=0.10)
    parser.add_argument("-fd_model", default='models/face-detection-retail-0004', required=False)
    parser.add_argument("-fl_model", default='models/landmarks-regression-retail-0009', required=False)
    parser.add_argument("-hp_model", default='models/head-pose-estimation-adas-0001', required=False)
    parser.add_argument("-ga_model", default='models/gaze-estimation-adas-0002', required=False)
    parser.add_argument('--input_type', default='video', required=False)
    parser.add_argument('--version', default='2020', required=False)

    return parser

if __name__ == '__main__':
    log.basicConfig(filename="logging_basic.txt", level=log.INFO)
    log.info("Start computer_pointer.py")
    main()
