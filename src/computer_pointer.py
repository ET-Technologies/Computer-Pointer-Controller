'''
Linux
source /opt/intel/openvino/bin/setupvars.sh

python3 src/computer_pointer.py \
--video bin/demo.mp4 \
--output_path output/demo_output.mp4 \
--fd_model models/2020.4.1/FP32-INT1/face-detection-adas-binary-0001 \
--fl_model models/2020.4.1/FP16-INT8/landmarks-regression-retail-0009 \
--hp_model models/2020.4.1/FP16-INT8/head-pose-estimation-adas-0001 \
--ga_model models/2020.4.1/FP16-INT8/gaze-estimation-adas-0002 \
--threshold 0.6 \
--input_type video \
--device CPU \
--version 2020
'''

import time
import argparse
import cv2
import logging as log
formatter = log.Formatter('%(asctime)s %(levelname)s %(message)s')
import time

import openvino

from face_detection import Facedetection
#from facial_landmarks_detection import Facial_Landmarks

from facial_landmarks_detection_copy import Facial_Landmarks
from head_pose_estimation import Head_Pose_Estimation
from gaze_estimation import Gaze_Estimation

from input_feeder import InputFeeder
from mouse_controller import MouseController


'''
I used following resources:
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
    threshold = args.threshold
    extension = args.extension
    version = args.version
    device = args.device
    face_model = args.fd_model
    facial_model = args.fl_model
    headpose_model = args.hp_model
    gaze_model = args.ga_model

    # Start logger   
    # Basic logger
    log = setup_logger('basic_logger', 'log/logging_basic.log')
    log.info("Start computer_pointer.py")
    # Time logger
    log_time = setup_logger('time_logger', "log/logging_time.log")
    log_time.info("Start time logger")
    
    # Basic logging
    #log.basicConfig(filename="log/logging_basic.log", level=log.DEBUG)
    #log.info("Start computer_pointer.py")

    # Get Openvinoversion
    openvino_version = (openvino.__file__)
    print ("Openvino version: "+ str(openvino_version))
        
    # Load class Facedetection
    
    facedetection = Facedetection(face_model, threshold, device, extension, version)
    
    print("Load class Facedetection = OK")
    print("--------")
    start_load_time_face = time.time()
    # Load model Facedetection
    facedetection.load_model()
    print("Load model facedetection = Finished")
    print("--------")
    total_model_load_time_face = (time.time() - start_load_time_face)*1000
    log_time.info('Facedetection load time: ' + str(round(total_model_load_time_face, 3)))
    #print('Facedetection load time: ', str(total_model_load_time_face))
    
    # Load facial landmark
    faciallandmarks = Facial_Landmarks(facial_model, threshold, device, extension, version)
    print("Load class Facial_Landmarks = OK")
    print("--------")
    start_load_time_facial = time.time()
    faciallandmarks.load_model()
    print("Load model Facial_Landmarks = Finished")
    print("--------")
    total_model_load_time_facial = (time.time() - start_load_time_facial)*1000
    log_time.info('Facial_Landmarks load time: ' + str(round(total_model_load_time_facial, 3)))
    
    # Load head_pose_estimation
    headposeestimation = Head_Pose_Estimation(headpose_model, device, extension, version, threshold)
    print("Load class head_pose_estimation = OK")
    print("--------")
    start_load_time_headpose = time.time()
    headposeestimation.load_model()
    print("Load model head_pose_estimation = Finished")
    print("--------")
    total_model_load_time_headpose = (time.time() - start_load_time_headpose)*1000
    log_time.info('Headpose load time: ' + str(round(total_model_load_time_headpose, 3)))
    
    # Load gaze_estimation
    gazeestimation = Gaze_Estimation(gaze_model, threshold, device, extension, version)
    print("Load class gaze_estimation = OK")
    print("--------")
    start_load_time_gaze = time.time()
    gazeestimation.load_model()
    print("Load model gaze_estimation = Finished")
    print("--------")
    total_model_load_time_gaze = (time.time() - start_load_time_gaze)*1000
    total_model_load_time = (time.time() - start_load_time_face)*1000
    
    log_time.info('Gaze load time: ' + str(round(total_model_load_time_gaze, 3)))
    log_time.info('Total model load time: ' + str(round(total_model_load_time, 3)))
    log_time.info('##################')
    log.info('All models are loaded!')
    
    feed = InputFeeder(input_type, input_file)
    log.info('Input Feeder is loaded')
    feed.load_data()

    # Output video
    initial_w = int(feed.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(feed.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(feed.cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, fps, (initial_w, initial_h))

    inference_time_face_total = []
    inference_time_facial_total = []
    inference_time_headpose_total = []
    inference_time_gaze_total = []

    start_inference_time = time.time()
    try:
        for batch in feed.next_batch():
            if batch is None:
                break
            
            
            ## facedetection ##
            ## Inference time
            
            start_inference_time_face = time.time()
            print("Start facedetection")
            print("Cap is feeded to the face detection!")
            face_batch = batch.copy()
            face_image, face_cropped, coords= facedetection.predict(face_batch)
            
            #Average inference time
            inference_time_face = (time.time() - start_inference_time_face)*1000
            inference_time_face_total.append(inference_time_face)
            len_face = len(inference_time_face_total)
            avg_inference_time_face = sum(inference_time_face_total)/len_face
            log_time.info(('Average face inference time: ' + str(avg_inference_time_face)))
            
            if not coords:
                print("No face detected")
                continue
            
            print("The video from the face detection is writen to the output path")
            out_video.write(face_image)
            print("End facedetection")

            ## faciallandmark ##
            ## Inference time
            start_inference_time_facial = time.time()
            if (face_cropped is None) or (len(face_cropped)==0):
                print("No Face above threshold detected")
            else:
                print("Start faciallandmark")
                print("The cropped face image is feeded to the faciallandmarks detection.")
                #faciallandmarks.get_initial_w_h(face_cropped)
                left_eye_image, right_eye_image = faciallandmarks.predict(face_cropped.copy())
                print("End faciallandmarks")

                #Average inference time
                inference_time_facial = (time.time() - start_inference_time_facial)*1000
                inference_time_facial_total.append(inference_time_facial)
                len_facial = len(inference_time_facial_total)
                avg_inference_time_facial = sum(inference_time_facial_total)/len_facial
                log_time.info(('Average facial inference time: ' + str(avg_inference_time_facial)))

                # headposeestimation
                ## Inference time
                
                start_inference_time_headpose = time.time()
                print("Start headposeestimation")
                print("The cropped face image is feeded to the headposeestimation.")
                head_pose_angles = headposeestimation.predict(face_cropped)
                #print("Head pose angeles: ", head_pose_angles)
                print("End faciallheadposeestimationandmarks")

                #Average inference time
                inference_time_headpose = (time.time() - start_inference_time_headpose)*1000
                inference_time_headpose_total.append(inference_time_headpose)
                len_headpose = len(inference_time_headpose_total)
                avg_inference_time_headpose = sum(inference_time_headpose_total)/len_headpose
                log_time.info(('Average headpose inference time: ' + str(avg_inference_time_headpose)))

                # gazeestimation
                ## Inference time
                
                start_inference_time_gaze = time.time()
                print("Start gazeestimation")
                gaze_result, tmpX, tmpY, gaze_vector02 = gazeestimation.predict(left_eye_image, right_eye_image,
                                                                                head_pose_angles)
                print("End gazeestimation")
                #print('Gaze results:', gaze_result)
                log.info("Gaze results: ({})".format(str(gaze_result)))

                #Average inference time
                inference_time_gaze = (time.time() - start_inference_time_gaze)*1000
                inference_time_gaze_total.append(inference_time_gaze)
                len_gaze = len(inference_time_gaze_total)
                avg_inference_time_gaze = sum(inference_time_gaze_total)/len_gaze
                log_time.info(('Average gaze inference time: ' + str(avg_inference_time_gaze)))

                log_time.info('----')
                cv2.imshow('Test', face_cropped)
                cv2.waitKey(28)
                
                # mouse controller
                mousecontroller = MouseController('medium', 'fast')
                mousecontroller.move(tmpX, tmpY)

        input_feed.close()
        cv2.destroyAllWindows()
    #total_model_load_time = (time.time() - start_load_time_face)*1000
        end_inference_time = (time.time() - start_inference_time)*1000
        log.info('Total model inference time: ' + str(round(end_inference_time, 3)))

    except Exception as e:
        print ("Could not run Inference: ", e)
        log.error(e)

def setup_logger(name, log_file, level=log.INFO):

    handler = log.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = log.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def build_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fd_model", required=True)
    parser.add_argument("--fl_model", required=True)
    parser.add_argument("--hp_model", required=True)
    parser.add_argument("--ga_model", required=True)
    parser.add_argument('--device', default = 'CPU')
    parser.add_argument('--extension', default= None)
    parser.add_argument('--video', default=None)
    parser.add_argument('--output_path', required=False)
    parser.add_argument('--threshold', type=float, default=0.6)
    parser.add_argument('--input_type', required=False)
    parser.add_argument('--version', default='2020', required=False)

    return parser

if __name__ == '__main__':
    
    main()
