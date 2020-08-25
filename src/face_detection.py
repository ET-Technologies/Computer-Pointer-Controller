# source /opt/intel/openvino/bin/setupvars.sh
# cd /home/thomas/PycharmProjects/Intel/Computer-Pointer-Controller-master/src
# python3 face_detection.py --model /home/thomas/PycharmProjects/models/face-detection-retail-0004 --video demo.mp4
# --model /home/thomas/PycharmProjects/models/face-detection-retail-0004
# --video demo.mp4

# Udacity Workspace
# Model Downloader python3 downloader.py --name face-detection-retail-0004 --precisions FP32 -o /home/workspace
# python3 face_detection.py --model models/face-detection-retail-0004 --device CPU --video demo.mp4 --output_path demo_output.mp4
'''
Raspberry Pi
python3 face_detection.py --model /home/pi/Udacity/Computer-Pointer-Controller-master/models/face-detection-adas-0001 --device MYRIAD --extension None --video /home/pi/Udacity/Computer-Pointer-Controller-master/src/demo.mp4 --output_path /home/pi/Udacity/Computer-Pointer-Controller-master/src/demo_output.mp4
'''
#/home/pi/Udacity/Computer-Pointer-Controller-master/models/face-detection-adas-0001.xml


import numpy as np
import time
import os
import cv2
import argparse
import sys
from os import path
from openvino.inference_engine import IENetwork, IECore
#from input_feeder import InputFeeder
import logging as log

class Facedetection:
    
    # Load all relevant variables into the class
    def __init__(self, model_name, threshold, device, extension):
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extension = extension
        self.threshold = threshold
        
        print("--------")
        print("START Facedetection")
        print("model_weights: " + str(self.model_weights))
        print("model_structure: " + str(self.model_structure))
        print("device: " + str(self.device))
        print("extension: " + str(self.extension))
        print("threshold: " + str(self.threshold))
        print("--------")
    
    # Loads the model
    def load_model(self):

        # Initialise the network and save it in the self.network variables
        try:
            log.info("Reading model ...")
            self.network = IENetwork(model=self.model_structure, weights=self.model_weights)
            #self.network = IECore.read_network(self.model_structure, self.model_weights) #new version
            modelisloaded = True
        except Exception as e:
            modelisloaded = False
            raise ValueError ("Could not initialise the network")
        print("--------")
        print("Model is loaded as self.network : " + str(self.network ))
        
        if modelisloaded == True:

            # Get the input layer
            self.input_name = next(iter(self.network .inputs))
            # Gets all input_names
            self.input_name_all = [i for i in self.network .inputs.keys()]
            self.input_name_all_02 = self.network .inputs.keys()
            self.input_name_first_entry = self.input_name_all[0]
        
            self.input_shape = self.network .inputs[self.input_name].shape
        
            self.output_name = next(iter(self.network .outputs))
            self.output_name_type = self.network .outputs[self.output_name]
            self.output_names = [i for i in self.network .outputs.keys()]  # gets all output_names
            self.output_names_total_entries = len(self.output_names)

            self.output_shape = self.network .outputs[self.output_name].shape
            self.output_shape_second_entry = self.network .outputs[self.output_name].shape[1]

            print("--------")
            print("input_name: " + str(self.input_name))
            print("input_name_all: " + str(self.input_name_all))
            print("input_name_all_total: " + str(self.input_name_all_02))
            print("input_name_first_entry: " + str(self.input_name_first_entry))
            print("--------")

            print("input_shape: " + str(self.input_shape))
            print("--------")

            print("output_name: " + str(self.output_name))
            print("output_name type: " + str(self.output_name_type))
            print("output_names: " + str(self.output_names))
            print("output_names_total_entries: " + str(self.output_names_total_entries))
            print("--------")

            print("output_shape: " + str(self.output_shape))
            print("output_shape_second_entry: " + str(self.output_shape_second_entry))
            print("--------")
        
        self.core = IECore()
        # Add extension
        if "CPU" in self.device:
            log.info("Add extension: ({})".format(str(self.extension)))
            self.core.add_extension(self.extension, self.device)
        
        # Load the network into an executable network
        self.exec_network = self.core.load_network(network=self.network , device_name=self.device, num_requests=1)
        print("Exec_network is loaded as:" + str(self.exec_network))
        print("--------")
        self.check_model()
        
    
    # Start inference and prediction
    def predict(self, frame, initial_w, initial_h):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        print("--")
        print("Start predictions")
        self.width = initial_w
        self.height = initial_h
        requestid = 0
        preprocessed_image = self.preprocess_input(frame)
        # Starts synchronous inference
        print("Start syncro inference")
        log.info("Start syncro inference")
        outputs = self.exec_network.infer({self.input_name: preprocessed_image})
        print("Output of the inference request: " + str(outputs))
        outputs = self.exec_network.requests[requestid].outputs[self.output_name]
        print("Output of the inference request (self.output_name): " + str(outputs))
        processed_image = self.boundingbox(outputs, frame)
        print("End predictions")
        print("--------")
        return processed_image

    def check_model(self):
        ### TODO: Check for supported layers ###
        if "CPU" in self.device:
            #supported_layers = self.core.query_network(self.exec_network, "CPU")
            supported_layers = self.core.query_network(self.network, "CPU")
            print("--------")
            print("Check for supported layers")
            print("supported_layers: " + str(supported_layers)) 
            not_supported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
            print("not_supported_layers: " + str(not_supported_layers))
            print("You are lucky, all layers are supported")
            print("--------")
            if len(not_supported_layers) != 0:
                sys.exit(1)
            sys.exit(1)

    def preprocess_input(self, frame):
        # In this function the original image is resized, transposed and reshaped to fit the model requirements.
        print("--------")
        print("Start: preprocess image")
        log.info("Start: preprocess image")
        n, c, h, w = (self.core, self.input_shape)[1]
        image = cv2.resize(frame, (w, h))
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        print("Original image size is W= ({}) x H= ({})".format(str(self.width),str(self.height)))
        print("Image is now [BxCxHxW]: " + str(image.shape))
        print("End: preprocess image")
        print("--------")

        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        return image

    def boundingbox(self, outputs, frame):
        #coords = []
        print("--------")
        print("Start: boundingbox")
        print("Bounding box input: " + str(outputs))
        #print("Coords: " + str(coords))
        print("Original image size is (W x H): " + str(self.width) + "x" + str(self.height))
        for obj in outputs[0][0]:
            if obj[2] > self.threshold:
                obj[3] = int(obj[3] * self.width)
                obj[4] = int(obj[4] * self.height)
                obj[5] = int(obj[5] * self.width)
                obj[6] = int(obj[6] * self.height)
                #coords.append([obj[3], obj[4], obj[5], obj[6]])
                cv2.rectangle(frame, (obj[3], obj[4]), (obj[5], obj[6]), (0, 55, 255), 1)
                print("Bounding box output coordinates of frame: " + str(obj[3]) + " x " + str(obj[4]) + " x " + str(obj[5]) + " x " + str(obj[6]))
                self.xmin = int(obj[3])
                self.ymin = int(obj[4])
                self.xmax = int(obj[5])
                self.ymax = int(obj[6])

        print("End: boundingbox")
        print("--------")
        frame_cropped = frame.copy()
        #frame_cropped = frame_cropped[self.ymin:(self.ymax + 1), self.xmin:(self.xmax + 1)]
        cv2.imwrite("cropped image.png", frame_cropped)
        self.preprocess_output(frame)

        #frame = self.cropimage(frame)

        return frame

    def preprocess_output(self, frame):
        # crop image to fit the next model
        print("--------")
        print("Start: preprocess_output")
        print("Coordinates for cropped frame are xmin x ymin x xmax x ymax: " + str(
            self.xmin) + " x " + str(self.ymin) + " x " + str(self.xmax) + " x " + str(self.ymax))
        frame_cropped = None
        frame_cropped = frame[self.ymin:(self.ymax + 1), self.xmin:(self.xmax + 1)]
        cv2.imwrite("cropped_image.png", frame_cropped)
        print("--------")
        print("End: preprocess_output")
        return

    def videofile(self, video):
        if video =='video':
            input_type = 'video'
        else:
            input_type ='cam'

        return input_type
    
def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--extension', default='/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so')
    #parser.add_argument('--extension', default='/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so')
    parser.add_argument('--video', default=None)
    parser.add_argument('--output_path', default='demo_output.mp4')
    #parser.add_argument('--output_path', default='/home/pi/Udacity/Computer-Pointer-Controller-master/src/demo_output.mp4')
    parser.add_argument('--threshold', default=0.60)

    return parser    
    
def main():
    args = build_argparser().parse_args()
    model_name = args.model
    device = args.device
    extension = args.extension
    video = args.video
    output_path = args.output_path
    #output_path = ("/home/pi/Udacity/Computer-Pointer-Controller-master/src/demo.mp4")
    threshold = args.threshold
    print (extension)

    # Load class Facedetection
    inference = Facedetection(model_name, threshold, device, extension)
    print("Load class Facedetection = OK")
    print("--------")

    # Loads the model
    inference.load_model()
    print("Load Model = OK")
    print("--------")
    cap = cv2.VideoCapture(video)

    # Get the input video stream
    try:
        print("Reading video file", video)
        cap = cv2.VideoCapture(video)
        cap.open(video)
        if not path.exists(video):
            print("Cannot find video file: " + video)
    except FileNotFoundError:
        print("Cannot find video file: " + video)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)

    # Capture information about the input video stream
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print("--------")
    print("Input video Data")
    print("initial_w: " + str(initial_w))
    print("initial_h: " + str(initial_h))
    print("video_len: " + str(video_len))
    print("fps: " + str(fps))
    print("--------")

    # Define output video

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, fps, (initial_w, initial_h))

    try:
        while cap.isOpened():
            result, frame = cap.read()
            if not result:
                break
            image = inference.predict(frame, initial_w, initial_h)
            print("The video is writen to the output path")
            out_video.write(image)
    except Exception as e:
        print("Could not run Inference: ", e)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    log.basicConfig(filename="logging.txt", level=log.INFO)
    log.info("Start logging")
    main()