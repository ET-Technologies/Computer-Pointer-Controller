'''
Udacity Workspace
source /opt/intel/openvino/bin/setupvars.sh
cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader
Model Downloader python3 downloader.py --name landmarks-regression-retail-0009 --precisions FP32 -o /home/workspace

python3 facial_landmarks_detection.py --model models/landmarks-regression-retail-0009 --device CPU --video demo.mp4 --output_path demo_output.mp4 --inputtype video
'''

'''
Rapberry Pi

'''

import numpy as np
import time
import os
import cv2
import argparse
import sys
from openvino.inference_engine import IENetwork, IECore
from input_feeder import InputFeeder
import face_detection as fd
import logging as log

class Facial_Landmarks:

    # Load all relevant variables into the class
    def __init__(self, model_name, threshold, device, extension):
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.extension = extension
        self.device = device
        self.threshold = threshold
        
        print("--------")
        print("START Facial_Landmarks")
        print("model_weights: " + str(self.model_weights))
        print("model_structure: " + str(self.model_structure))
        print("device: " + str(self.device))
        print("extension: " + str(self.extension))
        print("--------")

    # Loads the model
    def load_model(self):

        # Initialise the network and save it in the self.model variables
        try:
            log.info("Reading model ...")
            self.network = IENetwork(self.model_structure, self.model_weights)
            # self.network = core.read_network(self.model_structure, self.model_weights) # new openvino version
            modelisloaded = True
        except Exception as e:
            modelisloaded = False
            raise ValueError("Could not initialise the network")
        print("--------")
        print("Model is loaded as self.model: " + str(self.network))

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

        # Adds Extension
        if 'CPU' in self.device:
            log.info("Add extension: ({})".format(str(self.extension)))
            self.core.add_extension(self.extension, self.device)
            
        # Load the network into an executable network
        self.exec_network = self.core.load_network(network=self.network, device_name=self.device, num_requests=1)
        print("Exec_network is loaded as:" + str(self.exec_network))
        print("--------")
        self.check_model()

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

    # Start inference and prediction
    def predict(self, face_cropped):

        print("--------")
        print("Start predictions Facial_Landmarks")
        #self.width = initial_w
        #self.height = initial_h
        requestid = 0
        # Pre-process the image
        preprocessed_image = self.preprocess_input(face_cropped)
        # Starts synchronous inference
        print("Start syncro inference")
        log.info("Start syncro inference")
        outputs = self.exec_network.infer({self.input_name:preprocessed_image})
        print("Output of the inference request: " + str(outputs))
        outputs = self.exec_network.requests[requestid].outputs[self.output_name]
        print("Output of the inference request (self.output_name): " + str(outputs))
        #landmark_results = self.landmark_detection(outputs, frame)
        
        left_eye, right_eye, left_eye_frame_cropped, right_eye_frame_cropped = self.landmark_detection(outputs, face_cropped)
        print("End predictions")
        print("--------")
        return left_eye_frame_cropped, right_eye_frame_cropped

    def preprocess_input(self, frame):
        # In this function the original image is resized, transposed and reshaped to fit the model requirements.
        print("--------")
        print("Start: preprocess image")
        frame_input = frame.copy()
        n, c, h, w = (self.core, self.input_shape)[1]
        image = cv2.resize(frame, (w, h))
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        print("The input shape from the facial landmarks is n= ({})  c= ({})  h= ({})  w= ({})".format(str(n),str(c), str(h), str(w)))
        #print("Original image size is (W x H): " + str(self.width) + "x" + str(self.height))
        print("Image is now [BxCxHxW]: " + str(image.shape))
        print("End: preprocess image")
        print("--------")
        return image

    def landmark_detection(self, outputs, frame):
        print("--------")
        print("Start: landmark_detection")
        result_len = len(outputs)
        print("total number of entries: " + str(result_len))
        coords = []
        for obj in outputs[0]:
            obj= obj[0]
            c = obj[0]
            print("Coordinaten: " + str(c))
            coords.append(c)
        print("Coords: " + str(coords))
        self.left_eye_coordinates_x = int(coords[0]*self.initial_w)
        self.left_eye_coordinates_y = int(coords[1]*self.initial_h)
        self.right_eye_coordinates_x = int(coords[2]*self.initial_w)
        self.right_eye_coordinates_y = int(coords[3]*self.initial_h)

        print ("initial_w: ", self.initial_w)
        print("left_eye_coordinates_x: " + str(self.left_eye_coordinates_x))
        print("left_eye_coordinates_y: " + str(self.left_eye_coordinates_y))

        #### Not necessary for gaze estimation 
        self.nose_coordinates_x = int(coords[4] * self.initial_w)
        self.nose_coordinates_y = int(coords[5] * self.initial_h)
        self.left_mouth_coordinates_x = int(coords[6] * self.initial_w)
        self.left_mouth_coordinates_y = int(coords[7] * self.initial_h)
        self.right_mouth_coordinates_x = int(coords[8] * self.initial_w)
        self.right_mouth_coordinates_y = int(coords[9] * self.initial_h)
        ####
        
        # left eye
        self.left_eye_x_min = self.left_eye_coordinates_x-30
        self.left_eye_x_max = self.left_eye_coordinates_x+30
        self.left_eye_y_min = self.left_eye_coordinates_y-30
        self.left_eye_y_max = self.left_eye_coordinates_y+30
        # right eye 
        self.right_eye_x_min = self.right_eye_coordinates_x-30
        self.right_eye_x_max = self.right_eye_coordinates_x+30
        self.right_eye_y_min = self.right_eye_coordinates_y-30
        self.right_eye_y_max = self.right_eye_coordinates_y+30
        
        print("Rectangle coordinates: ({}) + ({}) + ({}) + ({})".format(str(self.left_eye_x_min),str(self.left_eye_x_max), str(self.left_eye_y_min), str(self.left_eye_y_max)))
        #log.info("Add extension: ({})".format(str(CPU_EXTENSION)))
        left_eye, right_eye, left_eye_frame_cropped, right_eye_frame_cropped = self.draw_landmarks(frame)

        return left_eye, right_eye, left_eye_frame_cropped, right_eye_frame_cropped

    def draw_landmarks(self, frame):
        print("--------")
        print("Start: draw_landmarks")
        
        self.frame_original = frame.copy()
        left_eye_image = frame.copy()
        right_eye_image = frame.copy()

        center_left_eye = (self.left_eye_coordinates_x, self.left_eye_coordinates_y)
        center_right_eye = (self.right_eye_coordinates_x, self.right_eye_coordinates_y)
        center_nose= (self.nose_coordinates_x, self.nose_coordinates_y)
        left_mouth_coordinates = (self.left_mouth_coordinates_x, self.left_mouth_coordinates_y)
        right_mouth_coordinates = (self.right_mouth_coordinates_x, self.right_mouth_coordinates_y)
        image = cv2.circle(frame, center_left_eye, 10, (255,0,0), 2)
        image = cv2.circle(frame, center_right_eye, 10, (255, 0, 0), 2)
        image = cv2.circle(frame, center_nose, 10, (255, 0, 0), 2)
        image = cv2.circle(frame, left_mouth_coordinates, 10, (255, 0, 0), 2)
        image = cv2.circle(frame, right_mouth_coordinates, 10, (255, 0, 0), 2)
        self.image_path = ("landmark_image.png")
        cv2.imwrite(self.image_path, image)
        
        #Draw rectangle
        
        #image_rectangle = cv2.rectangle(frame_rectangle, start_point, end_point, color, thickness)
        #image_rectangle = cv2.rectangle(frame_rectangle, start_point, end_point, color, thickness)
        left_eye = cv2.rectangle(left_eye_image, (self.left_eye_x_min, self.left_eye_y_min), (self.left_eye_x_max, self.left_eye_y_max), (255,0,0), 2)
        right_eye = cv2.rectangle(right_eye_image, (self.right_eye_x_min, self.right_eye_y_min), (self.right_eye_x_max, self.right_eye_y_max), (255,0,0), 2)
        self.left_eye_image_rectangle_path = ("left_eye_image.png")
        self.right_eye_image_rectangle_path = ("right_eye_image.png")
        cv2.imwrite(self.left_eye_image_rectangle_path, left_eye)
        cv2.imwrite(self.right_eye_image_rectangle_path, right_eye)
        
        left_eye_frame_cropped, right_eye_frame_cropped = self.preprocess_output(self.frame_original)
        
        print("End: draw_landmarks")
        print("--------")
        return left_eye, right_eye, left_eye_frame_cropped, right_eye_frame_cropped
    
    def preprocess_output(self, frame):
        # crop image to fit the next model
        print("--------")
        print("Start: preprocess_output")
        print("Coordinates for cropped left eye are xmin x ymin x xmax x ymax: " + str(
            self.left_eye_x_min) + " x " + str(self.left_eye_y_min) + " x " + str(self.left_eye_x_max) + " x " + str(self.left_eye_y_max))
        print("Coordinates for cropped right eye are xmin x ymin x xmax x ymax: " + str(
            self.right_eye_x_min) + " x " + str(self.right_eye_y_min) + " x " + str(self.right_eye_x_max) + " x " + str(self.right_eye_y_max))
        left_eye_frame_cropped = None
        right_eye_frame_cropped = None
        left_eye_frame_cropped = frame[self.left_eye_y_min:(self.left_eye_y_max + 1), self.left_eye_x_min:(self.left_eye_x_max + 1)]
        right_eye_frame_cropped = frame[self.right_eye_y_min:(self.right_eye_y_max + 1), self.right_eye_x_min:(self.right_eye_x_max + 1)]

        # reshape image for gaze estimation
        w = left_eye_frame_cropped.shape[1]
        h = left_eye_frame_cropped.shape[0]
        c = left_eye_frame_cropped.shape[2]
        n = left_eye_frame_cropped.shape[3]
        print (w, h, c, n)
        print ("Reshape image")
        left_eye_frame_cropped = cv2.resize(left_eye_frame_cropped, (60, 60))
        left_eye_frame_cropped = left_eye_frame_cropped.transpose((3, 0, 1))
        left_eye_frame_cropped = left_eye_frame_cropped.reshape((1, 3, 60, 60))

        right_eye_frame_cropped = cv2.resize(right_eye_frame_cropped, (60, 60))
        right_eye_frame_cropped = right_eye_frame_cropped.transpose((3, 0, 1))
        right_eye_frame_cropped = right_eye_frame_cropped.reshape((1, 3, 60, 60))
        #1x3x60x60

        cv2.imwrite("left_eye_frame_cropped.png", left_eye_frame_cropped)
        cv2.imwrite("right_eye_frame_cropped.png", right_eye_frame_cropped)
        print("--------")
        print("End: preprocess_output")
        return left_eye_frame_cropped, right_eye_frame_cropped

    def getinputstream(self, inputtype, video, output_path):
        # gets the inputtype
        print("--------")
        print("Start: getinputstream")
        try:
            if inputtype == 'video':
                print("Reading video file:", video)
                cap = cv2.VideoCapture(video)
            elif inputtype =='cam':
                print("Reading webcam")
                cap = cv2.VideoCapture(0)
            else:
                print("Reading image:", video)
                cap = cv2.imread(video)    
        except FileNotFoundError:
            print("Cannot find video file: " + video)
        except Exception as e:
            print("Something else went wrong with the video file: ", e)
            
        # Capture information about the input video stream
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
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(output_path, fourcc, self.fps, (self.initial_w, self.initial_h))

        try:
            while cap.isOpened():
                result, frame = cap.read()
                if not result:
                    break
                #image = inference.predict(frame, initial_w, initial_h)
                image = self.predict(frame)
                print("The video is writen to the output path")
                out_video.write(image)
        except Exception as e:
            print("Could not run Inference: ", e)

            cap.release()
            cv2.destroyAllWindows()
            
        return
    
    def get_initial_w_h (self, frame_cropped):
        self.initial_w = frame_cropped.shape[1]
        self.initial_h = frame_cropped.shape[0]
        print("Initialize initial_w in facial landmarks: " + str(self.initial_w))
        print("Initialize initial_h in facial landmarks: " + str(self.initial_h))
    
# Collect all the necessary input values
def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--extension', default='/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so')
    parser.add_argument('--video', default=None)
    parser.add_argument('--output_path', default='results/')
    parser.add_argument('--threshold', default=0.60)
    parser.add_argument('--inputtype', default='video')

    return parser


def main():
    args = build_argparser().parse_args()
    model_name = args.model
    device = args.device
    extension = args.extension
    #video = args.video
    video = ("cropped_image.png")
    video = ("face_full_image.png")
    output_path = args.output_path
    threshold = args.threshold
    inputtype = args.inputtype

    # Load class Facial_Landmarks
    inference = Facial_Landmarks(model_name, threshold, device, extension)
    print("Load class Facial_Landmarks = OK")
    print("--------")

    # Loads the model
    start_model_load_time = time.time()  # Time to load the model (Start)
    inference.load_model()
    total_model_load_time = time.time() - start_model_load_time  # Time model needed to load
    print("Load Model = OK")
    print("Time to load model: " + str(total_model_load_time))
    print("--------")

    # Get the input video stream
    inference.getinputstream(inputtype, video, output_path)

# Start program
if __name__ == '__main__':
    log.basicConfig(filename="logging_landmarks.txt", level=log.INFO)
    log.info("Start logging")
    main()
