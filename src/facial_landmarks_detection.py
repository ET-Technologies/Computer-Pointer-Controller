'''
Udacity Workspace
source /opt/intel/openvino/bin/setupvars.sh
cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader
Model Downloader python3 downloader.py --name landmarks-regression-retail-0009 --precisions FP32 -o /home/workspace

python3 facial_landmarks_detection.py --model models/landmarks-regression-retail-0009 --device CPU --video demo.mp4 --output_path demo_output.mp4 --inputtype video
'''

'''
Rapberry Pi
python3 facial_landmarks_detection.py \
--model /home/pi/Udacity/Computer-Pointer-Controller-master/models/landmarks-regression-retail-0009 \
--device MYRIAD \
--video /home/pi/Udacity/Computer-Pointer-Controller-master/bin/demo.mp4 \
--output_path demo_output.mp4 \
--version 2020 \
--inputtype cam

Rapberry Pi
python3 facial_landmarks_detection.py \
--model /home/pi/Udacity/Computer-Pointer-Controller-master/models/landmarks-regression-retail-0009 \
--device MYRIAD \
--video /home/pi/Udacity/Computer-Pointer-Controller-master/bin/demo.mp4 \
--output_path demo_output.mp4 \
--version 2020 \
--inputtype image


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
    def __init__(self, model_name, threshold, device, extension, version):
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.extension = extension
        self.device = device
        self.threshold = threshold
        self.version = version
        
        print("--------")
        print("START Facial_Landmarks")
        print (self.model_weights)
        print ('device ', device)
        print("--------")

    
    def load_model(self):
        # Loads the model

        # Initialise the network and save it in the self.model variables
        try:
            self.core = IECore()
            self.network = IENetwork(self.model_structure, self.model_weights)
            # self.network = core.read_network(self.model_structure, self.model_weights) # new openvino version
            self.input_name = next(iter(self.network.inputs))
        except Exception as e:
            log.error("Could not initialise the network")
            raise ValueError("Could not initialise the network")
        print("--------")
        print("Model is loaded as self.model: " + str(self.network))
        print (self.extension)

        # Add extension
        if self.extension and "CPU" in self.device:
            log.info("Add extension: ({})".format(str(self.extension)))
            self.core.add_extension(self.extension, self.device)
            print ("Load extension")
        
        # Check supported layers
        self.check_model()
        # Load the network into an executable network
        self.exec_network = self.core.load_network(network=self.network, device_name=self.device, num_requests=1)
        log.info("Exec_network is loaded as:" + str(self.exec_network))
        print("Exec_network is loaded as:" + str(self.exec_network))
        print("--------")

        model_data = [self.model_weights, self.model_structure, self.device, self.extension, self.threshold, self.core, self.network]
        modellayers = self.getmodellayers()

        return model_data, modellayers

    def getmodellayers(self):
        # Get all necessary model values. 
        self.input_name = next(iter(self.network.inputs))
        self.output_name = next(iter(self.network .outputs))

        # Gets all input_names. Just for information.
        self.input_name_all = [i for i in self.network.inputs.keys()]
        self.input_name_all_02 = self.network .inputs.keys() # gets all output_names
        self.input_name_first_entry = self.input_name_all[0]
        
        self.input_shape = self.network .inputs[self.input_name].shape
        
        self.output_name_type = self.network .outputs[self.output_name]
        self.output_names = [i for i in self.network .outputs.keys()]  # gets all output_names
        self.output_names_total_entries = len(self.output_names)

        self.output_shape = self.network .outputs[self.output_name].shape
        self.output_shape_second_entry = self.network .outputs[self.output_name].shape[1]
        #model_info = ("model_weights: {}\nmodel_structure: {}\ndevice: {}\nextension: {}\nthreshold: {}\n".format.str(self.model_weights), str(self.model_structure), str(self.device), str(self.extension, str(self.threshold)))
        modellayers = [self.input_name, self.input_name_all, self.input_name_all_02,  self.input_name_first_entry, self.input_shape, self.output_name, self.output_name_type, \
            self.output_names, self.output_names_total_entries, self.output_shape, self.output_shape_second_entry]

        return modellayers

    def check_model(self):
        # Check for supported layers
        log.info("Checking for unsupported layers")
        if "CPU" in self.device:
            supported_layers = self.core.query_network(self.network, "CPU")
            print("--------")
            print("Check for supported layers")
            #print("supported_layers: " + str(supported_layers))
            not_supported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]          
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported:", not_supported_layers)
                print("You are not lucky, not all layers are supported")
                sys.exit(1)
        log.info("All layers are supported")
        print("All layers are supported")
        print("--------")

    # Start inference and prediction
    def predict(self, face_cropped):

        print("--------")
        print("Start predictions Facial_Landmarks")
        
        # Pre-process the image
        preprocessed_image = self.preprocess_input(face_cropped)
        
        # Starts synchronous inference
        print("Start syncro inference")
        log.info("Start syncro inference")
       
        outputs = self.exec_network.infer({self.input_name:preprocessed_image})
        print("Output of the inference request: " + str(outputs))
        
        requestid = 0
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
        log.info("Start: preprocess image face")
        n, c, h, w = (self.core, self.input_shape)[1]
        preprocessed_image = cv2.resize(frame, (w, h))
        preprocessed_image = preprocessed_image.transpose((2, 0, 1))
        preprocessed_image = preprocessed_image.reshape((n, c, h, w))
        print("The input shape from the facial landmarks is n= ({})  c= ({})  h= ({})  w= ({})".format(str(n),str(c), str(h), str(w)))
        print("Image is now [BxCxHxW]: " + str(preprocessed_image.shape))
        print("End: preprocess image")
        print("--------")

        return preprocessed_image

    def landmark_detection(self, outputs, frame):
        print("--------")
        print("Start: landmark_detection")
        result_len = len(outputs)
        print("total number of entries: " + str(result_len))
        self.initial_w = frame.shape[1]
        self.initial_h = frame.shape[0]
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
        print (w, h, c)
        print ("Reshape image")
       # left_eye_frame_cropped = cv2.resize(left_eye_frame_cropped, (60, 60))
       # left_eye_frame_cropped = left_eye_frame_cropped.transpose((2, 0, 1))
       # left_eye_frame_cropped = left_eye_frame_cropped.reshape((1, 2, 60, 60))

        cv2.imwrite("left_eye_frame_cropped.png", left_eye_frame_cropped)
        cv2.imwrite("right_eye_frame_cropped.png", right_eye_frame_cropped)
        print("--------")
        print("End: preprocess_output")
        return left_eye_frame_cropped, right_eye_frame_cropped

    def load_data(self, input_type, input_file):

        print ("Start load_data from InputFeeder")
        if input_type=='video':
            cap=cv2.VideoCapture(input_file)
            print ("Input = video")
            log.info("Input = video")
        elif input_type=='cam':
            cap=cv2.VideoCapture(0)
            print ("Input = cam")
            log.info("Input = cam")
        else:
            cap=cv2.imread(input_file)
            print ("Input = image")
            log.info("Input = image")

    def start(self, frame, inputtype):
          # Start predictions
        if inputtype == 'video' or 'cam':
            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = self.predict(frame)
                    cap.release()
            except Exception as e:
                print("Could not run Inference: ", e)

        if inputtype == 'image':
            print("Image")
            #image = '/home/pi/KeyBox/face_test.jpg'
            #frame=cv2.imread(image)
            frame = self.predict(frame)
            path = '/home/pi/KeyBox/Face_cropped image.png'
            image = cv2.imread(path)
            cv2.imshow("test", image)
            cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    args = build_argparser().parse_args()
    model_name = args.model
    device = args.device
    extension = args.extension
    video = args.video
    video = ("cropped_image.png")
    #video = ("face_full_image.png")
    output_path = args.output_path
    threshold = args.threshold
    inputtype = args.inputtype
    version = args.version

    # Load class Facial_Landmarks
    inference = Facial_Landmarks(model_name, threshold, device, extension, version)
    print("Load class Facial_Landmarks = OK")
    print("--------")

    # Loads the model
    # Time to load the model (Start)
    start_model_load_time = time.time()  
    model_data, modellayers = inference.load_model()
    # Time model needed to load
    total_model_load_time = time.time() - start_model_load_time  
    print("Load Model = OK")
    print("Time to load model: " + str(total_model_load_time))
    print("--------")
    
    # Load data (video, cam or image)
    cap = inference.load_data(inputtype, video)

    #  Start predictions

    if inputtype == 'video' or 'cam':
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = inference.predict(frame)
            cap.release()
                
        except Exception as e:
            print("Could not run Inference: ", e)
            
    if inputtype == 'image':
        print("Image")
        frame=cv2.imread(video)
        frame = inference.predict(frame)
        path = '/home/pi/Udacity/Computer-Pointer-Controller-master/src/landmark_image.png'
        #path = '/home/pi/Udacity/Computer-Pointer-Controller-master/src/landmark_image.png'
        image = cv2.imread(path)
        cv2.imshow("test", image)
        cv2.waitKey(0) 

    cv2.destroyAllWindows()
    
def build_argparser():
    # Collect all the necessary input values
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device')
    parser.add_argument('--extension', default='/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so')
    parser.add_argument('--video', default=None)
    parser.add_argument('--output_path', default='results/')
    parser.add_argument('--threshold', default=0.60)
    parser.add_argument('--inputtype', default='video')
    parser.add_argument('--version', default='2020')

    return parser


if __name__ == '__main__':
    # Start program
    log.basicConfig(filename="logging_landmarks.txt", level=log.INFO)
    log.info("Start logging")
    main()
