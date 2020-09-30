'''
Linux:
source /opt/intel/openvino/bin/setupvars.sh

python3 src/gaze_estimation.py \
--model models/2020.4.1/FP16/gaze-estimation-adas-0002 \
--device CPU \
--extension None \
--video bin/demo.mp4 \
--output_path output/demo_output.mp4 \
--threshold 0.6 \
--input_type video \
--version 2020
'''
import numpy as np
import os
import time
import cv2
import argparse
import sys
from openvino.inference_engine import IENetwork, IECore
import logging as log
import math

# I used following resources: 
# https://knowledge.udacity.com/questions/254779

class Gaze_Estimation:
    # Load all relevant variables into the class
    def __init__(self, model_name, threshold, device, extension, version):
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extension = extension
        self.threshold = threshold
        self.version = version

        print("--------")
        print("START Gaze_Estimation")
        print("--------")

    
    def load_model(self):
        # Loads the model

        # Initialise the network and save it in the self.model variables
        try:
            self.core = IECore()
            #self.network = self.core.read_network(model=self.model_structure, weights=self.model_weights) #new version
            self.network = IENetwork(model=self.model_structure, weights=self.model_weights)
            #log.info("Model is loaded as: ", self.network)
            self.input_name = next(iter(self.network.inputs))
        except Exception as e:
            log.error("Could not initialise the network")
            raise ValueError("Could not initialise the network")
        print("--------")
        print("Model is loaded as self.network : " + str(self.network))

        # Add extension
        if self.extension and "CPU" in self.device:
            log.info("Add extension: ({})".format(str(self.extension)))
            self.core.add_extension(self.extension, self.device)

        # Check supported layers
        self.check_model()
        # Load the network into an executable network
        self.exec_network = self.core.load_network(network=self.network, device_name=self.device, num_requests=1)
        #log.info("Exec_network is loaded as:" + str(self.exec_network))
        #print("Exec_network is loaded as:" + str(self.exec_network))
        #print("--------")

        model_data = [self.model_weights, self.model_structure, self.device, self.extension, self.threshold, self.core, self.network]
        modellayers = self.getmodellayers()

        return model_data, modellayers

    def getmodellayers(self):
        # Get all necessary model values. 
        self.input_name = next(iter(self.network.inputs))
        self.output_name = next(iter(self.network .outputs))
        self.input_shape = self.network .inputs[self.input_name].shape

        # Gets all inputs and outputs. Just for information.
        self.input_name_all = [i for i in self.network.inputs.keys()]
        self.input_name_all_02 = self.network .inputs.keys()
        self.input_name_first_entry = self.input_name_all[0]
        
        self.output_name_type = self.network .outputs[self.output_name]
        self.output_names = [i for i in self.network .outputs.keys()]
        self.output_names_total_entries = len(self.output_names)

        self.output_shape = self.network .outputs[self.output_name].shape
        self.output_shape_second_entry = self.network .outputs[self.output_name].shape[1]
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
            print("--------")
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported:", not_supported_layers)
                #print("Sorry, not all layers are supported")
                sys.exit(1)
        log.info("All layers are supported")
        #print("All layers are supported")

    
    def predict(self, left_eye, right_eye, head_pose_angles):
        # Start inference and prediction

        print("--------")
        print("Start predictions Gaze_Estimation")

        requestid = 0
        
        # Pre-process the images (left an right eye)
        left_eye_preprocess_image, right_eye_preprocess_image = self.preprocess_input(left_eye, right_eye)

        # Starts synchronous inference
        print("Start syncro inference")
        log.info("Start syncro inference")
        
        outputs = self.exec_network.infer({'left_eye_image': left_eye_preprocess_image, 'head_pose_angles': head_pose_angles, 'right_eye_image': right_eye_preprocess_image})
        print("Output of the inference request: " + str(outputs))
        
        outputs = self.exec_network.requests[requestid].outputs[self.output_name]
        print("Output of the inference request (self.output_name): " + str(outputs))
        
        mouse_coordinates, tmpX, tmpY, gaze_vector02 = self.gaze_estimation(outputs, head_pose_angles)
        #head_pose_results = self.preprocess_output(outputs)

        print("End predictions")
        print("--------")
        
        return mouse_coordinates, tmpX, tmpY, gaze_vector02 

    def preprocess_input(self, left_eye, right_eye):
        # In this function the original image is resized, transposed and reshaped to fit the model requirements.
        print("--------")
        print("Start preprocess image Gaze_Estimation")
        log.info("Start preprocess image Gaze_Estimation")
        n, c, h, w = 1, 3, 60, 60
        print("The input shape from the gaze estimation is n= ({})  c= ({})  h= ({})  w= ({})".format(str(n),str(c), str(h), str(w)))
        
        left_eye_image = left_eye.copy()
        print ("left_eye_image", left_eye_image)
        left_eye_preprocess_image = cv2.resize(left_eye_image, (w, h))
        print ("left_eye_preprocess_image 1", left_eye_preprocess_image)
        left_eye_preprocess_image = np.transpose(np.expand_dims(left_eye_preprocess_image, axis=0), (0, 3, 1, 2))
        print ("left_eye_preprocess_image 2", left_eye_preprocess_image)
        #left_eye_preprocess_image = left_eye.transpose((2, 0, 1))
        #left_eye_preprocess_image = left_eye.reshape((n, c, h, w))
        
        right_eye_image = right_eye.copy()
        right_eye_preprocess_image = cv2.resize(right_eye, (w, h))
        right_eye_preprocess_image = np.transpose(np.expand_dims(right_eye_preprocess_image, axis=0), (0, 3, 1, 2))
        #right_eye_preprocess_image = right_eye.transpose((2, 0, 1))
        #right_eye_preprocess_image = right_eye.reshape((n, c, h, w))
 
        print("End preprocess image")
        log.info("End preprocess image")
        print("--------")
        
        return left_eye_preprocess_image, right_eye_preprocess_image

    def gaze_estimation(self, outputs, head_pose_angles):
        print("--------")
        print("Start gaze_estimation")
        log.info("Start gaze_estimation")
        result_len = len(outputs)
        print("total number of entries: " + str(result_len))

        gazes =[]
        gaze_vector = self.exec_network.requests[0].outputs['gaze_vector']
        print("Output of the inference request (self.gaze_vector): " + str(gaze_vector))

        print("gaze_vector: " + str(gaze_vector))

        gaze_vector02 = outputs[0]
        print ("gaze_vector02", gaze_vector02)
        roll = gaze_vector02[2]
        print ("roll", roll)
        gaze_vector02 = gaze_vector02 / np.linalg.norm(gaze_vector02)
        cs = math.cos(roll * math.pi / 180.0)
        sn = math.sin(roll * math.pi / 180.0)
        tmpX = gaze_vector02[0] * cs + gaze_vector02[1] * sn
        tmpY = -gaze_vector02[0] * sn + gaze_vector02[1] * cs
        print (tmpX, tmpY, gaze_vector02)

        print("End gaze_estimation")
        log.info("End gaze_estimation")
        print("--------")

        return gaze_vector, tmpX, tmpY, gaze_vector02
    
    def getinputstream(self, left_eye, right_eye):
        try:
            right_eye_feed = cv2.imread(right_eye)
            left_eye_feed = cv2.imread(left_eye)
            right_eye_feed = cv2.imread(right_eye)
        except FileNotFoundError:
            print("Cannot find video file: " + video)
        except Exception as e:
            print("Something else went wrong with the video file: ", e)

        print("--------")
        print("Input video Data")
        print("--------")

        return left_eye_feed, right_eye_feed

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--extension', default=None)
    parser.add_argument('--video', default='video')
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--threshold', default=0.60)
    parser.add_argument('--inputtype', default='video')

    return parser

def main():
    args = build_argparser().parse_args()
    model_name = args.model
    device = args.device
    extension = args.extension
    video = args.video
    threshold = args.threshold
    inputtype = args.inputtype
    left_eye = args.left_eye_image
    right_eye = args.right_eye_image
    head_pose_angles  = args.head_pose_angles 
    output_path=args.output_path

    # Load class Gaze_Estimation
    inference = Gaze_Estimation(model_name, device, extension)
    print("Load Model = OK")
    print("--------")

    # Loads the model
    start_model_load_time = time.time()  # Time to load the model (Start)
    inference.load_model()
    total_model_load_time = time.time() - start_model_load_time  # Time model needed to load
    print("Time to load model: " + str(total_model_load_time))
    print("--------")

    # Get the input video stream and the coordinates of the gaze direction vector
    left_eye_feed, right_eye_feed = inference.getinputstream(left_eye, right_eye)
    
    
    # Gets the coordinates of gaze direction vector
    mouse_coordinates = inference.predict(left_eye_feed, right_eye_feed, head_pose_angles)
    print ("Coordinates for mouse are: ", mouse_coordinates)


if __name__ == '__main__':
    log.basicConfig(filename="log/logging_gaze.txt", level=log.DEBUG)
    log.info("Start logging")
    main()
