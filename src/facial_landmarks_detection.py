'''
Linux:
source /opt/intel/openvino/bin/setupvars.sh

python3 src/facial_landmarks_detection.py \
--model models/2020.4.1/FP16-INT8/landmarks-regression-retail-0009 \
--device CPU \
--extension None \
--video bin/demo.mp4 \
--output_path output/demo_output.mp4 \
--threshold 0.6 \
--input_type video \
--version 2020
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

    def __init__(self, model_name, threshold, device, extension, version):
        # Load all relevant variables into the class
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extension = extension
        self.threshold = threshold
        self.version = version

        print("--------")
        print("START Facial Landmarks")
        print("--------")

    def load_model(self):
        # Loads the model
        
        # Initialise the network and save it in the self.network variables
        try:
            self.core = IECore()
            #self.network = self.core.read_network(self.model_structure, self.model_weights) #new version
            self.network = IENetwork(model=self.model_structure, weights=self.model_weights)
            #log.info("Model is loaded as: ", self.network)
            self.input_name = next(iter(self.network.inputs))
        except Exception as e:
            log.error("Could not initialise the network", e)
            raise ValueError("Could not initialise the network")
        print("--------")
        print("Model is loaded as self.network : " + str(self.network))

        # Add extension
        if "CPU" in self.device and (self.version == 2019):
            log.info("Add extension: ({})".format(str(self.extension)))
            self.core.add_extension(self.extension, self.device)

        # Check supported layers
        self.check_model()

        # Load the network into an executable network
        self.exec_network = self.core.load_network(network=self.network, device_name=self.device, num_requests=1)
        #log.info("Exec_network is loaded as:" + str(self.exec_network))
        #print("Exec_network is loaded as:" + str(self.exec_network))
        #print("--------")

        model_data = [self.model_weights, self.model_structure, self.device, self.extension, self.threshold]
        modellayers = self.getmodellayers()

        return model_data, modellayers

    def getmodellayers(self):
        # Get all necessary model values. 
        self.input_name = next(iter(self.network.inputs))
        self.output_name = next(iter(self.network .outputs))
        self.input_shape = self.network.inputs[self.input_name].shape

        # Gets all input_names. Just for information.
        self.input_name_all = [i for i in self.network.inputs.keys()]
        self.input_name_all_02 = self.network .inputs.keys() # gets all output_names
        self.input_name_first_entry = self.input_name_all[0]

        self.output_name_type = self.network.outputs[self.output_name]
        self.output_names = [i for i in self.network .outputs.keys()]  # gets all output_names
        self.output_names_total_entries = len(self.output_names)

        self.output_shape = self.network.outputs[self.output_name].shape
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
            print("--------")
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported:", not_supported_layers)
                #print("Sorry, not all layers are supported")
                sys.exit(1)
        log.info("All layers are supported")
        #print("All layers are supported")
    
    def predict(self, frame):
        # Starts predictions face_detection
        print("--------")
        print("Starts predictions for facial landmarks")
        log.info("Starts predictions for facial landmarks")

        # Pre-process the image
        preprocess_input = self.preprocess_input(frame.copy())

        # Starts synchronous inference
        print("Start syncro inference facial landmarks")
        log.info('Start syncro inference facial landmarks')

        outputs = self.exec_network.infer({self.input_name:preprocess_input})
        #print("#######################")
        #print("Output of the inference request: " + str(outputs))
        #print("#######################")
        
        # Gets coordinates for eyes, nose and lip
        coords, coords_nose, coords_lip = self.preprocess_output(outputs)
        print('---------------------')
        print(f'Coords eyes:{coords}')
        h, w = frame.shape[0], frame.shape[1]

        print(f'h: {h} w: {w}')
        
        ## eyes
        # left eye coordinates
        left_eye_xmin, left_eye_ymin=int(coords[0]*w-20), int(coords[1]*h-20)
        left_eye_xmax, left_eye_ymax=int(coords[0]*w+20), int(coords[1]*h+20)
        # right eye coordinates
        right_eye_xmin, right_eye_ymin=int(coords[2]*w-20), int(coords[3]*h-20)
        right_eye_xmax, right_eye_ymax=int(coords[2]*w+20), int(coords[3]*h+20)
        # left eye image
        left_eye_image = frame[left_eye_ymin:left_eye_ymax, left_eye_xmin:left_eye_xmax]
        # right exe image
        right_eye_image = frame[right_eye_ymin:right_eye_ymax, right_eye_xmin:right_eye_xmax]

        ## Nose
        nose_xmin, nose_ymin = int(coords_nose[0]*w -30), int(coords_nose[1]*h -30)
        nose_xmax, nose_ymax = int(coords_nose[0]*w +30), int(coords_nose[1]*h +30)
        #nose_coords = [nose_xmin, nose_ymin, nose_xmax, nose_ymax]
        #print (f'Coords nose: {nose_coords}')
        nose_image = frame[nose_ymin:nose_ymax, nose_xmin:nose_xmax]

        ## Lip
        lip_corner_right_xmin, lip_corner_right_ymin = int(coords_lip[0]*w -30), int(coords_lip[1]*h -30)
        lip_corner_right_xmax, lip_corner_right_ymax = int(coords_lip[0]*w +30), int(coords_lip[1]*h +30)
        #lip_corner_right_coords = [lip_corner_right_xmin, lip_corner_right_ymin, lip_corner_right_xmax, lip_corner_right_ymax]
        #print (f'Coords lip_corner_right_ymax: {lip_corner_right_coords}')
        lip_corner_right_image = frame[lip_corner_right_ymin:lip_corner_right_ymax, lip_corner_right_xmin:lip_corner_right_xmax]

        lip_corner_left_xmin, lip_corner_left_ymin = int(coords_lip[2]*w -30), int(coords_lip[3]*h -30)
        lip_corner_left_xmax, lip_corner_left_ymax = int(coords_lip[2]*w +30), int(coords_lip[3]*h +30)
        #lip_corner_left_coords = [lip_corner_left_xmin, lip_corner_left_ymin, lip_corner_left_xmax, lip_corner_left_ymax]
        #print (f'Coords lip_corner_left_ymax: {lip_corner_left_coords}')
        lip_corner_left_image = frame[lip_corner_left_ymin:lip_corner_left_ymax, lip_corner_left_xmin:lip_corner_left_xmax]

        return left_eye_image, right_eye_image, nose_image, lip_corner_left_image, lip_corner_right_image

    def preprocess_input(self, frame):
        # Preprcess input to feed into the model

        ## Itś important convert RGB to BGR 
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        initial_h, initial_w = self.input_shape[2], self.input_shape[3]
        print(f'h:{initial_h}, w:{initial_w}')

        frame_resized = cv2.resize(frame_bgr, (initial_w, initial_h))
        frame = frame_resized.transpose((2,0,1))
        frame_processed = frame.reshape(1, 3, initial_h, initial_w)
        log.info('Image is preprocessed: facial_landmarks')


        return frame_processed


    def preprocess_output(self, outputs):
        '''
        https://docs.openvinotoolkit.org/2019_R1/_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html
        The net outputs a blob with the shape: [1, 10], containing a row-vector of 
        10 floating point values for five landmarks coordinates in the form 
        (x0, y0, x1, y1, ..., x5, y5). All the coordinates are normalized to be in range [0,1].
        two eyes, nose, and two lip corners.
        '''
        print("Start: preprocess_output")
        print(outputs)

        outs = outputs[self.output_name][0]
        print("outs")

        # Eyes
        left_eye_x, left_eye_y = outs[0][0][0], outs[1][0][0]
        right_eye_x, right_eye_y = outs[2][0][0], outs[3][0][0]
        coords_lr = (left_eye_x, left_eye_y, right_eye_x, right_eye_y)
        print('##############')
        print(f'Coords eyes:{coords_lr}')

        # Nose
        nose_x, nose_y = outs[4][0][0], outs[5][0][0]
        coords_nose = (nose_x, nose_y)
        print(f'Coords nose:{coords_nose}')
        
        #Lip corners
        lip_corner_left_x, lip_corner_left_y = outs[7][0][0], outs[7][0][0]
        lip_corner_right_x, lip_corner_right_y = outs[8][0][0], outs[9][0][0]
        coords_lip = (lip_corner_left_x, lip_corner_left_y, lip_corner_right_x, lip_corner_right_y)
        print(f'Coords lips:{coords_lip}')
        print('#########################')

        return coords_lr, coords_nose, coords_lip


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--extension', default=None)
    parser.add_argument('--video', default=None)
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--threshold', default=0.60)
    parser.add_argument('--input_type', default=video)
    parser.add_argument('--version', default='2020')

    return parser

if __name__ == '__main__':
    # Start program
    log.basicConfig(filename="log/logging_facedetection.log", level=log.INFO)
    log.info("Start logging")
    main()
