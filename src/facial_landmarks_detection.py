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
            log.error("Could not initialise the network")
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

        # Pre-process the image
        preprocess_input = self.preprocess_input(frame.copy())

        # Starts synchronous inference
        print("Start syncro inference facial landmarks")
        log.info("Start syncro inference facial landmarks")

        outputs = self.exec_network.infer({self.input_name:preprocess_input})
        print("Output of the inference request: " + str(outputs))
        print ("finish")

        coords = self.preprocess_output(outputs)
        z = 20

        # print(image.shape)
        h, w = frame.shape[0], frame.shape[1]

        coords = coords* np.array([w, h, w, h])

        coords = coords.astype(np.int32) #(lefteye_x, lefteye_y, righteye_x, righteye_y)

        ## left eye moving range
        leye_xmin, leye_ymin=coords[0]-20, coords[1]-20
        leye_xmax, leye_ymax=coords[0]+20, coords[1]+20
        ## right eye moving range
        reye_xmin, reye_ymin=coords[2]-20, coords[3]-20
        reye_xmax, reye_ymax=coords[2]+20, coords[3]+20


        ## leye_ymin:leye_ymax, leye_xmin:leye_xmax --> left eye heigh, width
        left_eye_box = frame[leye_ymin:leye_ymax, leye_xmin:leye_xmax]
        ## reye_ymin:reye_ymax, reye_xmin:reye_xmax --> right eye heigh, width
        right_eye_box = frame[reye_ymin:reye_ymax, reye_xmin:reye_xmax]
        # print(left_eye_box.shape, right_eye_box.shape) # left eye and right eye image

        ## [left eye box, right eye box] 
        eyes_coords = [[leye_xmin,leye_ymin,leye_xmax,leye_ymax], [reye_xmin,reye_ymin,reye_xmax,reye_ymax]]

        return left_eye_box, right_eye_box #, eyes_coords

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        # print(image.shape)
        # print(image[2][1])
        # cv2.imshow('image',image)
        ## convert RGB to BGR 
        image_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(image_cvt.shape) # (374, 238, 3)
        # cv2.imshow('cvt',image_cvt)
        # print('====',image_cvt[2][1])
        # print(self.input_shape) # [1, 3, 48, 48]
        H, W = self.input_shape[2], self.input_shape[3]
        # print(H, W) # (48, 48)

        image_resized = cv2.resize(image_cvt, (W, H))
        # print(image_resized.shape) # (48, 48, 3)
        ## (optional)
        # image_processed = np.transpose(np.expand_dims(image_resized, axis=0), (0,3,1,2))
        image = image_resized.transpose((2,0,1))
        # print(image.shape) # (3, 48, 48)
        # add 1 dim at very start, then channels then H, W
        image_processed = image.reshape(1, 3, self.input_shape[2], self.input_shape[3])
        # print(image_processed.shape) # (1, 3, 48, 48)

        return image_processed


    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.

        The net outputs a blob with the shape: [1, 10], containing a row-vector of 10 floating point values 
        for five landmarks coordinates in the form (x0, y0, x1, y1, ..., x5, y5). 
        All the coordinates are normalized to be in range [0,1].
        '''
        print(outputs)
        # print(outputs[self.output_names].shape) # (1, 10, 1, 1)
        # print(outputs[self.output_names][0].shape) # (10, 1, 1)
        # print(outputs[self.output_names][0])        
        # print('-----', outputs[self.output_names][0][0])

        ## here only need left eye and right eye
        outs = outputs[self.output_names][0]
        print("outs")
        # print(outs.shape)
        # print(outs[0][0][0])
        # print(outs[0].tolist()) # [[0.37333157658576965]]
        # print(outs[0].tolist()[0][0]) # [[0.37333157658576965]]        
        # print(type(outs)) # numpy.ndarry

        leye_x, leye_y = outs[0][0][0], outs[1][0][0]
        reye_x, reye_y = outs[2][0][0], outs[3][0][0]
        coords_lr = (leye_x, leye_y, reye_x, reye_y)
        # print(coords_lr)

        return coords_lr
