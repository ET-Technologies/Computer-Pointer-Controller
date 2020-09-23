# source /opt/intel/openvino/bin/setupvars.sh
# cd /home/thomas/PycharmProjects/Intel/Computer-Pointer-Controller-master/src
# python3 head_pose_estimation.py --model /home/thomas/PycharmProjects/models/head-pose-estimation-adas-0001 --video demo.mp4

#intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml

'''
# Udacity Workspace
# source /opt/intel/openvino/bin/setupvars.sh
# cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader
# Model Downloader python3 downloader.py --name gaze-estimation-adas-0002 --precisions FP32 -o /home/workspace
# python3 gaze_estimation.py --model models/gaze-estimation-adas-0002 --video demo.mp4
'''

'''
Windows
python3 gaze_estimation.py
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

    # Loads the model
    def load_model(self):

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
            print("--------")
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported:", not_supported_layers)
                print("You are not lucky, not all layers are supported")
                sys.exit(1)
        log.info("All layers are supported")
        print("All layers are supported")

    # Start inference and prediction
    def predict(self, left_eye, right_eye, head_pose_angles):

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
        print("Start: preprocess image Gaze_Estimation")
        log.info("Start: preprocess image")
        n = 1
        c = 3
        h = 60
        w = 60
        print("The input shape from the gaze estimation is n= ({})  c= ({})  h= ({})  w= ({})".format(str(n),str(c), str(h), str(w)))
        
        left_eye_image = left_eye.copy()
        print ("left_eye_image", left_eye_image)
        left_eye_preprocess_image = cv2.resize(left_eye_image, (w, h))
        print ("left_eye_preprocess_image 1", left_eye_preprocess_image)
        left_eye_preprocess_image = np.transpose(np.expand_dims(left_eye_preprocess_image, axis=0), (0, 3, 1, 2))
        print ("left_eye_preprocess_image 2", left_eye_preprocess_image)
        #left_eye_preprocess_image = left_eye.transpose((2, 0, 1))
        #left_eye_preprocess_image = left_eye.reshape((n, c, h, w))
        
        #right_eye_image = right_eye.copy()
        right_eye_preprocess_image = cv2.resize(right_eye, (w, h))
        right_eye_preprocess_image = np.transpose(np.expand_dims(right_eye_preprocess_image, axis=0), (0, 3, 1, 2))
        #right_eye_preprocess_image = right_eye.transpose((2, 0, 1))
        #right_eye_preprocess_image = right_eye.reshape((n, c, h, w))
        
        #print("Original image size is (W x H): " + str(self.width) + "x" + str(self.height))
        print("left_eye_preprocess_image is now [BxCxHxW]: " + str(left_eye_preprocess_image.shape))
        print("right_eye_preprocess_image is now [BxCxHxW]: " + str(right_eye_preprocess_image.shape))
        print("End: preprocess image")
        print("--------")
        
        return left_eye_preprocess_image, right_eye_preprocess_image

    def gaze_estimation(self, outputs, head_pose_angles):
        print("--------")
        print("Start: gaze_estimation")
        result_len = len(outputs)
        print("total number of entries: " + str(result_len))
        #output_name: gaze_vector
        gazes =[]
        gaze_vector = self.exec_network.requests[0].outputs['gaze_vector']
        print("Output of the inference request (self.gaze_vector): " + str(gaze_vector))
        #gaze_vector = int(gaze_vector)
        print("gaze_vector: " + str(gaze_vector))
        #gazes.append([gaze_vector])

        #print("gaze_vector: " + str(gazes))
        print("End: gaze_estimation")
        print("--------")

        # like https://knowledge.udacity.com/questions/254779

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

        return gaze_vector, tmpX, tmpY, gaze_vector02

    def preprocess_output(self, image):
        '''
        https://docs.openvinotoolkit.org/2019_R1/_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html
        The network takes three inputs: square crop of left eye image, square crop of right eye image, and three head pose angles – (yaw, pitch, and roll) 
        (see figure). The network outputs 3-D vector corresponding to the direction of a person’s gaze in a Cartesian coordinate system in which 
        z-axis is directed from person’s eyes (mid-point between left and right eyes’ centers) to the camera center, y-axis is 
        vertical, and x-axis is orthogonal to both z,y axes so that (x,y,z) constitute a right-handed coordinate system.
        '''

        print("--------")
        print("Start: head_pose_estimation")
        outputs = []
        outputs2 = []
        
        outputs.append(image['angle_y_fc'].tolist()[0][0])
        outputs2.append(image['angle_y_fc'][0][0])
        angle_y_fc = (image['angle_y_fc'][0][0])
        
        outputs.append(image['angle_p_fc'].tolist()[0][0])
        outputs2.append(image['angle_p_fc'][0][0])
        angle_p_fc = (image['angle_p_fc'][0][0])
        
        outputs.append(image['angle_r_fc'].tolist()[0][0])
        outputs2.append(image['angle_r_fc'][0][0])
        angle_r_fc = (image['angle_r_fc'][0][0])
        
        print ("outputs: " +str(outputs))
        print ("outputs2: " +str(outputs2))
        print ("outputs2: " +str(outputs2))
        print ("outputs: " +str(outputs))
        print ("outputs2: " +str(outputs2))
        print ("outputs: " +str(outputs))
        print ("outputs2: " +str(outputs2))
        print ("angle_y_fc: " +str(angle_y_fc))
        print ("angle_p_fc: " +str(angle_p_fc))
        print ("angle_r_fc: " +str(angle_r_fc))

        return outputs
    
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
    parser.add_argument('--model', default='./models/gaze-estimation-adas-0002')
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--extension', default='/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so')
    parser.add_argument('--video', default='demo.mp4')
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--threshold', default=0.60)
    parser.add_argument('--inputtype', default='video')
    parser.add_argument('--left_eye_image', default='left_eye_frame_cropped.png')
    parser.add_argument('--right_eye_image', default='right_eye_frame_cropped.png')
    parser.add_argument('--head_pose_angles', default = None)
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
    head_pose_angles = [2,5,10]
    #left_eye_image = ('cropped_image.png')
    #right_eye = ('left_eye_frame_cropped.png')

    # Load class Gaze_Estimation
    inference = Gaze_Estimation(model_name, device, extension)
    print("Load Model = OK")
    print("--------")

    # Loads the model
    start_model_load_time = time.time()  # Time to load the model (Start)
    inference.load_model()
    total_model_load_time = time.time() - start_model_load_time  # Time model needed to load
    print("Load Model = OK")
    print("Time to load model: " + str(total_model_load_time))
    print("--------")

    # Get the input video stream and the coordinates of the gaze direction vector
    left_eye_feed, right_eye_feed = inference.getinputstream(left_eye, right_eye)
    
    
    # Gets the coordinates of gaze direction vector
    mouse_coordinates = inference.predict(left_eye_feed, right_eye_feed, head_pose_angles)
    print ("Coordinates for mouse are: ", mouse_coordinates)


if __name__ == '__main__':
    log.basicConfig(filename="logging_gaze.txt", level=log.INFO)
    log.info("Start logging")
    main()
