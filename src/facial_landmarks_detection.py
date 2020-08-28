# source /opt/intel/openvino/bin/setupvars.sh
# cd /home/thomas/PycharmProjects/Intel/Computer-Pointer-Controller-master/src
# python3 head_pose_estimation.py --model /home/thomas/PycharmProjects/models/head-pose-estimation-adas-0001 --video demo.mp4

#intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml

'''
# Udacity Workspace
# cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader
# Model Downloader python3 downloader.py --name gaze-estimation-adas-0002 --precisions FP32 -o /home/workspace
# python3 gaze_estimation.py --model models/gaze-estimation-adas-0002 --video demo.mp4
'''
import numpy as np
import time
import os
import cv2
import argparse
import sys
from openvino.inference_engine import IENetwork, IECore
import logging as log

class Gaze_Estimation:

    # Load all relevant variables into the class
    def __init__(self, model_name, device, extension):
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extension = extension

        print("--------")
        print("START Gaze_Estimation")
        print("model_weights: " + str(self.model_weights))
        print("model_structure: " + str(self.model_structure))
        print("device: " + str(self.device))
        print("extensions: " + str(self.extension))
        print("--------")

    # Loads the model
    def load_model(self):

        # Initialise the network and save it in the self.model variables
        try:
            log.info("Reading model ...")
            self.network = IENetwork(self.model_structure, self.model_weights)
            # self.model = core.read_network(self.model_structure, self.model_weights) # new openvino version
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
            self.input_name_all_02 = self.network .inputs.keys() # gets all output_names
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
        if "CPU" in self.device:
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
    def predict(self, left_eye, right_eye, head_pose_angles):

        print("--------")
        print("Start predictions")
        #self.width = initial_w
        #self.height = initial_h
        requestid = 0
        # Pre-process the image
        left_eye_preprocess_image, right_eye_preprocess_image = self.preprocess_input(left_eye, right_eye)
        left_eye_preprocess_image = left_eye
        right_eye_preprocess_image = right_eye
        
        head_pose = head_pose_angles
        head_pose = [2,5,10]

        # Starts synchronous inference
        #outputs = self.exec_network.infer({self.input_name: preprocessed_image})
        print("Start syncro inference")
        log.info("Start syncro inference")
        #outputs = self.exec_network.infer({'left_eye_image': left_eye_preprocess_image, 'head_pose_angles': head_pose, 'right_eye_image': right_eye_preprocess_image})
        outputs = self.exec_network.infer({"head_pose_angles": head_pose, "left_eye_image": left_eye_preprocess_image, "right_eye_image": right_eye_preprocess_image })
        #input_dict = {"head_pose_angles": head_angles, "left_eye_image": image_input_preprocessed[0],
                      #"right_eye_image": image_input_preprocessed[1]}
        #outputs = self.exec_network.infer({'head_pose_angles':[2,5,10]})
        # (['left_eye_image', 'head_pose_angles', 'right_eye_image'])
        print("Output of the inference request: " + str(outputs))
        outputs = self.exec_network.requests[requestid].outputs[self.output_name]
        print("Output of the inference request (self.output_name): " + str(outputs))
        head_pose_results = self.gaze_estimation(outputs, frame)
        #head_pose_results = self.preprocess_output(outputs)

        print("End predictions")
        print("--------")
        
        return head_pose_results

    def preprocess_input(self, left_eye1, right_eye1):
        # In this function the original image is resized, transposed and reshaped to fit the model requirements.
        print("--------")
        print("Start: preprocess image")
        log.info("Start: preprocess image")
        print (left_eye1)
        #left_eye = left_eye1.copy()
        #right_eye = right_eye1.copy()
        
        n, c, h, w = (self.core, self.input_shape)[1]
        print (n)
        print (c)
        print (h)
        print (w)
        left_eye_preprocess_image = cv2.resize(left_eye, (w, h))
        left_eye_preprocess_image = left_eye.transpose((2, 0, 1))
        left_eye_preprocess_image = left_eye.reshape((n, c, h, w))
        
        right_eye_preprocess_image = cv2.resize(right_eye, (w, h))
        right_eye_preprocess_image = right_eye.transpose((2, 0, 1))
        right_eye_preprocess_image = right_eye.reshape((n, c, h, w))
        
        #print("Original image size is (W x H): " + str(self.width) + "x" + str(self.height))
        print("Image is now [BxCxHxW]: " + str(left_eye_preprocess_image.shape))
        print("End: preprocess image")
        print("--------")
        
        return left_eye_preprocess_image, right_eye_preprocess_image

    def gaze_estimation(self, outputs,frame):
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
        return gazes

    def preprocess_output(self, image):

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

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--extension', default='/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so')
    parser.add_argument('--video', default=None)
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--threshold', default=0.60)
    parser.add_argument('--inputtype', default='video')
    parser.add_argument('--left_eye_image', default='None')
    parser.add_argument('--right_eye_image', default='None')
    parser.add_argument('--head_pose_angles', default='None')

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
    left_eye = ("left_eye_frame_cropped.png")
    right_eye = ("right_eye_frame_cropped.png")
    #left_eye = ("landmark_image_rectangle.png")
    #right_eye = ("right_eye_computer_pointer.png")
    
    output_path=args.output_path
    #CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
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
    #image = [left_eye, right_eye]
    # open image
    cap = cv2.VideoCapture(left_eye)
    #cap = cv2.imread(left_eye[1])
    res, eyes = cap.read()
    # Gets the coordinates of gaze direction vector
    coordinates = inference.predict(eyes, head_pose_angles)
    cap.release()


if __name__ == '__main__':
    log.basicConfig(filename="logging_gaze.txt", level=log.INFO)
    log.info("Start logging")
    main()