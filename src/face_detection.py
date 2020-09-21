'''
Udacity Workspace
Video:
python3 face_detection_v1.py \
--model models/face-detection-retail-0004 \
--extension /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so \
--video demo.mp4 \
--output_path demo_output.mp4 \
--inputtype video

Image:
python3 face_detection_v1.py \
--model models/face-detection-retail-0004 \
--extension /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so \
--video face.png \
--output_path demo_output.mp4
'''

'''
Linux:
source /opt/intel/openvino/bin/setupvars.sh

Webcam:

python3 face_detection.py --model /home/thomas/PycharmProjects/Intel/Computer-Pointer-Controller-master/models/face-detection-retail-0004 \
--video /home/thomas/PycharmProjects/Intel/Computer-Pointer-Controller-master/src/demo.mp4 \
--output_path /home/thomas/PycharmProjects/Intel/Computer-Pointer-Controller-master/src/demo_output.mp4 \
--inputtype cam \
--version 2020

Video:
python3 face_detection_v1.py --model /home/thomas/PycharmProjects/Intel/Computer-Pointer-Controller-master/models/face-detection-retail-0004 \
--video demo.mp4 \
--output_path /home/thomas/PycharmProjects/Intel/Computer-Pointer-Controller-master/src/demo_output.mp4 \
--inputtype video \
--version 2019

Image:
python3 face_detection_v1.py --model /home/thomas/PycharmProjects/Intel/Computer-Pointer-Controller-master/models/face-detection-retail-0004 \
--video face.jpg \
--output_path /home/thomas/PycharmProjects/Intel/Computer-Pointer-Controller-master/src/face.jpg \
--version 2019
'''
'''
Raspberry
python3 face_detection.py  \
--model /home/pi/Udacity/Computer-Pointer-Controller-master/models/face-detection-adas-0001  \
--device MYRIAD \
--extension None \
--video /home/pi/Udacity/Computer-Pointer-Controller-master/src/demo.mp4 \
--output_path ./output/demo_output.mp4 \
--inputtype video \
--version 2019

python3 face_detection.py  \
--model /home/pi/Udacity/Computer-Pointer-Controller-master/models/face-detection-adas-0001  \
--device MYRIAD \
--extension None \
--video /home/pi/KeyBox/face_test.jpeg \
--output_path ./output/demo_output.mp4 \
--inputtype image \
--version 2019
'''

import numpy as np
import time
import os
import cv2
import argparse
import sys
from os import path
from openvino.inference_engine import IENetwork, IECore
from input_feeder import InputFeeder
import logging as log

class Facedetection:

    def __init__(self, model_name, threshold, device, extension, version):
        # Load all relevant variables into the class
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extension = extension
        self.threshold = threshold
        self.version = version

        print("--------")
        print("START Facedetection")
        print("--------")

    def load_model(self):

        # Initialise the network and save it in the self.network variables
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
        if "CPU" in self.device and (self.version == 2019):
            log.info("Add extension: ({})".format(str(self.extension)))
            self.core.add_extension(self.extension, self.device)

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
            print("--------")
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported:", not_supported_layers)
                print("You are not lucky, not all layers are supported")
                sys.exit(1)
        log.info("All layers are supported")
        print("All layers are supported")

    def predict(self, frame):
        # Starts predictions face_detection
        print("--------")
        print("Starts predictions for face_detection")

        # Pre-process the image
        preprocessed_image = self.preprocess_input(frame)

        # Starts synchronous inference
        print("Start syncro inference")
        log.info("Start syncro inference")

        outputs = self.exec_network.infer({self.input_name: preprocessed_image})
        print("Output of the inference request: " + str(outputs))

        requestid = 0
        outputs = self.exec_network.requests[requestid].outputs[self.output_name]
        print("Output of the inference request (self.output_name): " + str(outputs))
        processed_image, frame_cropped, coords = self.preprocess_output(outputs, frame)
        cv2.imwrite("cropped_image_02.png", frame_cropped)
        print("End predictions face_detection")
        print("--------")

        return processed_image, frame_cropped, coords

    def preprocess_input(self, frame):
        # In this function the original image is resized, transposed and reshaped to fit the model requirements.
        print("--------")
        print("Start: preprocess image")
        log.info("Start: preprocess image")
        n, c, h, w = (self.core, self.input_shape)[1]
        print (w,h)
        preprocessed_image = cv2.resize(frame, (w, h))
        preprocessed_image = preprocessed_image.transpose((2, 0, 1))
        preprocessed_image = preprocessed_image.reshape((n, c, h, w))
        print("The input shape from the face detection is n= ({})  c= ({})  h= ({})  w= ({})".format(str(n),str(c), str(h), str(w)))
        print("Image is now [BxCxHxW]: " + str(preprocessed_image.shape))
        print("End: preprocess image")
        print("--------")

        return preprocessed_image

    def preprocess_output(self, outputs, frame):
        
        coords = []
        print("--------")
        print("Start: preprocess_output")
        print("Bounding box input: " + str(outputs))
        self.initial_w = frame.shape[1]
        self.initial_h = frame.shape[0]
        print("Original image size is (W x H): " + str(self.initial_w) + "x" + str(self.initial_h))
        for obj in outputs[0][0]:
            confidence = obj[2]
            if confidence >= self.threshold:
                obj[3] = int(obj[3] * self.initial_w)
                obj[4] = int(obj[4] * self.initial_h)
                obj[5] = int(obj[5] * self.initial_w)
                obj[6] = int(obj[6] * self.initial_h)
                coords.append([obj[3], obj[4], obj[5], obj[6]])
                cv2.rectangle(frame, (obj[3], obj[4]), (obj[5], obj[6]), (0, 55, 255), 1)
                print("Bounding box output coordinates of frame: " + str(obj[3]) + " x " + str(obj[4]) + " x " + str(obj[5]) + " x " + str(obj[6]))
                self.xmin = int(obj[3])
                self.ymin = int(obj[4])
                self.xmax = int(obj[5])
                self.ymax = int(obj[6])
                

        print("End: boundingbox")
        print("--------")
        frame_cropped = frame.copy()
        frame_cropped = frame_cropped[self.ymin:(self.ymax + 1), self.xmin:(self.xmax + 1)]
        cv2.imwrite("Face_cropped image.png", frame_cropped)
        cv2.imwrite("Face_image.png", frame)
        
        return frame, frame_cropped, coords

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
            
        return cap
    
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
    output_path = args.output_path
    threshold = args.threshold
    inputtype = args.inputtype
    version = args.version

    # Load class Facedetection
    inference = Facedetection(model_name, threshold, device, extension, version)
    print("Load class Facedetection = OK")
    print("--------")

    # Loads the model
    # Time to load the model (Start)
    start_model_load_time = time.time()  
    model_data, modellayers = inference.load_model()
    #print("Model data: ", model_data)
    #print("Model layers: ", modellayers)
    # Time model needed to load
    total_model_load_time = time.time() - start_model_load_time  
    print("Load Model = OK")
    print("Time to load model: " + str(total_model_load_time))
    print("--------")
    
    # Load data (video, cam or image)
    cap = inference.load_data(inputtype, video)
    print ("cap:",cap)

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
        path = '/home/pi/KeyBox/Face_cropped image.png'
        image = cv2.imread(path)
        cv2.imshow("test", image)
        cv2.waitKey(0) 
    
    cv2.destroyAllWindows()  

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--extension')
    parser.add_argument('--video', default=None)
    parser.add_argument('--output_path')
    parser.add_argument('--threshold', default=0.60)
    parser.add_argument('--inputtype')
    parser.add_argument('--version', default='2020')

    return parser


# Start program
if __name__ == '__main__':
    log.basicConfig(filename="logging.txt", level=log.INFO)
    log.info("Start logging")
    main()
