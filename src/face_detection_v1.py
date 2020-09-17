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
import imutils


class Facedetection:

    # Load all relevant variables into the class
    def __init__(self, model_name, threshold, device, extension, version):
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extension = extension
        self.threshold = threshold
        self.version = version

        # noinspection PyStringFormat
        print("model_weights: {}\n".format(self.model_weights))
        # model_info = [self.model_weights, self.model_structure]
       # print("model_weights: {}\nmodel_structure: {}\ndevice: {}\nextension: {}\nthreshold: {}\n".format.(
        #    self.model_weights), (self.model_structure), (self.device), (self.extension, (self.threshold)))
        # log.info ("model_weights: {}\nmodel_structure: {}\ndevice: {}\nextension: {}\nthreshold: {}\n".format.str(self.model_weights), str(self.model_structure), str(self.device), str(self.extension, str(self.threshold)))
        # model_info = ("model_weights: {}\nmodel_structure: {}\ndevice: {}\nextension: {}\nthreshold: {}\n".format.str(self.model_weights), str(self.model_structure), str(self.device), str(self.extension, str(self.threshold)))
        print("--------")
        print("START Facedetection")
        print("model_weights: " + str(self.model_weights))
        print("model_structure: " + str(self.model_structure))
        print("device: " + str(self.device))
        print("extension: " + str(self.extension))
        print("threshold: " + str(self.threshold))
        print("--------")

    def load_model(self):

        # Initialise the network and save it in the self.network variables
        try:
            log.info("Reading model ...")
            if self.version == "2020":
                # new openvino version
                self.network = IECore.read_network(self.model_structure, self.model_weights)
            self.network = IENetwork(model=self.model_structure, weights=self.model_weights)
            modelisloaded = True

        except Exception as e:
            modelisloaded = False
            raise ValueError("Could not initialise the network")
        print("--------")
        print("Model is loaded as self.network : " + str(self.network))

        if modelisloaded == True:
            log.info("Model is loaded...")
            # Get the input layer
            self.input_name = next(iter(self.network.inputs))

        self.core = IECore()

        # Add extension
        if "CPU" in self.device and (self.version == 2019):
            log.info("Add extension: ({})".format(str(self.extension)))
            self.core.add_extension(self.extension, self.device)

        # Load the network into an executable network
        self.exec_network = self.core.load_network(network=self.network, device_name=self.device, num_requests=1)
        log.info("Exec_network is loaded as:" + str(self.exec_network))
        print("Exec_network is loaded as:" + str(self.exec_network))
        print("--------")
        self.check_model()
        model_data = [self.model_weights, self.model_structure]

        return model_data

    def check_model(self):

        # Check for supported layers 
        if "CPU" in self.device and (self.version == 2019):
            supported_layers = self.core.query_network(self.network, "CPU")
            print("--------")
            print("Check for supported layers")
            print("supported_layers: " + str(supported_layers))
            not_supported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
            print("not_supported_layers: " + str(not_supported_layers))
            print("You are lucky, all layers are supported")
            print("--------")
            if len(not_supported_layers) != 0:
                log.info("Check for supported layers")
                sys.exit(1)

    def getinputstream(self, video):
        cap = cv2.VideoCapture(video)

        while True:
            for _ in range(10):
                _, frame = cap.read()
                print("load frame")
                print(video)
                cv2.imwrite('test28.png', frame)
            return frame

    def example_videocapture_01():
        cap = cv2.VideoCapture(video)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            print("load frame")
            print(video)
            cv2.imwrite('test28.png', frame)

            return frame


    def example_videocapture_04(self, video):
        cap = cv2.VideoCapture(video)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            print("load frame")
            print(video)
            cv2.imwrite('test.png', frame)

            return frame

    def example_videocapture_01(self, video):
        cap = cv2.VideoCapture(video)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            print("load ha frame")
            print(video)
            cv2.imwrite('test28.png', frame)

            return frame

    def example_videostream(self, video):
        from imutils.video import VideoStream
        from imutils.video import FPS
        #cap = VideoStream(usePiCamera=True).start()
        cap = VideoStream(0).start()
        cap = VideoStream(video).start()
        # camera warm up
        time.sleep(2.0)
        while True:
            frame = cap.read()
            #frame = imutils.resize(frame, width=450)
            #(h, w) = frame.shape[:2]
            #print(h, w)
            #cv2.imshow("Test", frame)
            cv2.waitKey(2)

        cv2.destroyAllWindows()
        cap.stop()
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
    start_model_load_time = time.time()  # Time to load the model (Start)
    model_info = inference.load_model()
    print(model_info)
    total_model_load_time = time.time() - start_model_load_time  # Time model needed to load
    print("Load Model = OK")
    print("Time to load model: " + str(total_model_load_time))
    print("--------")

    inference.example_videocapture_01(video)

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--extension')
    parser.add_argument('--video', default=None)
    parser.add_argument('--output_path')
    parser.add_argument('--threshold', default=0.20)
    parser.add_argument('--inputtype')
    parser.add_argument('--version', default='2020')

    return parser


# Start program
if __name__ == '__main__':
    log.basicConfig(filename="logging.txt", level=log.INFO)
    log.info("Start logging")
    main()
