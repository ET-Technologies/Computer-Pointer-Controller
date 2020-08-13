'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
# source /opt/intel/openvino/bin/setupvars.sh
# cd /home/thomas/PycharmProjects/Intel/Computer-Pointer-Controller-master/src
# python3 face_detection.py --model /home/thomas/PycharmProjects/models/face-detection-retail-0004 --video demo.mp4
# --model /home/thomas/PycharmProjects/models/face-detection-retail-0004
# --video demo.mp4

import numpy as np
import time
import os
import cv2
import argparse
import sys
from os import path
from openvino.inference_engine import IENetwork, IECore
import input_feeder

class Model_X:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, threshold, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extensions = extensions
        self.threshold = threshold
        print("model_weights: " + str(self.model_weights))
        print("model_structure: " + str(self.model_structure))
        print("device: " + str(self.device))
        print("extensions: " + str(self.extensions))



    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''

        try:
            self.model = IENetwork(model=self.model_structure, weights=self.model_weights)
            #self.model = IECore.read_network(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError ("Could not dot it")

        print("--------")
        print("Model is loaded as self.model: " + str(self.model))
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape
        self.exec_network = None

        print("input_name: " + str(self.input_name))
        print("input_shape: " + str(self.input_shape))
        print("output_name: " + str(self.output_name))
        print("output_shape: " + str(self.output_shape))
        print("--------")
        self.core = IECore()
        #self.core.add_extension(self.extensions, self.device)
        self.exec_network = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        print("Exec_network is loaded as:" + str(self.exec_network))
        print("--------")

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
        outputs = self.exec_network.infer({self.input_name: preprocessed_image})
        print("Output of the inference request: " + str(outputs))
        outputs = self.exec_network.requests[requestid].outputs[self.output_name]
        print("Output of the inference request (self.output_name): " + str(outputs))
        processed_image = self.boundingbox(outputs, frame)
        print("End predictions")
        print("--------")
        return processed_image

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, frame):
        # In this function the original image is resized, transposed and reshaped to fit the model requirements.
        print("--------")
        print("Start: preprocess image")
        n, c, h, w = (self.core, self.input_shape)[1]
        image = cv2.resize(frame, (w, h))
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        print("Original image size is (W x H): " + str(self.width) + "x" + str(self.height))
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
        print("End: boundingbox")
        print("--------")

        return frame


    def preprocess_output(self, outputs):
        return


    '''
    Before feeding the output of this model to the next model,
    you might have to preprocess the output. This function is where you can do that.
    '''




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
    parser.add_argument('--extension', default=None)
    parser.add_argument('--video', default=None)
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--threshold', default=0.60)

    return parser


def main():
    args = build_argparser().parse_args()
    model_name = args.model
    device = args.device
    extension = args.extension
    video = args.video
    output_path = args.output_path
    threshold = args.threshold


    # Load class Model_X
    inference = Model_X(model_name, threshold, device, extension)
    print("Load class Model_X = OK")
    print("--------")


    # Handles videofile LATER
    #input_type = inference.videofile(video)
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

    #out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video3.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps,
    #                            (initial_w, initial_h), True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #out = cv2.VideoWriter(filename, fourcc, 20, (width, height))
    out_video = cv2.VideoWriter('output_video3.mp4', fourcc, fps, (initial_w, initial_h))
    # Define output video
    #out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video3.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)

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
    main()
