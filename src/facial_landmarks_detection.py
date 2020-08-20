# source /opt/intel/openvino/bin/setupvars.sh
# cd /home/thomas/PycharmProjects/Intel/Computer-Pointer-Controller-master/src
# python3 facial_landmarks_detection.py --model /home/thomas/PycharmProjects/models/landmarks-regression-retail-0009 --video demo.mp4
# images/sitting-on-car.jpg
# images/car.png
# demo.mp4

# Udacity Workspace
# cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader
# Model Downloader python3 downloader.py --name landmarks-regression-retail-0009 --precisions FP32 -o /home/workspace
# python3 facial_landmarks_detection.py --model models/landmarks-regression-retail-0009 --video demo.mp4

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

class Model_X:
    '''
    Class with all relevant tools to do object detection
    '''

    # Load all relevant variables into the class
    def __init__(self, model_name, device, threshold, extension):
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        #self.extensions = extensions
        self.device = device
        self.threshold = threshold
        print("--------")
        print("START")
        print("model_weights: " + str(self.model_weights))
        print("model_structure: " + str(self.model_structure))
        print("device: " + str(self.device))
        #print("extensions: " + str(self.extensions))
        print("--------")

    # Loads the model
    def load_model(self, device, extension):

        # Initialise the network and save it in the self.model variables
        try:
            log.info("Reading model ...")
            self.model = IENetwork(self.model_structure, self.model_weights)
            # self.model = core.read_network(self.model_structure, self.model_weights) # new openvino version
        except Exception as e:
            raise ValueError("Could not initialise the network")
        print("--------")
        print("Model is loaded as self.model: " + str(self.model))

        # Get the input layer
        self.input_name = next(iter(self.model.inputs))
        # Gets all input_names
        self.input_name_all = [i for i in self.model.inputs.keys()]
        self.input_name_all_02 = self.model.inputs.keys()  # gets all output_names
        self.input_name_first_entry = self.input_name_all[0]

        self.input_shape = self.model.inputs[self.input_name].shape

        self.output_name = next(iter(self.model.outputs))
        self.output_name_type = self.model.outputs[self.output_name]
        self.output_names = [i for i in self.model.outputs.keys()]  # gets all output_names
        self.output_names_total_entries = len(self.output_names)

        self.output_shape = self.model.outputs[self.output_name].shape
        self.output_shape_second_entry = self.model.outputs[self.output_name].shape[1]

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
        CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
        if "CPU" in device:
            log.info("Add extension: ({})".format(str(CPU_EXTENSION)))
            self.core.add_extension(CPU_EXTENSION, device)

        # Load the network into an executable network
        self.exec_network = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        print("Exec_network is loaded as:" + str(self.exec_network))
        print("--------")

    # Start inference and prediction
    def predict(self, frame, initial_w, initial_h):

        print("--------")
        print("Start predictions")
        self.width = initial_w
        self.height = initial_h
        requestid = 0

        # save original image
        #input_img = image

        # Pre-process the image
        preprocessed_image = self.preprocess_input(frame)
        # Starts synchronous inference
        print("Start syncro inference")
        outputs = self.exec_network.infer({self.input_name:preprocessed_image})
        print("Output of the inference request: " + str(outputs))
        outputs = self.exec_network.requests[requestid].outputs[self.output_name]
        print("Output of the inference request (self.output_name): " + str(outputs))
        landmark_results = self.landmark_detection(outputs, frame)

        print("End predictions")
        print("--------")

        return landmark_results

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
        return image

    # Get the inference output
    def get_output(self, infer_request_handle, request_id, output):
        if output:
            res = infer_request_handle.output[output]
        else:
            res = self.exec_network.requests[request_id].outputs[self.output_name]
        return res

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
        self.left_eye_coordinates_x = int(coords[0]*self.width)
        self.left_eye_coordinates_y = int(coords[1]*self.height)
        self.right_eye_coordinates_x = int(coords[2]*self.width)
        self.right_eye_coordinates_y = int(coords[3]*self.height)
        self.nose_coordinates_x = int(coords[4] * self.width)
        self.nose_coordinates_y = int(coords[5] * self.height)
        self.left_mouth_coordinates_x = int(coords[6] * self.width)
        self.left_mouth_coordinates_y = int(coords[7] * self.height)
        self.right_mouth_coordinates_x = int(coords[8] * self.width)
        self.right_mouth_coordinates_y = int(coords[9] * self.height)
        print("left_eye_coordinates_x: " + str(self.left_eye_coordinates_x))
        print("left_eye_coordinates_y: " + str(self.left_eye_coordinates_y))
        
        self.left_eye_x_min = self.left_eye_coordinates_x-30
        self.left_eye_x_max = self.left_eye_coordinates_x+30
        self.left_eye_y_min = self.left_eye_coordinates_y-30
        self.left_eye_y_max = self.left_eye_coordinates_y+30
        
        self.right_eye_x_min = self.right_eye_coordinates_x-30
        self.right_eye_x_max = self.right_eye_coordinates_x+30
        self.right_eye_y_min = self.right_eye_coordinates_y-30
        self.right_eye_y_max = self.right_eye_coordinates_y+30
        
        print("Rectangle coordinates: ({}) + ({}) + ({}) + ({})".format(str(self.left_eye_x_min),str(self.left_eye_x_max), str(self.left_eye_y_min), str(self.left_eye_y_max)))
        #log.info("Add extension: ({})".format(str(CPU_EXTENSION)))
        self.draw_landmarks(frame)
        return

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
        
        eyes_cropped_image = self.preprocess_output(self.frame_original)
        
        print("End: draw_landmarks")
        print("--------")
        return
    
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
        cv2.imwrite("left_eye_frame_cropped.png", left_eye_frame_cropped)
        cv2.imwrite("right_eye_frame_cropped.png", right_eye_frame_cropped)
        print("--------")
        print("End: preprocess_output")
        return
    
# Collect all the necessary input values
def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--extension', default=None)
    parser.add_argument('--video', default=None)
    parser.add_argument('--output_path', default='results/')
    parser.add_argument('--threshold', default=0.60)

    return parser


def main():
    args = build_argparser().parse_args()
    model_name = args.model
    device = args.device
    extension = args.extension
    video = args.video
    video = ("cropped_image.png")
    threshold = args.threshold
    output_path = args.output_path
    #CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
    start_model_load_time = time.time()  # Time to load the model (Start)

    # Load class Model_X
    inference = Model_X(model_name, device,threshold, extension)
    print("Load Model = OK")
    print("--------")

    # Loads the model
    inference.load_model(device,extension)
    total_model_load_time = time.time() - start_model_load_time  # Time model needed to load
    print("Load Model = OK")
    print("Time to load model: " + str(total_model_load_time))
    print("--------")

    # Get the input frame
    cap = cv2.VideoCapture(video)
    try:
        print("Reading video file name:", video)
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

    # Define output video

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter('output_video3.mp4', fourcc, fps, (initial_w, initial_h))

    
#    feed=InputFeeder(input_type='video', input_file=video)
#    feed.load_data()
#    initial_w = int(feed.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#    initial_h = int(feed.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#    print ("initial_w: from feed " +str(initial_w))
#    print ("initial_h: " +str(initial_h))

    try:
        while cap.isOpened():
            result, frame = cap.read()
            if not result:
                break
            image = inference.predict(frame, initial_w, initial_h)
            print("The video is writen to the output path: landmark_image.png")
            #out_video.write(image)
    except Exception as e:
        print("Could not run Inference: ", e)

        cap.release()
        cv2.destroyAllWindows()
    
  #  for batch in feed.next_batch():
   #     if batch is None:
    #        break
     #   image = batch.copy()
      #  image = inference.predict(image, initial_w, initial_h)
        
        
    
    #frame = input_feeder_helper.next_batch()
    #res, image = inference.predict(frame, initial_w, initial_h)
    #frame = input_feeder_helper.next_batch(image)
    #image = inference.predict(frame, initial_w, initial_h)
    #result, frame = input_feeder_helper.load_data()
    #image = inference.predict(frame, initial_w, initial_h)
    #while cap.isOpened():
     #   frame = input_feeder_helper.load_data()
        #result, frame = cap.read()
      #  image = inference.predict(frame, initial_w, initial_h)
     #       frame = input_feeder_helper.load_data()
            
            
            
    cap.release()
    cv2.destroyAllWindows()
    
    # Read the input image
    #image = cv2.imread(video_file)
    # Scale the output text by the image shape
    #scaler = max(int(image.shape[0] / 1000), 1)
    # Write the text of color and type onto the image
    #image = cv2.putText(image, "Color: {}, Type: {}".format(color, car_type), (50 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 2 * scaler, (255, 255, 255), 3 * scaler)

# Start sequence
if __name__ == '__main__':
    log.basicConfig(filename="logging_landmarks.txt", level=log.INFO)
    log.info("Start logging")
    main()
