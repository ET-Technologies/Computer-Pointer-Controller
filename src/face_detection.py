# source /opt/intel/openvino/bin/setupvars.sh
# cd /home/thomas/PycharmProjects/Intel/Computer-Pointer-Controller-master/src
# python3 face_detection.py --model /home/thomas/PycharmProjects/models/face-detection-retail-0004 --video demo.mp4
# --model /home/thomas/PycharmProjects/models/face-detection-retail-0004
# --video demo.mp4
'''
Udacity Workspace
Model Downloader python3 downloader.py --name face-detection-retail-0004 --precisions FP32 -o /home/workspace
python3 face_detection.py --model models/face-detection-retail-0004 --device CPU --video demo.mp4 --output_path demo_output.mp4 --inputtype video
'''

'''
Raspberry Pi
python3 face_detection.py --model /home/pi/Udacity/Computer-Pointer-Controller-master/models/face-detection-retail-0004 \
--device MYRIAD \
--extension None \
--video /home/pi/Computer-Pointer-Controller/src/demo.mp4 \
--output_path /home/pi/Computer-Pointer-Controller/src/demo_output.mp4 \
--inputtype cam
'''

'''
Linux
python3 face_detection.py --model /home/thomas/PycharmProjects/Intel/Computer-Pointer-Controller-master/models/face-detection-retail-0004 \
--video /home/thomas/PycharmProjects/Intel/Computer-Pointer-Controller-master/src/demo.mp4 \
--output_path /home/thomas/PycharmProjects/Intel/Computer-Pointer-Controller-master/src/demo_output.mp4 \
--inputtype cam \
--version 2020
'''

#/home/pi/Udacity/Computer-Pointer-Controller-master/models/face-detection-adas-0001.xml


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
    
    # Load all relevant variables into the class
    def __init__(self, model_name, threshold, device, extension, version):
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extension = extension
        self.threshold = threshold
        self.version = version
        
        print("--------")
        print("START Facedetection")
        print("model_weights: " + str(self.model_weights))
        print("model_structure: " + str(self.model_structure))
        print("device: " + str(self.device))
        print("extension: " + str(self.extension))
        print("threshold: " + str(self.threshold))
        print("--------")
    
    # Loads the model
    def load_model(self):

        # Initialise the network and save it in the self.network variables
        try:
            log.info("Reading model ...")
            self.network = IENetwork(model=self.model_structure, weights=self.model_weights)
            #self.network = IECore.read_network(self.model_structure, self.model_weights) #new openvino version
            modelisloaded = True
        except Exception as e:
            modelisloaded = False
            raise ValueError ("Could not initialise the network")
        print("--------")
        print("Model is loaded as self.network : " + str(self.network ))
        
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

            self.model_data = ["\n" + ("input_name: " + str(self.input_name)) + "\n" + ("input_shape: " + str(self.input_shape))+ "\n" + ("output_name: " + str(self.output_name)) + "\n" + ("output_shape: " + str(self.output_shape))]

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
        
        # Add extension
        if "CPU" in self.device and (self.version == 2019):
            log.info("Add extension: ({})".format(str(self.extension)))
            self.core.add_extension(self.extension, self.device)
        
        # Load the network into an executable network
        self.exec_network = self.core.load_network(network=self.network , device_name=self.device, num_requests=1)
        print("Exec_network is loaded as:" + str(self.exec_network))
        print("--------")
        self.check_model()

        return self.model_data

    def check_model(self):
        ### TODO: Check for supported layers ###
        if "CPU" in self.device and (self.version == 2019):
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
    def predict(self, frame):
 
        print("--------")
        print("Start predictions face_detection")
        #self.width = initial_w
        #self.height = initial_h
        requestid = 0
        # Pre-process the image
        preprocessed_image = self.preprocess_input(frame)
        # Starts synchronous inference
        print("Start syncro inference")
        log.info("Start syncro inference")
        outputs = self.exec_network.infer({self.input_name: preprocessed_image})
        print("Output of the inference request: " + str(outputs))
        outputs = self.exec_network.requests[requestid].outputs[self.output_name]
        print("Output of the inference request (self.output_name): " + str(outputs))
        processed_image, frame_cropped = self.boundingbox(outputs, frame)
        cv2.imwrite("cropped_image_02.png", frame_cropped)
        print("End predictions face_detection")
        print("--------")
        return processed_image, frame_cropped

    def preprocess_input(self, frame):
        # In this function the original image is resized, transposed and reshaped to fit the model requirements.
        print("--------")
        print("Start: preprocess image")
        log.info("Start: preprocess image")
        n, c, h, w = (self.core, self.input_shape)[1]
        preprocessed_image = cv2.resize(frame, (w, h))
        preprocessed_image = preprocessed_image.transpose((2, 0, 1))
        preprocessed_image = preprocessed_image.reshape((n, c, h, w))
        print("The input shape from the face detection is n= ({})  c= ({})  h= ({})  w= ({})".format(str(n),str(c), str(h), str(w)))
        print("Original image size is W= ({}) x H= ({})".format(str(self.initial_w),str(self.initial_h)))
        print("Image is now [BxCxHxW]: " + str(preprocessed_image.shape))
        print("End: preprocess image")
        print("--------")

        return preprocessed_image

    def boundingbox(self, outputs, frame):
        #coords = []
        print("--------")
        print("Start: boundingbox")
        print("Bounding box input: " + str(outputs))
        #print("Coords: " + str(coords))
        print("Original image size is (W x H): " + str(self.initial_w) + "x" + str(self.initial_h))
        for obj in outputs[0][0]:
            if obj[2] >= self.threshold:
                obj[3] = int(obj[3] * self.initial_w)
                obj[4] = int(obj[4] * self.initial_h)
                obj[5] = int(obj[5] * self.initial_w)
                obj[6] = int(obj[6] * self.initial_h)
                #coords.append([obj[3], obj[4], obj[5], obj[6]])
                cv2.rectangle(frame, (obj[3], obj[4]), (obj[5], obj[6]), (0, 55, 255), 1)
                print("Bounding box output coordinates of frame: " + str(obj[3]) + " x " + str(obj[4]) + " x " + str(obj[5]) + " x " + str(obj[6]))
                self.xmin = int(obj[3])
                self.ymin = int(obj[4])
                self.xmax = int(obj[5])
                self.ymax = int(obj[6])
        #print("Coordinates for cropped frame are xmin x ymin x xmax x ymax: " + str(self.xmin) + " x " + str(self.ymin) + " x " + str(self.xmax) + " x " + str(self.ymax))
        print("End: boundingbox")
        print("--------")
        frame_cropped = frame.copy()
        #frame_cropped = frame_cropped[self.ymin:(self.ymax + 1), self.xmin:(self.xmax + 1)]
        #cv2.imwrite("cropped image.png", frame_cropped)
        frame_cropped = self.preprocess_output(frame)
        cv2.imwrite("face_full_image.png", frame)
        return frame, frame_cropped

    def preprocess_output(self, frame):
        # crop image to fit the next model
        print("--------")
        print("Start: preprocess_output face")
        print(str(self.xmin))
        #print("Coordinates for cropped frame are xmin x ymin x xmax x ymax: " + str(self.xmin) + " x " + str(self.ymin) + " x " + str(self.xmax) + " x " + str(self.ymax))
        frame_cropped = None
        frame_cropped = frame[self.ymin:(self.ymax + 1), self.xmin:(self.xmax + 1)]
        cv2.imwrite("cropped_image.png", frame_cropped)
        
        print("--------")
        print("End: preprocess_output")
        return frame_cropped
    
    def getinputstream(self, inputtype, video, output_path):
        # gets the inputtype
        try:
            if inputtype == 'video':
                print("Reading video file:", video)
                cap = cv2.VideoCapture(video)
            elif inputtype =='cam':
                print("Reading webcam")
                cap = cv2.VideoCapture(0)
            elif inputtype =='picamera':
                import picamera
                camera = picamera.PiCamera()
                camera.resolution = (320,240)
                camera.framerate = 24
                time.sleep(2)
                image = np.empty((240 * 320 * 3,), dtype=np.uint8)
                camera.capture(image, 'bgr')
                cap = image.reshape((240,320, 3))
            else:
                print("Reading image:", video)
                cap = cv2.imread(video)    
        except FileNotFoundError:
            print("Cannot find video file: " + video)
        except Exception as e:
            print("Something else went wrong with the video file: ", e)
            
        # Capture information about the input video stream
        #self.initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #self.initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #self.video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        #print("--------")
        #print("Input video Data")
        #print("initial_w: " + str(self.initial_w))
        #print("initial_h: " + str(self.initial_h))
        #print("video_len: " + str(self.video_len))
        #print("fps: " + str(self.fps))
        print("--------")
        
        #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #out_video = cv2.VideoWriter(output_path, fourcc, self.fps, (self.initial_w, self.initial_h))

        try:
            while cap == True:
            #while cap.isOpened():
                result, frame = cap.read()
                if not result:
                    break
                #image = inference.predict(frame, initial_w, initial_h)
                image, frame_cropped = self.predict(frame)
                print("The video is writen to the output path")
                #out_video.write(image)
        except Exception as e:
            print("Could not run Inference: ", e)

            
            if not inputtype == "image":
                cap.release()
            cv2.destroyAllWindows()
            
        return
    
    def get_initial_w_h (self, initial_w, initial_h):
        self.initial_w = initial_w
        self.initial_h = initial_h
        print("Initialize initial_w in facedetection: " + str(initial_w))
        print("Initialize initial_h in facedetection: " + str(initial_h))
        
    
def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    #parser.add_argument('--extension', default='/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so')
    parser.add_argument('--extension')
    parser.add_argument('--video', default=None)
    parser.add_argument('--output_path', default='demo_output.mp4')
    #parser.add_argument('--output_path', default='/home/pi/Udacity/Computer-Pointer-Controller-master/src/demo_output.mp4')
    parser.add_argument('--threshold', default=0.20)
    parser.add_argument('--inputtype', default='video')
    parser.add_argument('--version', default='2020')

    return parser
    
def main():
    args = build_argparser().parse_args()
    model_name = args.model
    device = args.device
    extension = args.extension
    video = args.video
    output_path = args.output_path
    #output_path = ("/home/pi/Udacity/Computer-Pointer-Controller-master/src/demo.mp4")
    threshold = args.threshold
    inputtype = args.inputtype
    version = args.version

    # Load class Facedetection
    inference = Facedetection(model_name, threshold, device, extension, version)
    print("Load class Facedetection = OK")
    print("--------")

    # Loads the model
    start_model_load_time = time.time()  # Time to load the model (Start)
    inference.load_model()
    total_model_load_time = time.time() - start_model_load_time  # Time model needed to load
    print("Load Model = OK")
    print("Time to load model: " + str(total_model_load_time))
    print("--------")
    
    # Get the input video stream
    inference.getinputstream(inputtype, video, output_path)

# Start program
if __name__ == '__main__':
    log.basicConfig(filename="logging.txt", level=log.INFO)
    log.info("Start logging")
    main()

