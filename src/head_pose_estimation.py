# source /opt/intel/openvino/bin/setupvars.sh
# cd /home/thomas/PycharmProjects/Intel/Computer-Pointer-Controller-master/src
# python3 head_pose_estimation.py --model /home/thomas/PycharmProjects/models/head-pose-estimation-adas-0001 --video demo.mp4

#intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml
'''
Udacity Workspace
cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader
Model Downloader python3 downloader.py --name head-pose-estimation-adas-0001 --precisions FP32 -o /home/workspace
python3 head_pose_estimation.py --model models/head-pose-estimation-adas-0001 --video demo.mp4
'''
import numpy as np
import time
import os
import cv2
import argparse
import sys
from openvino.inference_engine import IENetwork, IECore
import logging as log

class Head_Pose_Estimation:

    # Load all relevant variables into the class
    def __init__(self, model_name, device, extension):

        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extension = extension

        print("--------")
        print("START Head_Pose_Estimation")
        print("model_weights: " + str(self.model_weights))
        print("model_structure: " + str(self.model_structure))
        print("device: " + str(self.device))
        print("extension: " + str(self.extension))
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
    def predict(self, frame):

        print("--------")
        print("Start predictions head_pose_estimation")
        #self.width = initial_w
        #self.height = initial_h
        requestid = 0
        # Pre-process the image
        preprocessed_image = self.preprocess_input(frame)
        # Starts synchronous inference
        print("Start syncro inference")
        log.info("Start syncro inference")
        outputs = self.exec_network.infer({self.input_name:preprocessed_image})
        print("Output of the inference request: " + str(outputs))
        outputs = self.exec_network.requests[requestid].outputs[self.output_name]
        print("Output of the inference request (self.output_name): " + str(outputs))
        head_pose_results = self.head_pose_detection(outputs, frame)
        #head_pose_results = self.preprocess_output(outputs)
        print("End predictions")
        print("--------")
        return head_pose_results

    def preprocess_input(self, frame):
        # In this function the original image is resized, transposed and reshaped to fit the model requirements.
        print("--------")
        print("Start: preprocess image")
        log.info("Start: preprocess image")
        n, c, h, w = (self.core, self.input_shape)[1]
        preprocessed_image = cv2.resize(frame, (w, h))
        preprocessed_image = preprocessed_image.transpose((2, 0, 1))
        preprocessed_image = preprocessed_image.reshape((n, c, h, w))
        print("Original image size is W= ({}) x H= ({})".format(str(self.initial_w),str(self.initial_h)))
        print("Image is now [BxCxHxW]: " + str(preprocessed_image.shape))
        print("End: preprocess image")
        print("--------")
        
        return preprocessed_image

    def head_pose_detection(self, outputs,frame):
        print("--------")
        print("Start: head_pose_estimation")
        result_len = len(outputs)
        print("total number of entries: " + str(result_len))
        angles =[]
        angle_p_fc = self.exec_network.requests[0].outputs['angle_p_fc']
        angle_r_fc = self.exec_network.requests[0].outputs['angle_r_fc']
        angle_y_fc = self.exec_network.requests[0].outputs['angle_y_fc']
        print("Output of the inference request (self.output_name): " + str(angle_p_fc))
        angle_p_fc = int(angle_p_fc)
        angle_r_fc = int(angle_r_fc)
        angle_y_fc = int(angle_y_fc)
        print("angle_p_fc pitch in degrees: " + str(angle_p_fc))
        print("angle_r_fc roll in degrees: " + str(angle_r_fc))
        print("angle_y_fc yaw in degrees: " + str(angle_y_fc))
        angles.append([angle_p_fc, angle_r_fc, angle_y_fc])

        print("angles: " + str(angles))
        print("End: head_pose_detection")
        print("--------")
        return angles

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

    def getinputstream(self, inputtype, video, output_path):
        # gets the inputtype
        try:
            if inputtype == 'video':
                print("Reading video file:", video)
                cap = cv2.VideoCapture(video)
            elif inputtype =='cam':
                print("Reading webcam")
                cap = cv2.VideoCapture(0)
            else:
                print("Reading image:", video)
                cap = cv2.imread(video)    
        except FileNotFoundError:
            print("Cannot find video file: " + video)
        except Exception as e:
            print("Something else went wrong with the video file: ", e)
            
        # Capture information about the input video stream
        self.initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        print("--------")
        print("Input video Data")
        print("initial_w: " + str(self.initial_w))
        print("initial_h: " + str(self.initial_h))
        print("video_len: " + str(self.video_len))
        print("fps: " + str(self.fps))
        print("--------")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(output_path, fourcc, self.fps, (self.initial_w, self.initial_h))

        try:
            while cap.isOpened():
                result, frame = cap.read()
                if not result:
                    break
                #image = inference.predict(frame, initial_w, initial_h)
                image = self.predict(frame)
                print("The video is writen to the output path")
                out_video.write(image)
        except Exception as e:
            print("Could not run Inference: ", e)

            cap.release()
            cv2.destroyAllWindows()
            
        return
    
    #def get_initial_w_h (self, initial_w, initial_h):
     #   self.initial_w = initial_w
      #  self.initial_h = initial_h
       # print("Initialize initial_w in headposeestimation: " + str(initial_w))
        #print("Initialize initial_h in headposeestimation: " + str(initial_h))

    def get_initial_w_h (self, frame_cropped):
        self.initial_w = frame_cropped.shape[1]
        self.initial_h = frame_cropped.shape[0]
        print("Initialize initial_w in headposeestimation: " + str(self.initial_w))
        print("Initialize initial_h in headposeestimation: " + str(self.initial_h))
        
def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--extension', default='/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so')
    parser.add_argument('--video', default=None)
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
    video = ("cropped_image.png")
    output_path=args.output_path
    #CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
    threshold = args.threshold
    inputtype = args.inputtype

    # Load class Head_Pose_Estimation
    inference = Head_Pose_Estimation(model_name, device, extension)
    print("Load Model = OK")
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
    log.basicConfig(filename="logging_head_pose.txt", level=log.INFO)
    log.info("Start logging")
    main()