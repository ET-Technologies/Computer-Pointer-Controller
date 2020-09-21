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
    def __init__(self, model_name, device, extension, version, threshold=0):

        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extension = extension
        self.version = version
        self.threshold = threshold

        print("--------")
        print("START Head_Pose_Estimation")
        print("--------")

    
    def load_model(self):
        # Loads the model

        # Initialise the network and save it in the self.model variables
        try:
            self.core = IECore()
            self.network = IENetwork(self.model_structure, self.model_weights)
            # self.network = core.read_network(self.model_structure, self.model_weights) # new openvino version
            self.input_name = next(iter(self.network.inputs))
        except Exception as e:
            log.error("Could not initialise the network")
            raise ValueError("Could not initialise the network")
        print("--------")
        print("Model is loaded as self.model: " + str(self.network))

        # Add extension
        if self.extension and "CPU" in self.device:
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
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported:", not_supported_layers)
                print("You are not lucky, not all layers are supported")
                sys.exit(1)
        log.info("All layers are supported")
        print("All layers are supported")
        print("--------")

    
    def predict(self, frame):
        # Start inference and prediction
        print("--------")
        print("Start predictions head_pose_estimation")

        # Pre-process the image
        preprocessed_image = self.preprocess_input(frame)

        # Starts synchronous inference
        print("Start syncro inference")
        log.info("Start syncro inference")

        outputs = self.exec_network.infer({self.input_name:preprocessed_image})
        print("Output of the inference request: " + str(outputs))

        requestid = 0
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
        log.info("Start: preprocess image head pose")
        n, c, h, w = (self.core, self.input_shape)[1]
        preprocessed_image = cv2.resize(frame, (w, h))
        preprocessed_image = preprocessed_image.transpose((2, 0, 1))
        preprocessed_image = preprocessed_image.reshape((n, c, h, w))
        print("The input shape from the head pose is n= ({})  c= ({})  h= ({})  w= ({})".format(str(n),str(c), str(h), str(w)))
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
        
def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--extension', default='/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so')
    parser.add_argument('--video', default=None)
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--threshold', default=0.60)
    parser.add_argument('--inputtype', default='video')
    parser.add_argument('--version', default='2020')

    return parser

def main():
    args = build_argparser().parse_args()
    model_name = args.model
    device = args.device
    extension = args.extension
    video = args.video
    video = ("cropped_image.png")
    output_path=args.output_path
    threshold = args.threshold
    inputtype = args.inputtype

    # Load class Head_Pose_Estimation
    inference = Head_Pose_Estimation(model_name, device, extension)
    print("Load Model = OK")
    print("--------")

    # Loads the model
    # Time to load the model (Start)
    start_model_load_time = time.time()  
    inference.load_model()
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

# Start program
if __name__ == '__main__':
    log.basicConfig(filename="logging_head_pose.txt", level=log.INFO)
    log.info("Start logging")
    main()
