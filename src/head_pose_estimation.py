'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
# python facial_landmarks_detection.py --model intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001 --video demo.mp4 --output_path outputs
# python facial_landmarks_detection.py --model intel/text-recognition-0012/FP16/text-recognition-0012 --video demo.mp4 --output_path outputs
# python facial_landmarks_detection.py --model intel/face-detection-adas-binary-0001/INT1/face-detection-adas-binary-0001 --video demo.mp4 --output_path outputs
# python facial_landmarks_detection.py --model intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009 --video demo.mp4 --output_path outputs
#intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml
#intel/text-recognition-0012/FP16/text-recognition-0012.xml
import numpy as np
import time
import os
import cv2
import argparse
import sys
from openvino.inference_engine import IENetwork, IECore
#import input_feeder

class Model_X:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extensions = extensions


    def load_model(self):

        #self.model_weights=model_name+'.bin'
        #self.model_structure=model_name+'.xml'
        model = IENetwork(model=self.model_structure, weights=self.model_weights)

        self.core = IECore()
        self.core.add_extension(self.extensions, self.device)
        self.exec_network = self.core.load_network(network=model, device_name=self.device, num_requests=1)
        
        self.input_name = next(iter(model.inputs))
        self.input_shape = model.inputs[self.input_name].shape
        self.output_name = next(iter(model.outputs)) # gets just the first output_name
        self.output_shape = model.outputs[self.output_name].shape
        self.output_names = [i for i in model.outputs.keys()] # gets all output_names
        print ("input_name:" +str(self.input_name))
        print ("input_shape:" +str(self.input_shape))
        print ("output_name: " +str(self.output_name)) 
        print ("output_shape: " +str(self.output_shape))
        print ("output_names: " +str(self.output_names)) 
        
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''

    def predict(self, image):
        
        image = self.preprocess_input(image)
        outputs = self.exec_network.infer({self.input_name:image}) #syncro inference
        outputs = self.preprocess_output(outputs)
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        return outputs

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        # Get the input shape
        n, c, h, w = (self.core, self.input_shape)[1]
        image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        
        return image

    def preprocess_output(self, image):
        # Her we get the outputs from the facial landmarks model.
        # The information comes from https://docs.openvinotoolkit.org/latest/omz_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html
        
        # Landmark model
        #print ("test")
        #outs = image[self.output_names][0]
        #print ("test")
        #leye_x = outs[0].tolist()[0][0]
        #print ("leye_x: " +str(leye_x))
        
        # Head Pose Model
        
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
    '''
    Before feeding the output of this model to the next model,
    you might have to preprocess the output. This function is where you can do that.
    '''


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--extension', default=None)
    parser.add_argument('--video', default=None)
    parser.add_argument('--output_path', default=None)

    return parser

def main():
    args = build_argparser().parse_args()
    model_name = args.model
    device = args.device
    extension = args.extension
    video = args.video
    output_path=args.output_path
    CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
    extension = CPU_EXTENSION

    # Load class Model_X
    inference = Model_X(model_name, device, extension)

    # Handles videofile LATER
    #input_type = inference.videofile(video)
    # Loads the model
    inference.load_model()
    # Gets output to load in the next model

    # Get the input video stream
    cap = cv2.VideoCapture(video)
    # Capture information about the input video stream
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define output video
    #out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video3.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)
    #out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video3.mp4'))

    try:
        while cap.isOpened():
            result, frame = cap.read()
            if not result:
                break

            image = inference.predict(frame)
            #out_video.write(image)
    except Exception as e:
        print("Could not run Inference: ", e)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()