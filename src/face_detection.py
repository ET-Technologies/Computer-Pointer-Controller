'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import numpy as np
import time
import os
import cv2
import argparse
import sys
from openvino.inference_engine import IENetwork, IECore
import input_feeder

class Model_X:
    '''
    Class for the Face Detection Model.
    '''
    Test
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extensions = extensions

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        model = IENetwork(model=model_structure, weights=model_weights)

        core = IECore()
        core.add_extension(self.extensions, self.device)
        exec_network = core.load_network(network=model, device_name=device, num_requests=1)

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        raise NotImplementedError

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        # Get the input shape
        n, c, h, w = (self.core, self.input_shape)[1]
        image = cv2.resize(frame, (w, h))
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))

        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        return image

    def preprocess_output(self, outputs):

    '''
    Before feeding the output of this model to the next model,
    you might have to preprocess the output. This function is where you can do that.
    '''
        return outputs

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

    return parser


def main():
    args = build_argparser().parse_args()
    model_name = args.model
    device = args.device
    extension = args.extension
    video = args.video

    # Load class Model_X
    inference = Model_X(model_name, device, extensions)

    # Handles videofile LATER
    #input_type = inference.videofile(video)
    # Loads the model
    inference.load_model()
    # Gets output to load in the next model
    output = inference.predict()

    # Get the input video stream
    cap = cv2.VideoCapture(video)
    # Capture information about the input video stream
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define output video
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video3.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)

    try:
        while cap.isOpened():
            retsult, frame = cap.read()
            if not result:
                break

            image = inference.predict(frame, initial_w, initial_h)
            out_video.write(image)
    except Exception as e:
        print("Could not run Inference: ", e)

        cap.release()
        cv2.destroyAllWindows()

        


if __name__ == '__main__':
    main()