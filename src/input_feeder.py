'''
This class can be used to feed input from an image, webcam, or video to your model.
Sample usage:
    feed=InputFeeder(input_type='video', input_file='video.mp4')
    feed.load_data()
    for batch in feed.next_batch():
        do_something(batch)
    feed.close()
'''
import cv2
from numpy import ndarray

class InputFeeder:
    def __init__(self, input_type, input_file=None):
        '''
        input_type: str, The type of input. Can be 'video' for video file, 'image' for image file,
                    or 'cam' to use webcam feed.
        input_file: str, The file that contains the input image or video file. Leave empty for cam input_type.
        '''
        print ("Start __init__ InputFeeder")
        self.input_type=input_type
        if input_type=='video' or input_type=='image':
            self.input_file=input_file
        print ("End __init__ InputFeeder")
    
    def load_data(self):
        print ("Start load_data from InputFeeder")
        if self.input_type=='video':
            self.cap=cv2.VideoCapture(self.input_file)
        elif self.input_type=='cam':
            self.cap=cv2.VideoCapture(0)
        else:
            self.cap=cv2.imread(self.input_file)
            
        print ("Input type: ", self.input_type)
        #print ("Input file: ",self.input_file)
        print ("Cap: ", self.cap)
        self.initial_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #print("initial_w: " + str(self.initial_w))
        print ("End load_data from InputFeeder")
        return self.cap

    def next_batch(self):
        '''
        Returns the next image from either a video file or webcam.
        If input_type is 'image', then it returns the same image.
        '''
        while True:
            for _ in range(10):
                _, frame=self.cap.read()
            yield frame


    def close(self):
        '''
        Closes the VideoCapture.
        '''
        if not self.input_type=='image':
            self.cap.release()

