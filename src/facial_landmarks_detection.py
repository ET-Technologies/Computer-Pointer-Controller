'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
from openvino.inference_engine import IECore


class Facial_Landmarks:
    '''
    Class for the Face Detection Model.
    '''
    #(model_name=args.fl_model, threshold=args.threshold, device=args.device, extension=args.extension, version=args.version)
    def __init__(self, model_name, threshold, device, extensions, version):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.plugin = IECore()
        ## check if read model without problem
        self.check_model(self.model_structure, self.model_weights)
        self.exec_net = None
        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_names = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_names].shape


## check supported layer and performence counts reference: 
# https://gist.github.com/justinshenk/9917891c0433f33967f6e8cd8fcaa49a
    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
        layers_unsupported = [ul for ul in self.network.layers.keys() if ul not in supported_layers]


        if len(layers_unsupported)!=0 and self.device=='CPU':
            print("unsupported layers found: {}".format(layers_unsupported))
            
            if self.extensions!=None:
                print("Adding cpu_extension now")
                self.plugin.add_extension(self.extensions, self.device)
                supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
                layers_unsupported = [ul for ul in self.network.layers.keys() if ul not in supported_layers]
                
                if len(layers_unsupported)!=0:
                    print("Please try again! unsupported layers found after adding the extensions.  device {}:\n{}".format(self.device, ', '.join(layers_unsupported)))
                    print("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
                    exit(1)
                print("Problem is resolved after adding the extension!")
                
            else:
                print("Please give the right path of cpu extension!")
                exit(1)

        self.exec_net = self.plugin.load_network(network=self.network, device_name=self.device, num_requests=1)


    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        processed_input = self.preprocess_input(image.copy())
        outputs = self.exec_net.infer({self.input_name:processed_input})

        #if perf_flag:
        #    self.performance()

        coords = self.preprocess_output(outputs)

        # print(image.shape)
        h, w = image.shape[0], image.shape[1]

        coords = coords* np.array([w, h, w, h])
        # print(coords) # [ 39.50625494 130.00975445 146.11010522 126.54997483]        
        coords = coords.astype(np.int32) #(lefteye_x, lefteye_y, righteye_x, righteye_y)
        # print(coords) # [ 39 130 146 126]

        ## left eye moving range
        leye_xmin, leye_ymin=coords[0]-20, coords[1]-20
        leye_xmax, leye_ymax=coords[0]+20, coords[1]+20
        ## right eye moving range
        reye_xmin, reye_ymin=coords[2]-20, coords[3]-20
        reye_xmax, reye_ymax=coords[2]+20, coords[3]+20

        ## draw left and right eye
        # cv2.rectangle(image,(leye_xmin,leye_ymin),(leye_xmax,leye_ymax),(0,255,0), 1)        
        # cv2.rectangle(image,(reye_xmin,reye_ymin),(reye_xmax,reye_ymax),(0,255,0), 1)
        # cv2.imshow("Left Right Eyes",image)

        ## leye_ymin:leye_ymax, leye_xmin:leye_xmax --> left eye heigh, width
        left_eye_box = image[leye_ymin:leye_ymax, leye_xmin:leye_xmax]
        ## reye_ymin:reye_ymax, reye_xmin:reye_xmax --> right eye heigh, width
        right_eye_box = image[reye_ymin:reye_ymax, reye_xmin:reye_xmax]
        # print(left_eye_box.shape, right_eye_box.shape) # left eye and right eye image

        ## [left eye box, right eye box] 
        eyes_coords = [[leye_xmin,leye_ymin,leye_xmax,leye_ymax], [reye_xmin,reye_ymin,reye_xmax,reye_ymax]]

        return left_eye_box, right_eye_box #, eyes_coords


    def check_model(self, model_structure, model_weights):
        # raise NotImplementedError
        try:
            # Reads a network from the IR files and creates an IENetwork, load IR files into their related class, architecture with XML and weights with binary file
            self.network = self.plugin.read_network(model=model_structure, weights=model_weights)
        except Exception as e:
            raise ValueError("Error occurred during facial_landmarks_detection network initialization.")


## check supported layer and performence counts reference: 
# https://gist.github.com/justinshenk/9917891c0433f33967f6e8cd8fcaa49a
    def performance(self):
        perf_counts = self.exec_net.requests[0].get_perf_counts()
        # print('\n', perf_counts)
        print("\n## Facial landmarks detection model performance:")
        print("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type', 'exet_type', 'status', 'real_time, us'))

        for layer, stats in perf_counts.items():            
            print("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer, stats['layer_type'], stats['exec_type'], 
                                                              stats['status'], stats['real_time']))


    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        # print(image.shape)
        # print(image[2][1])
        # cv2.imshow('image',image)
        ## convert RGB to BGR 
        image_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(image_cvt.shape) # (374, 238, 3)
        # cv2.imshow('cvt',image_cvt)
        # print('====',image_cvt[2][1])
        # print(self.input_shape) # [1, 3, 48, 48]
        H, W = self.input_shape[2], self.input_shape[3]
        # print(H, W) # (48, 48)

        image_resized = cv2.resize(image_cvt, (W, H))
        # print(image_resized.shape) # (48, 48, 3)
        ## (optional)
        # image_processed = np.transpose(np.expand_dims(image_resized, axis=0), (0,3,1,2))
        image = image_resized.transpose((2,0,1))
        # print(image.shape) # (3, 48, 48)
        # add 1 dim at very start, then channels then H, W
        image_processed = image.reshape(1, 3, self.input_shape[2], self.input_shape[3])
        # print(image_processed.shape) # (1, 3, 48, 48)

        return image_processed


    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.

        The net outputs a blob with the shape: [1, 10], containing a row-vector of 10 floating point values 
        for five landmarks coordinates in the form (x0, y0, x1, y1, ..., x5, y5). 
        All the coordinates are normalized to be in range [0,1].
        '''
        # print(output)
        # print(outputs[self.output_names].shape) # (1, 10, 1, 1)
        # print(outputs[self.output_names][0].shape) # (10, 1, 1)
        # print(outputs[self.output_names][0])        
        # print('-----', outputs[self.output_names][0][0])

        ## here only need left eye and right eye
        outs = outputs[self.output_names][0]
        # print(outs.shape)
        # print(outs[0][0][0])
        # print(outs[0].tolist()) # [[0.37333157658576965]]
        # print(outs[0].tolist()[0][0]) # [[0.37333157658576965]]        
        # print(type(outs)) # numpy.ndarry

        leye_x, leye_y = outs[0][0][0], outs[1][0][0]
        reye_x, reye_y = outs[2][0][0], outs[3][0][0]
        coords_lr = (leye_x, leye_y, reye_x, reye_y)
        # print(coords_lr)

        return coords_lr
