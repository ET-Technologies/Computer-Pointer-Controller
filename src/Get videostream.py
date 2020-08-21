    # Check for Webcam
    if args.input =="CAM":
        input_stream = 0
        
    # Check for Image (jpg, bmp, png)
    elif args.input.endswith(".jpg") or args.input.endswith(".bmp") or args.input.endswith(".png") :
        single_image_mode = True
        input_stream = args.input
        
    # Check for video    
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "There is no video file"
# or

    def load_data(self):
        if self.input_type=='video':
            self.cap=cv2.VideoCapture(self.input_file)
        elif self.input_type=='cam':
            self.cap=cv2.VideoCapture(0)
        else:
            self.cap=cv2.imread(self.input_file)

    # Get the input frame
    #cap = cv2.VideoCapture(video)
    try:
        print("Reading video file name:", left_eye)
        cap = cv2.VideoCapture(left_eye)
        right_cap = cv2.VideoCapture(right_eye)
        cap.open(left_eye)
        right_cap.open(right_eye)
        if not path.exists(left_eye):
            print("Cannot find video file: " + left_eye)
    except FileNotFoundError:
        print("Cannot find video file: " + left_eye)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
