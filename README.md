# Computer Pointer Controller
## Intro
The computer pointer application is used to control the mouse pointer through the user's gaze. A pipeline of four Openvino models will work together to accomplish this task.
* Face Detection Model
* Landmark Detection Model
* Head Pose Estimation Model
* Gaze Estimation Model

An input image (video file or webcam feed) is send to the face detection model. From there a cropped face image is send to the landmark model and the head pose model. The landmark model provides the cropped left and right eye image and the head model provides the head pose angels. Both inputs wil be feed to the gaze model which has as output a gaze vector which could be feed to the pyautogui which controls the mouse pointer.

## Project Set Up and Installation

The basic requirement for this program is the Intel OpenVINO toolkit (version >= 2020.1). 
* Please install [OpenVino](
https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html) and make sure you have python > 3.6
* Clon following Github repository  [Github](https://github.com/ET-Technologies/Computer-Pointer-Controller.git)
* Check the `requirement.txt` and if nesseary intall missing dependencies.
* Download missing models if nessesary and load it in the intended folder. (project directory structure)
* Start the program with the provided `arguments`. Please keep in mind, to change paths if you have a different directory struture or want to run the script with different arguments.
* Check logging_basic.log problems accure or logging_time.log if you are interessed in loading time respectively inference times.

### Following model are needed:
* [face-detection-adas-binary-0001](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
* [landmarks-regression-retail-0009](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
* [head-pose-estimation-adas-0001](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
* [gaze-estimation-adas-0002](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

#### project directory structure
<pre>
├── bin
│   └── demo.mp4
├── log
│   ├── logging_basic.log
│   ├── logging_time.log
│ 
├── models
│   └── 2020.4.1
│       └─── model_precision
│               ├── model_name.bin
│               └── model_name.xml
├── README.md
├── requirements.txt
├── src
    ├── computer_pointer.py
    ├── face_detection.py
    ├── facial_landmarks_detection.py
    ├── gaze_estimation.py
    ├── head_pose_estimation.py
    ├── input_feeder.py
    └── mouse_controller.py
</pre>



## Demo
If you want to run the program, following steps are nessesary:

**1. cd to `src` folder**

**2. Arguments to run the program** with e.g. **FP16**, **CPU** input **video**
<pre>
python3 src/computer_pointer.py \
--video bin/demo.mp4 \
--output_path output/demo_output.mp4 \
--fd_model models/2020.4.1/FP32/face-detection-adas-binary-0001 \
--fl_model models/2020.4.1/FP16/landmarks-regression-retail-0009 \
--hp_model models/2020.4.1/FP16/head-pose-estimation-adas-0001 \
--ga_model models/2020.4.1/FP16/gaze-estimation-adas-0002 \
--threshold 0.6 \
--input_type video \
--device CPU \
--version 2020 \
--show_image yes
</pre>

**3. Arguments to run the program** with e.g **FP32**, **CPU** input **video**
<pre>
python3 src/computer_pointer.py \
--video bin/demo.mp4 \
--output_path output/demo_output.mp4 \
--fd_model models/2020.4.1/FP32/face-detection-adas-binary-0001 \
--fl_model models/2020.4.1/FP32/landmarks-regression-retail-0009 \
--hp_model models/2020.4.1/FP32/head-pose-estimation-adas-0001 \
--ga_model models/2020.4.1/FP32/gaze-estimation-adas-0002 \
--threshold 0.6 \
--input_type video \
--device CPU \
--version 2020 \
--show_image yes
</pre>
**4. Arguments to run the program** with e.g **INT8**, **CPU** input **video**
<pre>
python3 src/computer_pointer.py \
--video bin/demo.mp4 \
--output_path output/demo_output.mp4 \
--fd_model models/2020.4.1/FP32/face-detection-adas-binary-0001 \
--fl_model models/2020.4.1/FP16-INT8/landmarks-regression-retail-0009 \
--hp_model models/2020.4.1/FP16-INT8/head-pose-estimation-adas-0001 \
--ga_model models/2020.4.1/FP16-INT8/gaze-estimation-adas-0002 \
--threshold 0.6 \
--input_type video \
--device CPU \
--version 2020 \
--show_image yes
</pre>
## Documentation

The project needs some basic input and some optimal input. The models for face recognition, head posture, landmark regression, and gaze estimation are required. You can choose your device and the  extension (if not required). 'video' If your input is a video or image, you can determine the path. The 'output_path' determines the path to the output video. The 'threshold' is 60% by default, but you can change it here. The 'input_type' allows you to select video, camera or image depending on the 'video' part. The 'version' determines the OpenVino version. The CPU expansion is no longer required since version 2020. We therefore recommend running the program with the openvino 2020.x version.
**If you need more help try `python3 src/computer_pointer.py` to get help for the command line arguments**
<pre>
--fd_model: 'Path to the face model'
--fl_model: 'Path to the landmark model'
--hp_model: 'Path to the head pose model'
--ga_model: path to the gaze model
--device: 'CPU' or 'MYRIAD'
--extension: 'Path to the extension if needed.'
--video: 'Path to the input video if you run it with input_type video'
--output_path: 'Path to the output video'
--threshold: 'Threshold for the face detection. Default 60%'
--input_type: 'CAM for a webcamera or video for an inputfile'
--version: 'Openvino version 2020 or 2019 (recommendation 2020)'
--show_image_help: 'yes to show ouput images, no to hide images'
</pre>

## Benchmarks
The model load time and the inference time can be found in the log protocol.logging_basic.log or logging_time.log
In my case the program was testet with `CPU` only, but with different precision levels. `FP32, FP16, INT8`
Note that the face model just come with precision FP32. All other models have all three precisions. 
Following results occurred while testing it with the demo video file:

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.
The shortest model loading time was with FP32. 357ms FP16 and INT8 had nearly the some loading time.

### Model load time ***FP32***
<pre>
INFO Facedetection load time: 126.828
INFO Facial_Landmarks load time: 75.75
INFO Headpose load time: 72.87
INFO Gaze load time: 86.088
INFO Total model load time: 361.893
</pre>

### Model load time ***FP16***
<pre>
INFO Facedetection load time: 134.411
INFO Facial_Landmarks load time: 82.6
INFO Headpose load time: 103.659
INFO Gaze load time: 137.994
INFO Total model load time: 458.994
</pre>

### Model load time ***INT8***
<pre>
INFO Facedetection load time: 118.046
INFO Facial_Landmarks load time: 104.521
INFO Headpose load time: 168.995
INFO Gaze load time: 190.046
INFO Total model load time: 582.289
</pre>
### Average inference time ***FP32***
<pre>
INFO Average face inference time: 76.81961786949029
INFO Average facial inference time: 2.027689400365797
INFO Average headpose inference time: 3.1615717936370333
INFO Average gaze inference time: 4.442558450213934
</pre>
### Average inference time ***FP16***
<pre>
INFO Average face inference time: 77.14002415285272
INFO Average facial inference time: 2.157453763282905
INFO Average headpose inference time: 3.1131283711578885
INFO Average gaze inference time: 4.4506081080032605
</pre>
### Average inference time ***INT8***
<pre>
INFO Average face inference time: 78.2736277176162
INFO Average facial inference time: 2.1134837199065646
INFO Average headpose inference time: 2.7073803594556907
INFO Average gaze inference time: 4.1432542315984175
</pre>

### Comparison: Total model load time/Inference time
||Total Model Load time|Total Inference Time|
| ------ | ------ | ------ |
|FP32|361.893 |5859 |
|FP16|458.994 |5836 |
|FP16-INT8|582.289 |5925 |

## Stand Out Suggestions
My intention was to solve `two problems`. First, to find a solution for the computer pointer as a whole (which was required) and, second, that each model can run on its own. For this reason I have also included parts like argparser, main () for each part (e.g. face_detection.py), which are only needed when this part is executed on its own. I also added an `output video` portion to record the session.

### Problems
- If you run the program with the demo video as input, you shouldn't have any inference problems. However, if you are running the program with a webcam, the inference may be interrupted due to light or multiple people in the frame.
- You can try changing the threshold or the precision of the model. Even so, it will break from time to time. To solve this problem, one could train better models or, if there are several people on the screen, only keep track of one of them to avoid inference breaks.
