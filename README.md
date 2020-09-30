# Computer Pointer Controller

The computer pointer app is used to navigate the mouse position through the user's gaze. A pipeline of four Openvino models will work together to accomplish this task.

## Project Set Up and Installation

The basic requirement for this program is the Intel OpenVINO toolkit (version >= 2020.1).[link](
https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html)
* Please install OpenVINO and make sure you have python > 3.6
* Clon following Github repository  [link](https://github.com/ET-Technologies/Computer-Pointer-Controller.git)
* Check the requirement.txt and if nesseary intall missing dependencies.
* Download missing models if nessesary and load it in the intended folder. (project directory structure)
* Start the program with the provided arguments. Please keep in mind, to change paths if you have a different directory struture or want to run the script with different arguments.
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
    ├── main_computer_pointer_controller.py
    ├── model_original.py
    └── mouse_controller.py
</pre>



## Demo
If you want to run the program, following arguments are nessesary:

**1. cd to main folder**

**2. Arguments to run the program** with e.g. **FP16**, **CPU** input **video**
<pre>
python3 src/computer_pointer.py \
--video bin/demo.mp4 \
--output_path output/demo_output.mp4 \
--fd_model models/2020.4.1/FP32-INT1/face-detection-adas-binary-0001 \
--fl_model models/2020.4.1/FP16-INT8/landmarks-regression-retail-0009 \
--hp_model models/2020.4.1/FP16-INT8/head-pose-estimation-adas-0001 \
--ga_model models/2020.4.1/FP16-INT8/gaze-estimation-adas-0002 \
--threshold 0.6 \
--input_type video \
--device CPU \
--version 2020
</pre>

**2a. Arguments to run the program** with e.g **FP32**, **CPU** input **video**
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
--version 2020
</pre>
**2b. Arguments to run the program** with e.g **INT8**, **CPU** input **video**
<pre>
python3 src/computer_pointer.py \
--video bin/demo.mp4 \
--output_path output/demo_output.mp4 \
--fd_model models/2020.4.1/FP32-INT1/face-detection-adas-binary-0001 \
--fl_model models/2020.4.1/FP16-INT8/landmarks-regression-retail-0009 \
--hp_model models/2020.4.1/FP16-INT8/head-pose-estimation-adas-0001 \
--ga_model models/2020.4.1/FP16-INT8/gaze-estimation-adas-0002 \
--threshold 0.6 \
--input_type video \
--device CPU \
--version 2020
</pre>
## Documentation

The project needs some basic input and some optimal input. The models for face recognition, head posture, landmark regression, and gaze estimation are required. You can choose your device and the  extension (if not required). 'video' If your input is a video or image, you can determine the path. The 'output_path' determines the path to the output video. The 'threshold' is 60% by default, but you can change it here. The 'input_type' allows you to select video, camera or image depending on the 'video' part. The 'version' determines the OpenVino version. The CPU expansion is no longer required since version 2020. We therefore recommend running the program with the openvino 2020.x version.
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
In my case the program was testet with CPU only, but with different precision levels. FP32, FP16, INT8
Note that the face model just come with precision FP32. All other models have all three precisions. 
Following results occurred:

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.
The shortest model loading time was with FP32. 357ms FP16 and INT8 had nearly the some loading time.

### Model load time ***FP32***
<pre>
2020-09-28 15:48:21,231 INFO Facedetection load time: 115.005
2020-09-28 15:48:21,309 INFO Facial_Landmarks load time: 78.372
2020-09-28 15:48:21,386 INFO Headpose load time: 76.256
2020-09-28 15:48:21,474 INFO Gaze load time: 87.544
2020-09-28 15:48:21,474 INFO Total model load time: 357.765
</pre>

### Model load time ***FP16***
<pre>
2020-09-28 15:46:57,388 INFO Facedetection load time: 125.031
2020-09-28 15:46:57,492 INFO Facial_Landmarks load time: 103.628
2020-09-28 15:46:57,657 INFO Headpose load time: 164.544
2020-09-28 15:46:57,840 INFO Gaze load time: 182.565
2020-09-28 15:46:57,840 INFO Total model load time: 576.545
</pre>

### Model load time ***INT8***
<pre>
2020-09-28 15:49:09,093 INFO Facedetection load time: 128.538
2020-09-28 15:49:09,193 INFO Facial_Landmarks load time: 99.551
2020-09-28 15:49:09,354 INFO Headpose load time: 160.507
2020-09-28 15:49:09,541 INFO Gaze load time: 187.431
2020-09-28 15:49:09,541 INFO Total model load time: 576.638
</pre>
### Average inference time ***FP32***
<pre>
2020-09-28 16:34:50,371 INFO Average face inference time: 79.9724934464794
2020-09-28 16:34:50,384 INFO Average facial inference time: 1.7137810335320942
2020-09-28 16:34:50,386 INFO Average headpose inference time: 2.864142595711401
2020-09-28 16:34:50,390 INFO Average gaze inference time: 4.331091702994653
</pre>
## Stand Out Suggestions
My intention was to solve two problems. First, to find a solution for the computer pointer as a whole (which was required) and, second, that each model can run on its own. For this reason I have also included parts like argparser, main () for each part (e.g. face_detection.py), which are only needed when this part is executed on its own. I also added an output video portion to record the session.

### Problems
If you run the program with the demo video as input, you shouldn't have any inference problems. However, if you are running the program with a webcam, the inference may be interrupted due to light or multiple people in the frame.
You can try changing the threshold or the precision of the model. Even so, it will break from time to time. To solve this problem, one could train better models or, if there are several people on the screen, only keep track of one of them to avoid inference breaks.
