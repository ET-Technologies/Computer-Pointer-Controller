# Computer Pointer Controller


## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.

For this project the Intel OpenVino toolkit is required (version >= 2020.1).

### Following model are needed:
* [face-detection-adas-binary-0001](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
* [landmarks-regression-retail-0009](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
* [head-pose-estimation-adas-0001](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
* [gaze-estimation-adas-0002](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

<pre>
├── bin
│   └── demo.mp4
├── log
│   ├── logging_basic.log
├── models
│   └── model_source
│       └── model_name
│           └── model_precision
│               ├── model_name.bin
│               └── model_name.xml
├── README.md
├── requirements.txt
├── src
    ├── face_detection.py
    ├── facial_landmarks_detection.py
    ├── gaze_estimation.py
    ├── head_pose_estimation.py
    ├── input_feeder.py
    ├── main_computer_pointer_controller.py
    ├── model.py
    └── mouse_controller.py
</pre>



## Demo
*TODO:* Explain how to run a basic demo of your model.

**1. cd to main folder**

**2. Arguments to run the program with**
-**FP16**, **CPU** input **video**
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

**2a. Arguments to run the program with**
-**FP32**, **CPU** input **video**
<pre>
python3 src/computer_pointer.py \
--video bin/demo.mp4 \
--output_path output/demo_output.mp4 \
--fd_model models/2020.4.1/FP32-INT1/face-detection-adas-binary-0001 \
--fl_model models/2020.4.1/FP32/landmarks-regression-retail-0009 \
--hp_model models/2020.4.1/FP32/head-pose-estimation-adas-0001 \
--ga_model models/2020.4.1/FP32/gaze-estimation-adas-0002 \
--threshold 0.6 \
--input_type video \
--device CPU \
--version 2020
</pre>
## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.
<pre>
--fd_model", required=True)
--fl_model", required=True)
--hp_model", required=True)
--ga_model", required=True)
'--device', default = 'CPU')
--extension', default= None)
--video', default=None)
--output_path', required=False)
--threshold', type=float, default=0.6)
--input_type', required=False)
--version', default='2020', required=False)
</pre>

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

### Model load time
|Model|Type|Size|Load time|
| ------ | ------ | ------ |------ |------ |
|face-detection-adas-binary-0001|FP32-INT1|1.86 MB|402.233 ms|50.7061845165188 ms|

### Total time to load all models


## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
