## Downloading Models

Go to directory:

`cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader`

**Downloading [Human Pose Model](https://docs.openvinotoolkit.org/latest/_models_intel_human_pose_estimation_0001_description_human_pose_estimation_0001.html)**

`sudo ./downloader.py --name human-pose-estimation-0001 -o /home/workspace`

**Downloading [Text Detection Model](http://docs.openvinotoolkit.org/latest/_models_intel_text_detection_0004_description_text_detection_0004.html)** (with precision)

`sudo ./downloader.py --name text-detection-0004 --precisions FP16 -o /home/workspace`

**Downloading [Car Metadata Model](https://docs.openvinotoolkit.org/latest/_models_intel_vehicle_attributes_recognition_barrier_0039_description_vehicle_attributes_recognition_barrier_0039.html)**

`sudo ./downloader.py --name vehicle-attributes-recognition-barrier-0039 --precisions INT8 -o /home/workspace`

### Pre-Trained Model Optimizations
- OpenVino adds different precisions. The lower the precision, the lower the memory and compute
- It also adds fusing layers leading to fewer operations
