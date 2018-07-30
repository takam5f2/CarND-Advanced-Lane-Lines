## Advanced Lane Finding

In this project, my pipeline for lane detection is designed. Details about my project is shown in the [writeup](https://github.com/takam5f2/CarND-Advanced-Lane-Lines/blob/master/writeup.md) Scope of my project is to apply my pipeline to [project_video.mp4](./project_video.mp4). Now I've excluded change video, but I will try them after first submission.

Important Directory and Files:
---
Important files and directory is:

* writeup.md: it describe how I implemented my pipeline

* src/: this directory includes the source code of my pipeline

* output_images/: this directory includes the result picture of lane detection

* src/project_video_after_lane_detection.mp4

The movie which was generated withapplying my lane detection pipeline to [project_video.mp4](./project_video.mp4).

src/ directory has several files as follows.

* apply_pipeline_movies.py

If you execute this, the movie will be generated

* LaneDetectionPipeline.py

Construction of Lane Detection Pipeline, and it is my pipeline

* LineCollection.py

It includes definition of class for express line data

* CameraCalib.py

Camera Calibration class is defined in this file, and it can use existing calibration parameter file called "calibration_pickle.p"

* EdgeDetection.py

Sub pipeline for edge detection with using Sobel and Color space conversion

* SobelCollection.py

The class definitions of Sobel function, SobelGratitude, and SobelDirection are described in this file

* ColorElementCollection.py

The class for abstracting color space is defined in this source file

* PerspectiveTransform.py

This include definition of class for perspective transformation.

* LineDetection.py

This source file includes the class and function(method) to detect both left and right line with using histogram and fitting(linear regression)

Report
---

Please check [writeup](https://github.com/takam5f2/CarND-Advanced-Lane-Lines/blob/master/writeup.md)

If you are interested in movies including result of detection, please watch at [src/project_video_after_lane_detection.mp4](./src/project_video_after_lane_detection.mp4)


Execution
---
clone my project

`$ git clone https://github.com/takam5f2/CarND-Advanced-Lane-Lines.git`

execution area is defined as src/ direction

`$ cd src/`

if you get pictures whose lane was detected by using my pipeline

`$ python LaneDetectionPipeline.py`

After execution, the generated pictures are located on output_images directory.

if you get movies whose lane was detected by using my pipeline

`$ python apply_pipeline_movies.py`

after execution, [src/project_video_after_lane_detection.mp4](./src/project_video_after_lane_detection.mp4) appears