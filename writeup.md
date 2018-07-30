## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/post_calib_comparisno_9.png "Undistorted"
[image2]: ./output_images/calib22.jpg "Road Transformed"
[image3]: ./output_images/binary_combo_example.png "Binary Example"
[image4]: ./output_images/warped.png "Warp Example"
[image5]: ./output_images/color_fit_lines.png "Fit Visual"
[image6]: ./output_images/detect_lane_2.jpg "Output"
[video1]: ./src/project_video_after_lane_detection.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step in pipeline is contained line #89, #90, #133, and #157 in LaneDetectionPipeline.py. #89 and #90 are for configuration, #133 and #157 are dedicated to image processing and movie processing respectively.
The main code for this step is contained in lines #11 through #66 of the file called `CameraCalib.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `self._objpoints` and `self._imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines #92 through #96 , line #134 and line #158 in `LaneDetectionPipeline.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a method called `warp_img`(wrapper method) and `_warp_img`, which appears in lines 21 through 32 in the file `PerspectiveTransform.py` (output_images/examples/example.py) .  The `warp_img` method takes as inputs an image (`img`), with using member variable `self._src_range` and `self._dst_range` of `PerspectiveTransform` class.  I chose the hardcode the source and destination points in the following manner:

```python
self._src_range = np.float32([[550, 470],
		  	      [200, 700],
			      [1100, 700],
			      [750, 470]])
			       
self._dst_range = np.float32([[180, 0],
 		    	      [180, 700],
			      [1100, 700],
			      [1100, 0]])
 
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 550, 470      | 180,    0     | 
| 200, 700      | 180,  700     |
| 1100,700      | 1100, 700     |
| 750, 470      | 1100,   0     |

I verified that my perspective transform was working as expected by drawing the `self._src_range` and `self._dst_range` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines #141 through #143 in my code in `my_other_file.py`
Main function is introduced as `get_lane_curvature` and `get_distance_to_center`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines #118 through #149 in my code in `LaneDetectionPipeline.py` in the method `LaneDetectionPipeline._draw_lane_to_image()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I have beyed the recommended step for building my pipeline. Especially, I was persistent with Edge detection using Sobel, Sobel gratitude, and HSL conversion. And, I made lots of effort perspective transformation. Both approach histogram based fitting and polynomial-based fitting are used to detect lines. Either of the fitting algorithm is chosen according to case.

However, my technique is hard-coded to "project_movie.mp4" thoroughly. So that, it is difficult to apply another movie which includes difficult case; going through tunnel, cutting-in or cutting-out situation, or going normal path(not highway)

I consider to improve my pipeline as line detection is performed to each line separately. Now, right and left lines are detected at the same time. It lose flexibility of building new algorightm.
Also, interporation logic is very poor. Without robust interporation, the pipeline seems to succeed in lane detection. I didn't make effort to improve this logic.

