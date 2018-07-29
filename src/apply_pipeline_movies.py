"""
Apply pipeline to movie
"""
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from moviepy.editor import VideoFileClip

from CameraCalib import CameraCalib
from EdgeDetection import EdgeDetection
from ColorElementCollection import AbstGray, AbstRGBElement, AbstHLSElement
from SobelCollection import Sobel, SobelGratitude, SobelDirection
from PerspectiveTransform import PerspectiveTransform
from LineDetection import LineDetection
from LineCollection import LinePolynomial, LineCurvature, Line
from LaneDetectionPipeline import sanity_check_curvature, sanity_check_distance, sanity_check_parallel, sanity_check
from LaneDetectionPipeline import LaneDetectionPipeline

pipeline = LaneDetectionPipeline()
pipeline.configure_auto()

def process_image(image):
    result =  pipeline.process_img_movie(image)
    return result

def process_movie(src, dst):
    clip1 = VideoFileClip(src)
    white_clip = clip1.fl_image(process_image)

    white_clip.write_videofile(dst, audio=False)

src_fname = "../project_video.mp4"
dst_fname = 'output_movie/project_video.mp4'
process_movie(src_fname, dst_fname)
