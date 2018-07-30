"""
Apply pipeline to movie
"""
import glob
from moviepy.editor import VideoFileClip

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
dst_fname = './project_video_after_lane_detection.mp4'
# src_fname = "../challenge_video.mp4"
# dst_fname = 'output_movie/challenge_video.mp4'
process_movie(src_fname, dst_fname)
