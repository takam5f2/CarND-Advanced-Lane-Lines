"""
Pipeline for line detection
"""
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

from CameraCalib import CameraCalib
from EdgeDetection import EdgeDetection
from ColorElementCollection import AbstGray, AbstRGBElement, AbstHLSElement
from SobelCollection import Sobel, SobelGratitude, SobelDirection
from PerspectiveTransform import PerspectiveTransform
from LineDetection import LineDetection
from LineCollection import LinePolynomial, LineCurvature, Line

def sanity_check_curvature(left_curvature, right_curvature):
    permitted_rate = 6
    permitted_min = left_curvature/permitted_rate
    permitted_max = left_curvature*permitted_rate
    if right_curvature >  permitted_min and right_curvature < permitted_max:
        return True
    return False

def sanity_check_distance(binary_warped, leftpoly, rightpoly):
    offset = 70
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    center_of_lane    = (leftpoly.deduce(max(ploty)) + rightpoly.deduce(max(ploty))) / 2
    pos_center_max = (binary_warped.shape[1] / 2) + offset
    pos_center_min = (binary_warped.shape[1] / 2) - offset
    if center_of_lane > pos_center_min and center_of_lane < pos_center_max:
        return True
    return False

def sanity_check_parallel(binary_warped, leftpoly, rightpoly):
    offset = 0.95
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    leftpoly_one = LinePolynomial(1)
    rightpoly_one = LinePolynomial(1)
    leftpoly_one.fit(ploty, leftpoly.deduce(ploty))
    rightpoly_one.fit(ploty, rightpoly.deduce(ploty))
    leftpoly_vec = np.array([leftpoly_one.fit_coef[0], 1])
    rightpoly_vec = np.array([rightpoly_one.fit_coef[0], 1])
    cos_angle = np.dot(leftpoly_vec, rightpoly_vec) / (np.linalg.norm(leftpoly_vec) * np.linalg.norm(rightpoly_vec))
    if abs(cos_angle) > offset:
        return True
    return False
    
def sanity_check(binary_warped, leftpoly, rightpoly, left_curve, right_curve):
    counter = 0
    counter +=  sanity_check_curvature(left_curve.curverad_real, right_curve.curverad_real)
    counter += sanity_check_distance(binary_warped, leftpoly, rightpoly)
    counter += sanity_check_parallel(binary_warped, leftpoly, rightpoly)
    if counter >= 2:
        return True
    return False

def get_line_curvature(binary_warped, ploty, plotx):
    line_curvature = LineCurvature()
    line_curvature.set_plot(out_img, ploty, plotx)
    return line_curvature.curverad_real

def get_lane_curvature(binary_warped, leftpoly, rightpoly):
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = leftpoly.deduce(ploty)
    right_fitx = rightpoly.deduce(ploty)
    lane_fitx = (left_fitx + right_fitx) / 2
    lane_curvature = LineCurvature()
    lane_curvature.set_plot(binary_warped, ploty, lane_fitx)
    return lane_curvature.curverad_real

class LaneDetectionPipeline(object):
    """
    this is the pipeline for detecting lane
    """
    def __init__(self):
        self.cam_calib = None
        self.edge_detection = None
        self.perspec_trans = None
        self.line_detection = None
        # uses movie processing
        self.left = Line()
        self.right = Line()
        

    def configure_auto(self):
        # Camera Calibration
        self.cam_calib = CameraCalib(9, 6)
        self.cam_calib.import_calib("calibration_pickle.p")
        # Edge Detection with gradient and color space
        self.edge_detection = EdgeDetection()
        # add function
        self.edge_detection.add_func(Sobel(7, 20, 120, 'x'))
        self.edge_detection.add_func(SobelGratitude(5, 70, 100))
        self.edge_detection.add_func(AbstHLSElement(160, 255, 'S'))
        # Perspective Transformation
        src_range = np.float32([[550, 470], [200, 700], [1100, 700], [750, 470]])
        dst_range = np.float32([[180, 0], [180, 700], [1100, 700], [1100, 0]])
        self.perspec_trans = PerspectiveTransform(src_range, dst_range)
        # Line Detection
        self.line_detection = LineDetection(margin=50)

    def _draw_lane_to_image(self, undist_img, binary_warped, leftpoly, rightpoly):
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = leftpoly.deduce(ploty)
        right_fitx = rightpoly.deduce(ploty)
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = self.perspec_trans.reverse_img(color_warp)
        # Combine the result with the original image
        result = cv2.addWeighted(undist_img, 1, newwarp, 0.3, 0)

        # add curvature information to image
        # calculates center of lane's curvature
        curvature = get_lane_curvature(result, leftpoly, rightpoly)
        curvature_disp = "Curvature: {}".format(curvature)
        cv2.putText(result, curvature_disp, (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
        return result
        
    def process_image_pic(self, src_img):
        undist_img = self.cam_calib.undist_img(src_img)
        out_img = self.edge_detection.execute(undist_img)
        out_img = self.perspec_trans.warp_img(out_img)
        leftx, lefty, rightx, righty = self.line_detection.find_lane_pixels(out_img)
        leftpoly = LinePolynomial(2)
        rightpoly = LinePolynomial(2)
        leftpoly.fit(lefty, leftx)
        rightpoly.fit(righty, rightx)
        
        leftx, lefty, rightx, righty = self.line_detection.search_around_poly(out_img, leftpoly, rightpoly)
        leftpoly.fit(lefty, leftx)
        rightpoly.fit(righty, rightx)

        left_curve = LineCurvature()
        right_curve = LineCurvature()
        left_curve.set_plot(out_img, lefty, leftx) 
        right_curve.set_plot(out_img, righty, rightx)
        # get curvature

        out_img = self._draw_lane_to_image(undist_img, out_img, leftpoly, rightpoly)
        return out_img
    
    def process_img_movie(self, src_img):
        # camera calibration, edge_detection, perspective transformation.
        undist_img = self.cam_calib.undist_img(src_img)
        out_img = self.edge_detection.execute(undist_img)
        out_img = self.perspec_trans.warp_img(out_img)
        # if lane detection succeeded at last cycle, fitting only used for finding lane in this cycle
        # but otherwise uses histogram based approach
        # here, all pixels around the line are detected.
        if self.left.detected == True and self.right.detected == True:
            leftx, lefty, rightx, righty = self.line_detection.search_around_poly(out_img,
                                                                                  self.left.linepoly, self.right.linepoly)
        else:
            leftx, lefty, rightx, righty = self.line_detection.find_lane_pixels(out_img)
        # 
        self.left.linepoly.fit(lefty, leftx)
        self.right.linepoly.fit(righty, rightx)
        # calculate curvature of both right line and left line
        left_curve = LineCurvature()
        right_curve = LineCurvature()
        left_curve.set_plot(out_img, lefty, leftx) 
        right_curve.set_plot(out_img, righty, rightx)
        # sanity check execution such as curvature, distance and 
        detected = sanity_check(out_img, self.left.linepoly, self.right.linepoly, left_curve, right_curve)
        # update line detection information
        self.left.update_lines(detected, out_img, leftx, lefty,
                               left_curve.curverad_real)
        self.right.update_lines(detected, out_img, rightx, righty,
                               right_curve.curverad_real)
        left_fitx = None
        right_fitx = None
        ploty = np.linspace(0, out_img.shape[0]-1, out_img.shape[0])
        # if detected line does not exist, uses recent_xfitted
        # if recent_xfitted does not exist, current_fit will be used for substitution(giving up).
        if detected == False:
            if len(self.left.recent_xfitted) > 0:
                left_fitx = self.left.recent_xfitted[-1]
            else:
                interpolatepoly = LinePolynomial()
                interpolatepoly.fit_coef = self.left.current_fit
                left_fitx = interpolatepoly.deduce(ploty)
            if len(self.right.recent_xfitted) > 0:
                right_fitx = self.right.recent_xfitted[-1]
            else:
                interpolatepoly = LinePolynomial()
                interpolatepoly.fit_coef = self.right.current_fit
                right_fitx = interpolatepoly.deduce(ploty)
        else:
            # get fresh fitted data
            left_fitx = self.left.recent_xfitted[-1]
            right_fitx = self.right.recent_xfitted[-1]
        # fitting again
        leftpoly = LinePolynomial()
        leftpoly.fit(ploty, left_fitx)
        rightpoly = LinePolynomial()
        rightpoly.fit(ploty, right_fitx)
        # draw lane to image
        out_img = self._draw_lane_to_image(undist_img, out_img, leftpoly, rightpoly)

        return out_img
        
if __name__ == '__main__':
    src_image_file = glob.glob('../test_images/*.jpg')

    lane_detect_pipeline = LaneDetectionPipeline()
    lane_detect_pipeline.configure_auto()

    for idx, fname in enumerate(src_image_file):
        img = mpimg.imread(fname)
        out_img = lane_detect_pipeline.process_image_pic(img)
        # output image
        outfname = '../output_images/detect_lane_{}.jpg'.format(idx)
        f, ax = plt.subplots(1, 1, figsize=(30, 12))
        f.tight_layout()
        ax.imshow(out_img)
        ax.set_title('Post Processing', fontsize=30)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.savefig(outfname)
        
