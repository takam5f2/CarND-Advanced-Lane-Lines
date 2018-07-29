"""
Detects Line used for Lane Definition 
"""
from LineCollection import LineCurvature, LinePolynomial, Line
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

class LineDetection(object):
    
    def __init__(self, nwindows=9, margin=100, minpix=100):
        self._nwindows = nwindows
        self._margin = margin
        self._minpix = minpix
        # internal variables

    def _get_initial_point(self, binary_warped):
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        return leftx_base, rightx_base

    def _initilize_windows(self, binary_warped):
        window_height = np.int(binary_warped.shape[0]//self._nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        return window_height, nonzeroy, nonzerox

    def find_lane_pixels(self, binary_warped):
        # Initialize first detection and window
        leftx_current, rightx_current = self._get_initial_point(binary_warped)
        window_height, nonzeroy, nonzerox = self._initilize_windows(binary_warped)
        left_lane_inds = []
        right_lane_inds = []

        for window_idx in range(self._nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window_idx+1)*window_height
            win_y_high = binary_warped.shape[0] - window_idx*window_height
            # define left/right and upper/bottom positions for each window
            win_xleft_low = leftx_current - self._margin  # Update this
            win_xleft_high = leftx_current + self._margin # Update this
            win_xright_low = rightx_current - self._margin # Update this
            win_xright_high = rightx_current + self._margin  # Update this

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            ### If found > minpix pixels, recenter next window ###
            ### (`right` or `leftx_current`) on their mean position ###
            if good_left_inds.shape[0] > self._minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if good_right_inds.shape[0] > self._minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))
                # pass # Remove this when you add your function

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        return leftx, lefty, rightx, righty

    def visualization(self, binary_warped, leftx, lefty, rightx, righty, polyleft, polyright):
        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = polyleft.deduce(ploty)
        right_fitx = polyright.deduce(ploty)
        
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-self._margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+self._margin, 
                                                                        ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-self._margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+self._margin, 
                                                                         ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
        # Plot the polynomial lines onto the image
        left_line = np.dstack((left_fitx, ploty))
        left_line = np.array(left_line, np.int32)
        left_line = left_line.reshape((-1,1,2))
        result = cv2.polylines(result,[left_line],False,(255,255,0), thickness=3)
        right_line = np.dstack((right_fitx, ploty))
        right_line = np.array(right_line, np.int32)
        right_line = right_line.reshape((-1,1,2))
        result = cv2.polylines(result,[right_line],False,(255,255,0), thickness=3)
        return result
    
    def search_around_poly(self, binary_warped, leftpoly, rightpoly):
        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        ### Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        left_lane_inds = ((nonzerox > (leftpoly.deduce(nonzeroy) - self._margin)) &
                          (nonzerox < (leftpoly.deduce(nonzeroy) + self._margin)))
        
        right_lane_inds = ((nonzerox > (rightpoly.deduce(nonzeroy) - self._margin)) &
                           (nonzerox < (rightpoly.deduce(nonzeroy) + self._margin)))
    
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty
    
    def execute(self, binary_warped):
        # Find our lane pixels first
        leftx, lefty, rightx, righty = self.find_lane_pixels(binary_warped)
        leftpoly = LinePolynomial(2)
        rightpoly = LinePolynomial(2)
        leftpoly.fit(lefty, leftx)
        rightpoly.fit(righty, rightx)
        
        leftx, lefty, rightx, righty = self.search_around_poly(binary_warped, leftpoly, rightpoly)
        leftpoly.fit(lefty, leftx)
        rightpoly.fit(righty, rightx)
        out_img = self.visualization(binary_warped, leftx, lefty, rightx, righty, leftpoly, rightpoly)
        
        return out_img
        
if __name__ == '__main__':
    import glob
    from CameraCalib import CameraCalib
    from EdgeDetection import EdgeDetection
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from ColorElementCollection import AbstGray, AbstRGBElement, AbstHLSElement
    from SobelCollection import Sobel, SobelGratitude, SobelDirection
    from PerspectiveTransform import PerspectiveTransform
    import copy
    
    cam_calib = CameraCalib(9, 6)
    src_image_file = glob.glob('../test_images/*.jpg')
    cam_calib.import_calib("calibration_pickle.p")

    edge_detection = EdgeDetection()
    # add function
    edge_detection.add_func(Sobel(7, 20, 120, 'x'))
    edge_detection.add_func(SobelGratitude(5, 70, 100))
    edge_detection.add_func(AbstHLSElement(170, 255, 'S'))

    src_range = np.float32([[550, 480], [220, 700], [1150, 700], [750, 480]])
    dst_range = np.float32([[180, 0], [180, 700], [1100, 700], [1100,100]])
    src_range = np.float32([[550, 470], [200, 700], [1100, 700], [750, 470]])
    dst_range = np.float32([[180, 0], [180, 700], [1100, 700], [1100, 0]])
    
    ptransform = PerspectiveTransform(src_range, dst_range)
    ldetect = LineDetection(margin=50)
    for idx, fname in enumerate(src_image_file):
    
        org_img = mpimg.imread(fname)
        calibed_img = cam_calib.undist_img_file(fname)
        edet_img = edge_detection.execute(calibed_img)
        trans_img = ptransform.warp_img(edet_img)
        interest_img = copy.deepcopy(org_img)
        ldet_img = ldetect.execute(trans_img)

        src_rect = np.array(src_range, np.int32)
        src_rect = src_rect.reshape((-1, 1, 2))
        cv2.polylines(interest_img, [src_rect], True, (255, 255, 255))
        
        outfile = './detect/detect_{}.png'.format(idx)
        f, ax = plt.subplots(2, 2, figsize=(30, 12))
        f.tight_layout()

        ax[0,0].imshow(org_img)
        ax[0,0].set_title('Edge Detection Image', fontsize=30)
        ax[0,1].imshow(edet_img, cmap='gray')
        ax[0,1].set_title('Edge Detection Image', fontsize=30)
        
        ax[1,0].imshow(interest_img, cmap='gray')
        ax[1,0].set_title('Edge Detection Image', fontsize=30)
        ax[1,1].imshow(ldet_img)
        ax[1,1].set_title('Line Detection', fontsize=30)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.0)
        plt.savefig(outfile)

    
