"""
class for line information
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import copy

class LinePolynomial(object):
    def __init__(self, order=2):
        self._order = order
        self.fit_coef = None

    def fit(self, ypositions, xpositions):
        self.fit_coef = np.polyfit(ypositions, xpositions, self._order)
        return

    def deduce(self, ypositions):
        try:
            xpositions = self.fit_coef[0]*ypositions**2 + self.fit_coef[1]*ypositions + self.fit_coef[2]
        except ValueError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            xpositions = 1*ypositions**2 + 1*ypositions
            
        return xpositions

    def get_curverad(self, y_eval):
        curverad = ((1 + (2*self.fit_coef[0]*y_eval + self.fit_coef[1])**2)**1.5) / np.absolute(2*self.fit_coef[0])
        return curverad

class LineCurvature(object):
    """
    Lane Result storing value of left
    """
    def __init__(self, ym_per_pix=30/720, xm_per_pix=3.7/700):
        # parameter
        self._ym_per_pix = ym_per_pix
        self._xm_per_pix = xm_per_pix
        # result
        self._leftpoly_pix = LinePolynomial()
        self._rightpoly_pix = LinePolynomial()
        self._leftpoly_real = LinePolynomial()
        self._rightpoly_real = LinePolynomial()
        self.left_curverad_pix   = 0
        self.right_curverad_pix  = 0
        self.left_curverad_real  = 0
        self.right_curverad_real = 0

    def _set_plot(self, ploty, plotx, poly_pix, poly_real):
        poly_pix.fit(ploty, plotx)
        poly_real.fit(ploty*self._ym_per_pix, plotx*self._xm_per_pix)
        
    def set_left(self, binary_warped, lefty, leftx):
        self._set_plot(lefty, leftx, self._leftpoly_pix, self._leftpoly_real)
        self.left_curverad_pix = self._leftpoly_pix.get_curverad(max(lefty))
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        self.left_curverad_real = self._leftpoly_real.get_curverad(max(ploty)*self._ym_per_pix)

    def set_right(self, binary_warped, righty, rightx):
        self._set_plot(righty, rightx, self._rightpoly_pix, self._rightpoly_real)
        self.right_curverad_pix = self._rightpoly_pix.get_curverad(max(righty))
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        self.right_curverad_real = self._rightpoly_real.get_curverad(max(ploty)*self._ym_per_pix)

class Line(object):
    """
    Line class served by Udacity
    """
    def __init__(self, frame_num=10, ym_per_pix=30/720, xm_per_pix=3.7/700):
        self._frame_num = frame_num
        self._ym_per_pix = ym_per_pix
        self._xm_per_pix = xm_per_pix
        # polynomial
        self.linepoly = LinePolynomial(2)
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        

    def _update_line_data(self, detected, binary_warped,
                          detectx, detecty, curvature):
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        if detected == True:
            plotx = self.linepoly.deduce(ploty)
            # polynominal
            if len(self.recent_xfitted) == self._frame_num:
                self.recent_xfitted = self.recent_xfitted[1:self._frame_num]
            self.recent_xfitted.append(plotx)
            self.bestx = np.mean(self.recent_xfitted, axis=0)
            self.best_fit = LinePolynomial(2)
            self.best_fit.fit(ploty, self.bestx)
            # fitting function
            self.diffs = self.current_fit - self.linepoly.fit_coef
            self.current_fit = copy.deepcopy(self.linepoly.fit_coef)
            # curvature and position
            self.radius_of_curvature = curvature
            self.line_base_pos = (self.linepoly.deduce(min(ploty)) - (binary_warped.shape[1]/2)) * self._xm_per_pix # distance from center of vehicle
            # all pixels value
            self.allx = detectx
            self.ally = detecty
        else:
            if len(self.recent_xfitted) > 0:
                self.recent_xfitted.pop(0) # delete object whose index is 0
            if len(self.recent_xfitted) > 0:
                self.bestx = np.mean(self.recent_xfitted, axis=0)
                self.best_fit = LinePolynomial(2)
                self.best_fit.fit(ploty, self.bestx)
            # all pixels value
            self.allx = None
            self.ally = None

    def update_lines(self, detected, binary_warped,
                          detectx, detecty, curvature):
        self.detected = detected
        self._update_line_data(detected, binary_warped,
                               detectx, detecty, curvature)
        
    
