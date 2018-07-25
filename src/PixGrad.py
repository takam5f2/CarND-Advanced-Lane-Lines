"""
Get Gradient of pixel in image.
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpmg

class PixGrad(object):
    
    def __init__(self, manner, thresh_min, thresh_max):
        self._manner = manner
        self._thresh_min = thresh_min
        self._thresh_max = thresh_max

    def _get_sobel(self, img, orient='x', sobel_kernel=3)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        else
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        return sobel
        
    def _get_gratitude(self, img, sobel_kernel=3):
        sobelx = self._run_sobel(img, 'x', sobel_kernel)
        sobely = self._run_sobel(img, 'y', sobel_kernel)
        gratitude = np.sqrt(np.power(sobelx, 2), np.power(sobely, 2))

        return gratitude

    def _get_orient(self, img, sobel_kernel=3):
        sobelx = self._run_sobel(img, 'x', sobel_kernel)
        sobely = self._run_sobel(img, 'y', sobel_kernel)
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        return absgraddir

    def _sobel_threshold(self, img, orient='x', sobel_kernel=3):
        sobel = self._get_sobel(img, orient, sobel_kernel)
        binary_output = np.zeros_like(sobel)
        binary_output[sobel > self._thresh_min & sobel < self._thresh_max] = 1
        return binary_output

    def _grat_threshold(self, img, sobel_kernel)

    
    # public
    
