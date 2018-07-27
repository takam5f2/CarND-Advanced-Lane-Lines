"""
This module abstracts Color space
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from abc import ABCMeta, abstractmethod

# common function

class AbstractColorElement(metaclass=ABCMeta):
    """
    Abstract class for Color Space Abstraction
    """
    def __init__(self, thresh_min, thresh_max):
        self._thresh_min = thresh_min
        self._thresh_max = thresh_max

    @abstractmethod
    def _get_color_element(self, img):
        raise NotImplemented
    # public function
    def cmp_thres(self, img, thresh_min=0, thresh_max=1):
        color_ele = self._get_color_element(img)
        binary_output = np.zeros_like(color_ele)
        binary_output[(color_ele >= self._thresh_min) & (color_ele <= self._thresh_max)] = 1
        return binary_output


class AbstGray(AbstractColorElement):
    """
    this class uses grayscale to change color space
    """
    def __init__(self, thresh_min, thresh_max):
        super(AbstGray, self).__init__(thresh_min, thresh_max)


    def _get_color_element(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
class AbstRGBElement(AbstractColorElement):
    """
    Abstract RGB Element and change binary image using threshold
    """
    def __init__(self, thresh_min, thresh_max, color='R'):
        super(AbstRGBElement, self).__init__(thresh_min, thresh_max)
        if color == 'R':
            self._color_index = 0
        elif color == 'G':
            self._color_index = 1
        else:
            self._color_index = 2
            
    def _get_color_element(self, img):
        return img[:,:,self._color_index]

class AbstHLSElement(AbstractColorElement):
    """
    Abstract RGB Element and change binary image using threshold
    """
    def __init__(self, thresh_min, thresh_max, channel='H'):
        super(AbstHLSElement, self).__init__(thresh_min, thresh_max)
        if channel == 'H':
            self._channel = 0
        elif channel == 'L':
            self._channel = 1
        else:
            self._channel = 2

    def _get_color_element(self, img):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        return hls[:,:,self._channel]
        

if __name__ == '__main__':
    """
    test code
    """
    from CameraCalib import CameraCalib
    import glob
    cam_calib = CameraCalib(9, 6)
    src_image_file = glob.glob('../test_images/*.jpg')
    cam_calib.import_calib("calibration_pickle.p")
    col_el = AbstHLSElement(100, 255, 'S')

    for idx, fname in enumerate(src_image_file):
        org_img = mpimg.imread(fname)
        src_img = cam_calib.undist_img_file(fname)
        col_el_img = col_el.cmp_thres(src_img)
        outfile = './colorel/colorel_{}.png'.format(idx)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(org_img)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(col_el_img, cmap='gray')
        ax2.set_title('Color Element', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.savefig(outfile)
        
