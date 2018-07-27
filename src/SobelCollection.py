"""
Get Gradient of pixel in image.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from abc import ABCMeta, abstractmethod

# common function
def get_sobel(img, orient='x', sobel_kernel=3):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    return sobel

class SobelBase(metaclass=ABCMeta):
    """
    Abstract class for Sobel, SobelGratitude, SobelDirection
    """
    def __init__(self, sobel_kernel=3, thresh_min=0, thresh_max=1):
        self._sobel_kernel = sobel_kernel
        self._thresh_min = thresh_min
        self._thresh_max = thresh_max

    @abstractmethod
    def _get_grad(self, img):
        pass

    def cmp_thres(self, img):
        grad = self._get_grad(img)
        binary_output = np.zeros_like(grad)
        binary_output[(grad >= self._thresh_min) & (grad <= self._thresh_max)] = 1
        return binary_output
        

class Sobel(SobelBase):
    def __init__(self, sobel_kernel=3, thresh_min=0, thresh_max=1, orient='x'):
        super(Sobel, self).__init__(sobel_kernel, thresh_min, thresh_max)
        self._orient = orient
    def _get_grad(self, img):
        sobel = get_sobel(img, self._orient, self._sobel_kernel)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        return scaled_sobel
    # def cmp_thres is prepared.

class SobelGratitude(SobelBase):
    def __init__(self, sobel_kernel=3, thresh_min=0, thresh_max=1):
        super(SobelGratitude, self).__init__(sobel_kernel, thresh_min, thresh_max)

    def _get_grad(self, img):
        sobelx = get_sobel(img, 'x', self._sobel_kernel)
        sobely = get_sobel(img, 'y', self._sobel_kernel)
        power_sobelx = np.power(sobelx, 2)
        power_sobely = np.power(sobely, 2)
        abs_sobel = np.sqrt((power_sobelx + power_sobely))
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        return scaled_sobel
    # def cmp_thres is prepared.

class SobelDirection(SobelBase):
    def __init__(self, sobel_kernel=3, thresh_min=0, thresh_max=1):
        super(SobelDirection, self).__init__(sobel_kernel, thresh_min, thresh_max)


    def _get_grad(self, img):
        sobelx = get_sobel(img, 'x', self._sobel_kernel)
        sobely = get_sobel(img, 'y', self._sobel_kernel)
        return np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    # def cmp_thres is prepared.

if __name__ == '__main__':
    from CameraCalib import CameraCalib
    import glob
    cam_calib = CameraCalib(9, 6)
    src_image_file = glob.glob('../test_images/*.jpg')
    cam_calib.import_calib("calibration_pickle.p")
    sobel = Sobel(3, 30, 100, 'x')

    for idx, fname in enumerate(src_image_file):
        org_img = mpimg.imread(fname)
        src_img = cam_calib.undist_img_file(fname)
        sobel_img = sobel.cmp_thres(src_img)
        outfile = './sobel/sobel_{}.png'.format(idx)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(org_img)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(sobel_img, cmap='gray')
        ax2.set_title('Thresholded Gradient', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.savefig(outfile)
        
