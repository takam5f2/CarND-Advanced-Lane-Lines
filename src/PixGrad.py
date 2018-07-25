"""
Get Gradient of pixel in image.
"""
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpmg
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
    def __init__(self, sobel_kernel=3):
        self._sobel_kernel = sobel_kernel

    @abstractmethod
    def _get_grad(self, img):
        pass

    def cmp_grad(self, img, thresh_min=0, thresh_max=1):
        grad = self._get_grad(img)
        binary_output = np.zeros_like(grad)
        binary_output[(grad >= thresh_min) & (grad <= thresh_max)] = 1
        return binary_output
        

class Sobel(SobelBase):
    def __init__(self, sobel_kernel=3, orient='x'):
        super(Sobel, self).__init__(sobel_kernel)
        self._orient = orient
    def _get_grad(self, img):
        sobel = get_sobel(img, self._orient, self._sobel_kernel)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        return scaled_sobel
    # def cmp_grad is prepared.

class SobelGratitude(SobelBase):
    def __init__(self, sobel_kernel=3):
        super(SobelGratitude, self).__init__(sobel_kernel)

    def _get_grad(self, img):
        sobelx = get_sobel(img, 'x', self._sobel_kernel)
        sobely = get_sobel(img, 'y', self._sobel_kernel)
        power_sobelx = np.power(sobelx, 2)
        power_sobely = np.power(sobely, 2)
        abs_sobel = np.sqrt((power_sobelx + power_sobely))
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        return scaled_sobel
    # def cmp_grad is prepared.

class SobelDirection(SobelBase):
    def __init__(self, sobel_kernel=3):
        super(SobelDirection, self).__init__(sobel_kernel)

    def _get_grad(self, img):
        sobelx = get_sobel(img, 'x', self._sobel_kernel)
        sobely = get_sobel(img, 'y', self._sobel_kernel)
        return np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    # def cmp_grad is prepared.

    
if __name__ == '__main__':
    """
    test code
    """
    # sobel = Sobel(3, 'x')
    # sobel = SobelGratitude(5)
    sobel = SobelDirection(5)
    images = glob.glob('./calib/*.jpg')
    img = cv2.imread(images[0])

    binary = sobel.cmp_grad(img, 0.6, 1.3)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(binary, cmap='gray')
    ax2.set_title('Thresholded Gradient', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig('test.jpg')
