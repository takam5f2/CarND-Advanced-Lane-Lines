import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ColorElementCollection import AbstGray, AbstRGBElement, AbstHLSElement
from SobelCollection import Sobel, SobelGratitude, SobelDirection

class EdgeDetection(object):
    """
    Edge detection pipeline using ColorElementCollection and SobelCollection
    """
    def __init__(self):
        self._pipeline = list()

    def add_func(self, func):
        self._pipeline.append(func)

    def execute(self, img):
        ret_binary = np.zeros_like(img[:,:,0])
        if len(self._pipeline) == 0:
            return 
        for idx, func in enumerate(self._pipeline):
            func_ret = func.cmp_thres(img)
            ret_binary[(ret_binary == 1) | (func_ret == 1)] = 1

        return ret_binary

    
# test code
if __name__ == '__main__':
    import glob
    from CameraCalib import CameraCalib
    
    cam_calib = CameraCalib(9, 6)
    src_image_file = glob.glob('../test_images/*.jpg')
    cam_calib.import_calib("calibration_pickle.p")

    
    edge_detection = EdgeDetection()
    # add function
    edge_detection.add_func(SobelGratitude(5, 90, 150))
    edge_detection.add_func(AbstHLSElement(140, 250, 'S'))

    # execute
    for idx, fname in enumerate(src_image_file):
        org_img = mpimg.imread(fname)
        calibed_img = cam_calib.undist_img_file(fname)
        sobel_img = edge_detection.execute(calibed_img)
        outfile = './edge_det/edge_det_{}.png'.format(idx)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(org_img)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(sobel_img, cmap='gray')
        ax2.set_title('Edge Detection', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.savefig(outfile)
        
