"""
Execute perspective transformation and get warped data
"""
import cv2
import numpy as np

WARP_INIT   = 0
WARP_MANUAL = 1
WARP_CONST  = 2
WARP_REV    = 3

class PerspectiveTransform(object):
    """
    This class includes function of perspecitve trasformation
    and reversing of this function.
    """
    def __init__(self, src_range, dst_range):
        """
        argument: src_range = [[np.float*2]*4]
                  dst_range = [[np.float*2]*4]
        """
        self._src_range = src_range
        self._dst_range = dst_range
        self._state = WARP_INIT

    def _warp_img(self, img, next_state):
        img_size = (img.shape[1], img.shape[0])
        M = cv2.getPerspectiveTransform(self._src_range, self._dst_range)
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
        self._state = next_state
        return warped

    def warp_img(self, img):
        """
        warp image with using warp perspectve
        """
        return self._warp_img(img, WARP_MANUAL)

    def warp_img_const_dst(self, img, dst_offset=[100,100]):
        
        self._dst_range = np.float32([[dst_offset[0], dst_offset[1]],
                                      [dst_offset[0], img.shape[0]-dst_offset[1]],
                                      [img.shape[1]-dst_offset[0], img.shape[0]-dst_offset[1]],
                                      [img.shape[1]-dst_offset[0], dst_offset[1]]])
        
        return self._warp_img(img, WARP_CONST)

    def reverse_img(self, img):
        tmp = self._src_range
        self._src_range = self._dst_range
        self._dst_range = tmp

        return _warp_img(self, img, WARP_REV)


if __name__ == '__main__':

    import glob
    from CameraCalib import CameraCalib
    from EdgeDetection import EdgeDetection
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from ColorElementCollection import AbstGray, AbstRGBElement, AbstHLSElement
    from SobelCollection import Sobel, SobelGratitude, SobelDirection

    
    cam_calib = CameraCalib(9, 6)
    src_image_file = glob.glob('../test_images/*.jpg')
    cam_calib.import_calib("calibration_pickle.p")

    edge_detection = EdgeDetection()
    # add function
    edge_detection.add_func(SobelGratitude(5, 70, 150))
    edge_detection.add_func(AbstHLSElement(150, 250, 'S'))

    src_range = np.float32([[450, 450], [150, 700], [1150, 700], [850,450]])
    dst_range = np.float32([[100, 100], [100, 700], [1200, 700], [1200,100]])
    
    ptransform = PerspectiveTransform(src_range, dst_range)

    for idx, fname in enumerate(src_image_file):
    
        org_img = mpimg.imread(fname)
        calibed_img = cam_calib.undist_img_file(fname)
        edet_img = edge_detection.execute(calibed_img)
        trans_img = ptransform.warp_img_const_dst(edet_img, (0,0))

        outfile = './trans/trans_{}.png'.format(idx)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(edet_img, cmap='gray')
        ax1.set_title('Edge Detection Image', fontsize=50)
        ax2.imshow(trans_img, cmap='gray')
        ax2.set_title('Transformation', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.savefig(outfile)
        
