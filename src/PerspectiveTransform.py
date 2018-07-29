"""
Execute perspective transformation and get warped data
"""
import cv2
import numpy as np


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

    def _warp_img(self, img, src_range, dst_range):
        img_size = (img.shape[1], img.shape[0])
        M = cv2.getPerspectiveTransform(src_range, dst_range)
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

        return warped

    def warp_img(self, img):
        """
        warp image with using warp perspectve
        """
        return self._warp_img(img, self._src_range, self._dst_range)

    def warp_img_const_dst(self, img, dst_offset=[100,100]):
        
        self._dst_range = np.float32([[dst_offset[0], dst_offset[1]],
                                      [dst_offset[0], img.shape[0]-dst_offset[1]],
                                      [img.shape[1]-dst_offset[0], img.shape[0]-dst_offset[1]],
                                      [img.shape[1]-dst_offset[0], dst_offset[1]]])
        
        return self._warp_img(img, self._src_range, self._dst_range)

    def reverse_img(self, img):
        return self._warp_img(img, self._dst_range, self._src_range)

if __name__ == '__main__':

    import glob
    from CameraCalib import CameraCalib
    from EdgeDetection import EdgeDetection
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from ColorElementCollection import AbstGray, AbstRGBElement, AbstHLSElement
    from SobelCollection import Sobel, SobelGratitude, SobelDirection
    import copy
    
    cam_calib = CameraCalib(9, 6)
    src_image_file = glob.glob('../test_images/*.jpg')
    cam_calib.import_calib("calibration_pickle.p")

    edge_detection = EdgeDetection()
    # add function
    edge_detection.add_func(Sobel(5, 30, 100, 'x'))
    edge_detection.add_func(AbstHLSElement(160, 255, 'S'))

    src_range = np.float32([[550, 470], [220, 680], [1150, 680], [750, 470]])
    dst_range = np.float32([[200, 0], [200, 700], [1100, 700], [1100,100]])
    
    ptransform = PerspectiveTransform(src_range, dst_range)

    for idx, fname in enumerate(src_image_file):
    
        org_img = mpimg.imread(fname)
        calibed_img = cam_calib.undist_img_file(fname)
        edet_img = edge_detection.execute(calibed_img)
        interest_img = copy.deepcopy(org_img)
        trans_img = ptransform.warp_img(edet_img)

        src_rect = np.array(src_range, np.int32)
        src_rect = src_rect.reshape((-1, 1, 2))
        cv2.polylines(interest_img, [src_rect], True, (255, 255, 255))

        outfile = './trans/trans_{}.png'.format(idx)
        f, ax = plt.subplots(2, 2, figsize=(30, 12))
        f.tight_layout()

        ax[0,0].imshow(org_img)
        ax[0,0].set_title('Edge Detection Image', fontsize=30)
        ax[0,1].imshow(edet_img, cmap='gray')
        ax[0,1].set_title('Edge Detection Image', fontsize=30)
        
        ax[1,0].imshow(interest_img, cmap='gray')
        ax[1,0].set_title('Edge Detection Image', fontsize=30)
        ax[1,1].imshow(trans_img, cmap='gray')
        ax[1,1].set_title('Transformation', fontsize=30)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.savefig(outfile)
        
