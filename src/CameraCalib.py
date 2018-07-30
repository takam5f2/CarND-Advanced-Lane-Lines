"""
Camera Calibration Function Class
"""
import sys, os
import numpy as np
import glob
import pickle
import cv2
import matplotlib.pyplot as plt

class CameraCalib(object):
    """
    CameraCalibration Function
    """
    def __init__(self, nx, ny):
        """
        argument nx and ny are the number of corner
        nx is the number of row, ny is the number of column
        """
        # public
        self.mtx = None
        self.dist = None
        # private
        self._objpoints = list()
        self._imgpoints = list()
        self._nx = nx
        self._ny = ny

    def _project_img(self, img):
        """
        project image to calibration
        """
        # prepare object point data.
        objp = np.zeros((self._ny*self._nx,3), np.float32)
        objp[:,:2] = np.mgrid[0:self._nx, 0:self._ny].T.reshape(-1, 2)
        # gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Fine the chessboard corners
        corner_num = (self._nx, self._ny)
        ret, corners = cv2.findChessboardCorners(gray, corner_num, None)

        # if corners find as expected, object points
        if ret == True:
            self._objpoints.append(objp)
            self._imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, corner_num, corners, ret)
            
        return img
    

    def undist_img(self, img):
        dst = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return dst

    # private functions
    def project_img_file(self, file_name):
        img = cv2.imread(file_name)
        return self._project_img(img)

    def calibrate(self):
        img_size = (img.shape[1], img.shape[0])
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(self._objpoints, self._imgpoints,
                                                                     img_size, None, None)
    def undist_img_file(self, file_name):
        img = cv2.imread(file_name)
        return self.undist_img(img)
    
    def import_calib(self, file_name):
        dist_pickle = pickle.load( open( file_name, "rb" ) )
        self.mtx = dist_pickle["mtx"]
        self.dist = dist_pickle["dist"]
    
    def export_calib(self, file_name):
        """
        export calib file using pickle
        """
        dist_pickle = {}
        dist_pickle["mtx"] = self.mtx
        dist_pickle["dist"] = self.dist
        pickle.dump(dist_pickle, open(file_name, 'wb'))

'''
test code
'''        
if __name__ == '__main__':
    import matplotlib.image as mpimg
    cam_calib = CameraCalib(9,6)
    images = glob.glob('../camera_cal/calibration*.jpg')
    # print(images)

    for idx, fname in enumerate(images):
        img = cam_calib.project_img_file(fname)
        write_name = './calib_corner/corners_found{}.jpg'.format(idx)
        cv2.imwrite(write_name, img)

    cam_calib.calibrate()
    cam_calib.export_calib("calibration_pickle.p")
    images.extend(glob.glob('../test_images/*.jpg'))

    for idx, fname in enumerate(images):
        org_img = mpimg.imread(fname)
        img = cam_calib.undist_img_file(fname)
        outfile = './calib/post_calib_comparisno_{}.png'.format(idx)
        f, ax = plt.subplots(1, 2, figsize=(30, 12))
        f.tight_layout()

        ax[0].imshow(org_img)
        ax[0].set_title('Original', fontsize=30)
        ax[1].imshow(img)
        ax[1].set_title('Undistorted', fontsize=30)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.0)
        plt.savefig(outfile)


    
