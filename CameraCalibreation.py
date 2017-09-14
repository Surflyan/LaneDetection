#coding=utf-8
import numpy as np
import cv2
import glob
# Read all images in a file folder using glob
img_file=glob.glob('camera_cal/calibration*.jpg')

#Array to store object points and imge points frome all the images
obj_points=[]
img_points=[]

def calib():
    objp=np.zeros((6*9,3),np.float32)
    objp[:,:2]=np.mgrid[0:9,0:6].T.reshape(-1,2)

    for curr_img in img_file:
        img=cv2.imread(curr_img)
        gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        ret,corners=cv2.findChessboardCorners(gray_img,(9,6),None)

        #If corners are found ,add object points , image points
        if ret==True:
            img_points.append(corners)
            obj_points.append(objp)
        else:
            continue
    ret ,mtx,dist,rvers,tvecs=cv2.calibrateCamera(obj_points,img_points,gray_img.shape[::-1],None,None)
    return mtx,dist

def undistort(img,mtx,dist):
    return cv2.undistort(img,mtx,dist,None,mtx)
