#coding=utf-8
import cv2
import numpy as np

import CameraCalibreation
import Thresh
from line import Line


# def DivideImg(img):
#
#    a, b,c= img.shape
#    divide_line = int(a * 0.5)
#    img1=img[:divide_line,:,:]
#    img2=img[divide_line:,:,:]
#    return img1,img2
#
# def VerticalDivide(img):
#     a,b=img.shape
#     divide_line=int(b*0.5)
#     img3=img[:,:divide_line]
#     img4=img[:,divide_line:]
#     return img3,img4
#
#
# def LineDetect(img,threshold,lower_angle,upper_angle):
#     all_lines=cv2.HoughLines(img,1,np.pi/180,threshold)
#     req_lines = []
#     lower=np.pi*lower_angle/180
#     upper=np.pi*upper_angle/180
#
#
#     if isinstance(all_lines,np.ndarray):
#         lines1 = all_lines[:, 0, :]  # 提取为为二维
#         for rho, theta in lines1[:]:
#             if theta<lower or theta>upper:
#                     a = np.cos(theta)
#                     b = np.sin(theta)
#                     x0 = a * rho
#                     y0 = b * rho
#                     x1 = int(x0 + 1000 * (-b))
#                     y1 = int(y0 + 1000 * (a))
#                     x2 = int(x0 - 1000 * (-b))
#                     y2 = int(y0 - 1000 * (a))
#                     req_lines.append([x1, y1, x2, y2, rho, theta])
#
#     return req_lines
#
# def FindShortestPoint(x1,y1,x2,y2,width):
#
#         if x2 == x1 :
#             ret=abs(width*0.5-x1)
#         else:
#             k=(y2-y1)/(x2-x1)
#             b=y1-k*x1
#             if k!=0:
#                dis=-b/k
#                ret=abs(width*0.5-dis)
#             else:
#                 ret= abs(width*0.5-x1)
#         return ret
#
#
#
# def FindBestLane(req_lines,width):
#     max_num=float('inf')
#     best=[]
#     for x1,y1,x2,y2 ,rho,theta in req_lines:
#         a=FindShortestPoint(x1,y1,x2,y2,width)
#
#         if a<max_num :
#             max_num=a
#             best.append([x1, y1, x2, y2, rho, theta])
#     if len(best)==0:
#         return best
#     else:
#         return [best[-1]]
#
#
# def DrawLine(img,req_lines):
#     if len(req_lines)!=0:
#         for x1, y1, x2, y2, rho, theta in req_lines:
#             cv2.line(img, (x1, y1), (x2, y2),(0,255,0) ,2)
#
#     return img
#
#
# def Pipline(frame,mtx,dist):
#
#     img1, img2 = DivideImg(frame)
#     a,b=img2.shape[:2]
#     undist_img=CameraCalibreation.undistort(img2,mtx,dist)
#
#     thresh_img = Thresh.combined_thresh(undist_img)
#    # cv2.imshow("thresh",thresh_img)
#    # cv2.waitKey()
#    # cv2.destroyAllWindows()
#
#     edge_img = cv2.Canny(thresh_img,100,200)
#     img3, img4 = VerticalDivide(edge_img)
#
#     req_lines3 = LineDetect(img3, 80, 70, 105)
#     req_lines4 = LineDetect(img4, 80, 70, 105)
#
#     best_line3 = FindBestLane(req_lines3, img2.shape[0])
#     best_line4 = FindBestLane(req_lines4, img2.shape[0])
#
#     line_img4 = DrawLine(img2[:,int(b*0.5):], best_line4)
#     line_img3 = DrawLine(img2[:,:int(b*0.5)], best_line3)
#
#     return line_img3 ,line_img4
#
# def main(input_video):
#     mtx, dist = CameraCalibreation.calib()
#     cap=cv2.VideoCapture(input_video)
#     while(cap.isOpened()):
#         ret,frame=cap.read()
#         if ret==True:
#
#             cv2.imshow('InitialVideo',frame)
#             img3,img4=Pipline(frame,mtx,dist)
#
#             cv2.imshow('LineImg4',img4)
#             cv2.imshow('LineImg3',img3)
#             if cv2.waitKey(1)& 0xFF==ord('q'):
#                 break
#         else:
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#
# #if __name__=="__mian__":
#  #   main(sys.argv[1])
# main('video.mp4')
#
#


def Pipline(img,mtx,dist,left_line,right_line,test):
    #img=cv2.resize(img,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)

    undist_img = CameraCalibreation.undistort(img, mtx, dist)
    img_thresh = Thresh.combined_thresh(undist_img)
   # cv2.imshow("thresh_img",img_thresh)
    warped, m, m_inv = Thresh.PerspectiveTransform(img_thresh)
    #cv2.imshow("warped",warped)

    searching_img = test.GetLaneLinesImg(warped, left_line, right_line)
    w_result,w_color_resule=test.IllustrateDrivingLane(searching_img,left_line,right_line)
   # cv2.imshow("w_result",w_result)
   # cv2.imshow("w_color_result",w_color_resule)

    color_result=cv2.warpPerspective(w_color_resule,m_inv,(undist_img.shape[1],undist_img.shape[0]))
    #cv2.imshow("color_result",color_result)

    result=cv2.addWeighted(undist_img,1,color_result,0.3,0)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return result
    #cv2.imshow("result",result)



if __name__=="__main__":
    left_line = Line()
    right_line = Line()
    test=Line()
    mtx, dist = CameraCalibreation.calib()

    cap = cv2.VideoCapture("video.mp4")
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:

            cv2.imshow('InitialVideo', frame)
            result=Pipline(frame,mtx,dist,left_line,right_line,test)


            cv2.imshow('result',result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

