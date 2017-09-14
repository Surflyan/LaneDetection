import numpy as np
import cv2
import Thresh
import CameraCalibreation

class Line:
    def __init__(self):
        self.detected=False
        self.window_margin=56

        self.prevx=[]
        self.current_fit=[np.array([False])]

        self.startx=None
        self.endx=None
        self.allx=None
        self.ally=None


    def Smoothing(self,lines,prev_n_lines=3):
        lines=np.squeeze(lines)
        avg_line=np.zeros((720))
        for i, line in enumerate(reversed(lines)):
            if i == prev_n_lines:
                break
            avg_line += line
        avg_line = avg_line/prev_n_lines
        return avg_line


    def LineSearchReset(self,binary_img,left_lane,right_lane):
        histogram=np.sum(binary_img[int(binary_img.shape[0]/2):,:],axis=0)
        out_img=np.dstack((binary_img,binary_img,binary_img))*255
        mid_point=np.int(histogram.shape[0]/2)
        left_x_base=np.argmax(histogram[:mid_point])
        right_x_base=np.argmax(histogram[mid_point:])+mid_point

        num_windows=9

        #set height of windows
        windows_height=np.int(binary_img.shape[0]/num_windows)

        #Identify the x and y positions of all nonzero pixel in the image
        nonzero=binary_img.nonzero()
        nonzeroy=np.array(nonzero[0])
        nonzerox=np.array(nonzero[1])

        # Current positions to be updated for each window
        current_left_x=left_x_base
        current_right_x= right_x_base

        # Set minimum number of pixels found to recenter window
        min_num_pixel=50

        # Creat empty lists to receive left and right lan pixel indices
        win_left_lane=[]
        win_right_lane=[]

        windows_margin=left_lane.window_margin

        # Step through the windows one by one
        for window in range (num_windows):

            #Identify window boundaries in x and y (and left and right)
            win_y_low=binary_img.shape[0]-(window+1)*windows_height
            win_y_high=binary_img.shape[0]-window*windows_height
            win_left_x_min=current_left_x-windows_margin
            win_left_x_max=current_left_x+windows_margin
            win_right_x_min=current_right_x-windows_margin
            win_right_x_max=current_right_x+windows_margin

            # Draw the windows on the visualization image
            #cv2.rectangle(out_img, (win_left_x_min, win_y_low), (win_left_x_max, win_y_high), (0, 255, 0), 2)
            #cv2.rectangle(out_img, (win_right_x_min, win_y_low), (win_right_x_max, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window
            left_window_inds=((nonzeroy>=win_y_low)&(nonzeroy<=win_y_high)&(nonzerox>=win_left_x_min)&(nonzerox<=win_left_x_max)).nonzero()[0]
            right_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_right_x_min) & (nonzerox <= win_right_x_max)).nonzero()[0]

            # Append these indices to the lists
            win_left_lane.append(left_window_inds)
            win_right_lane.append(right_window_inds)

            # if found > min pixels ,recenter next window on their mean position
            if len(left_window_inds)>min_num_pixel:
                current_left_x=np.int(np.mean(nonzerox[left_window_inds]))
            if len(right_window_inds)>min_num_pixel:
                current_right_x=np.int(np.mean(nonzerox[right_window_inds]))
        # Concatenate the arrays of indices
        win_left_lane=np.concatenate(win_left_lane)
        win_right_lane=np.concatenate(win_right_lane)

        # Extract left and right line pixel positions
        left_x=nonzerox[win_left_lane]
        left_y=nonzeroy[win_left_lane]
        right_x=nonzerox[win_right_lane]
        right_y=nonzeroy[win_right_lane]

        #out_img[left_y,left_x]=[255,0,0]
        #out_img[right_y,right_x]=[0,0,255]

        # Fit a second order polynomial to each
        left_fit=np.polyfit(left_y,left_x,2)

        right_fit=np.polyfit(right_y,right_x,2)

        left_lane.current_fit=left_fit
        right_lane.current_fit=right_fit

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_img.shape[0] - 1, binary_img.shape[0])
        # ax^2 + bx + c
        left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_plotx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        left_lane.prevx.append(left_plotx)
        right_lane.prevx.append(right_plotx)

        if len(left_lane.prevx)>10:
            left_avg_lane = self.Smoothing(left_lane.prevx,10)
            left_avg_fit=np.polyfit(ploty,left_avg_lane,2)
            left_fit_plotx=left_avg_fit[0]*ploty**2+left_avg_fit[1]*ploty+left_avg_fit[2]
            left_lane.current_fit=left_avg_fit
            left_lane.allx,left_lane.ally=left_fit_plotx,ploty

        else:
            left_lane.current_fit=left_fit
            left_lane.allx,left_lane.ally=left_plotx,ploty

        if len(right_lane.prevx) > 10:
            right_avg_line = self.Smoothing(right_lane.prevx, 10)
            right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
            right_fit_plotx = right_avg_fit[0] * ploty ** 2 + right_avg_fit[1] * ploty + right_avg_fit[2]
            right_lane.current_fit = right_avg_fit
            right_lane.allx, right_lane.ally = right_fit_plotx, ploty
        else:
            right_lane.current_fit = right_fit
            right_lane.allx, right_lane.ally = right_plotx, ploty

        left_lane.startx,right_lane.startx=left_lane.allx[len(left_lane.allx)-1],right_lane.allx[len(right_lane.allx)-1]
        left_lane.endx,right_lane.endx=left_lane.allx[0],right_lane.allx[0]

        # Set detected = True for both lane
        left_lane.detected,right_lane.detected=True,True

        return out_img


    def LineSearchTracking(self,binary_img,left_lane,right_lane):
        print "tracking"
        out_img=np.dstack((binary_img,binary_img,binary_img))*255

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero=binary_img.nonzero()
        nonzeroy=np.array(nonzero[0])
        nonzerox=np.array(nonzero[1])

        # Get margin of windows from line class. Adjust this number
        window_margin=left_lane.window_margin

        left_lane_fit=left_lane.current_fit
        right_lane_fit=right_lane.current_fit
        leftx_mid=left_lane_fit[0]*nonzeroy**2 + left_lane_fit[1]*nonzeroy+left_lane_fit[2]
        rightx_mid=right_lane_fit[0]*nonzeroy**2+right_lane_fit[1]*nonzeroy+right_lane_fit[2]
        leftx_min=leftx_mid-window_margin
        leftx_max=leftx_mid+window_margin
        rightx_min=rightx_mid-window_margin
        rightx_max=rightx_mid+window_margin
        try :
            left_inds=((nonzerox>=leftx_min)&(nonzerox<=leftx_max)).nonzero()[0]
            right_inds=((nonzerox>=rightx_min)&(nonzerox<=rightx_max)).nonzero()[0]

            leftx,lefty=nonzerox[left_inds],nonzeroy[left_inds]
            rightx,righty=nonzerox[right_inds],nonzeroy[right_inds]

            #out_img[lefty,leftx]=[255,0,0]
            #out_img[righty,rightx]=[0,0,255]

        # Fit a second order ploynomial to each
            left_fit=np.polyfit(lefty,leftx,2)
            right_fit=np.polyfit(righty,rightx,2)


            

            ploty=np.linspace(0,binary_img.shape[0]-1,binary_img.shape[0])
            left_plotx=left_fit[0]*ploty**2+left_fit[1]*ploty+left_fit[2]
            right_plotx=right_fit[0]*ploty**2 + right_fit[1]*ploty+right_fit[2]

            leftx_avg=np.average(left_plotx)
            right_avg=np.average(right_plotx)

            left_lane.prevx.append(left_plotx)
            right_lane.prevx.append(right_plotx)

            if len(left_lane.prevx) > 10:  # take at least 10 previously detected lane lines for reliable average
                left_avg_line = self.Smoothing(left_lane.prevx, 10)
                left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
                left_fit_plotx = left_avg_fit[0] * ploty ** 2 + left_avg_fit[1] * ploty + left_avg_fit[2]
                left_lane.current_fit = left_avg_fit
                left_lane.allx, left_lane.ally = left_fit_plotx, ploty
            else:
                left_lane.current_fit = left_fit
                left_lane.allx, left_lane.ally = left_plotx, ploty

            if len(right_lane.prevx) > 10:  # take at least 10 previously detected lane lines for reliable average
                right_avg_line = self.Smoothing(right_lane.prevx, 10)
                right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
                right_fit_plotx = right_avg_fit[0] * ploty ** 2 + right_avg_fit[1] * ploty + right_avg_fit[2]
                right_lane.current_fit = right_avg_fit
                right_lane.allx, right_lane.ally = right_fit_plotx, ploty
            else:
                right_lane.current_fit = right_fit
                right_lane.allx, right_lane.ally = right_plotx, ploty

            stddev = np.std(right_lane.allx - left_lane.allx)

            if (stddev > 80):
                left_lane.detected = False

            left_lane.startx, right_lane.startx = left_lane.allx[len(left_lane.allx) - 1], right_lane.allx[
                len(right_lane.allx) - 1]
            left_lane.endx, right_lane.endx = left_lane.allx[0], right_lane.allx[0]

        except:
            print "fit_error"
        return out_img

    def GetLaneLinesImg(self,binary_img,left_line,right_line):
        if left_line.detected==False:
            return self.LineSearchReset(binary_img,left_line,right_line)
        else:
            return self.LineSearchTracking(binary_img,left_line,right_line)

    def IllustrateDrivingLane(self,img,left_lane,right_lane,lane_color=(255,0,0),road_color=(0,255,0)):
        window_img=np.zeros_like(img)

        window_margin=left_lane.window_margin
        left_plotx,right_plotx=left_lane.allx,right_lane.allx
        ploty=left_lane.ally

        left_line_window1=np.array([np.transpose(np.vstack([left_plotx-window_margin/5,ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_plotx + window_margin / 5, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_plotx - window_margin / 5, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_plotx + window_margin / 5, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), lane_color)
        cv2.fillPoly(window_img, np.int_([right_line_pts]), lane_color)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_plotx + window_margin / 5, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_plotx - window_margin / 5, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([pts]), road_color)
        result = cv2.addWeighted(img, 1, window_img, 0.3, 0)

        return result,window_img

# if __name__=="__main__":
#     left_line=Line()
#     right_line=Line()
#
#     img=cv2.imread("drive.jpg")
#    # img2 = cv2.imread("test2.jpg")
#
#     mtx, dist = CameraCalibreation.calib()
#
#     undist_img=CameraCalibreation.undistort(img,mtx,dist)
#     img_thresh=Thresh.combined_thresh(undist_img)
#     warped,m,m_inv=Thresh.PerspectiveTransform(img_thresh)
#
#   #  undist_img2 = CameraCalibreation.undistort(img2, mtx, dist)
#   #  img_thresh2 = Thresh.combined_thresh(undist_img2)
#    # warped2, m2, m_inv2 = Thresh.PerspectiveTransform(img_thresh2)
#     #cv2.imshow("img",img)
#    # cv2.imshow("img_thresh",img_thresh)
#
#     #print img_thresh.shape
#     print warped.shape
#    # histogram = np.sum(warped[int(warped.shape[0] / 2):, :], axis=0)
#    # print histogram.shape
#   #  cv2.imshow("warped",warped2)
#     print warped
#    #cv2.imshow("unwarped",unwarped2)
#     test=Line()
#
#
#     searching_img=test.GetLaneLinesImg(warped,left_line,right_line)
#    # searching_img2 = test.GetLaneLinesImg(warped2, left_line, right_line)
#     cv2.imshow("serching_img",searching_img)
#   #  cv2.imshow("serching_img2", searching_img2)
#
#     w_result,w_color_resule=test.IllustrateDrivingLane(searching_img,left_line,right_line)
#     cv2.imshow("w_result",w_result)
#     cv2.imshow("w_color_result",w_color_resule)
#
#     color_result=cv2.warpPerspective(w_color_resule,m_inv,(img.shape[1],img.shape[0]))
#     cv2.imshow("color_result",color_result)
#
#     result=cv2.addWeighted(undist_img,1,color_result,0.3,0)
#     cv2.imshow("result",result)
#
#
#     cv2.waitKey()
#     cv2.destroyAllWindows()















