#coding=utf-8
import numpy as np
import cv2


def AbsSobelThresh(gray,orient='x',thresh_min=20,thresh_max=100):

    if orient=='x':
        abs_sobel=np.absolute(cv2.Sobel(gray,cv2.CV_64F,1,0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    scaled_sobel=np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output=np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel>=thresh_min)&(scaled_sobel<=thresh_max)]=1

    return binary_output


def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
	"""
	Return the magnitude of the gradient
	for a given sobel kernel size and threshold values
	"""
	# Convert to grayscale

	# Take both Sobel x and y gradients
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	# Calculate the gradient magnitude
	gradmag = np.sqrt(sobelx**2 + sobely**2)
	# Rescale to 8 bit
	scale_factor = np.max(gradmag)/255
	gradmag = (gradmag/scale_factor).astype(np.uint8)
	# Create a binary image of ones where threshold is met, zeros otherwise
	binary_output = np.zeros_like(gradmag)
	binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

	# Return the binary image
	return binary_output


def dir_threshold(gray, sobel_kernel=3, thresh=(0.7, np.pi/2)):
	"""
	Return the direction of the gradient
	for a given sobel kernel size and threshold values
	"""
	# Convert to grayscale

	# Calculate the x and y gradients
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	# Take the absolute value of the gradient direction,
	# apply a threshold, and create a binary image result
	absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
	binary_output =  np.zeros_like(absgraddir)
	binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

	# Return the binary image
	return binary_output


def hls_thresh(img, thresh=(80, 255)):
	"""
	Convert RGB to HLS and threshold to binary image using S channel
	"""
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	s_channel = hls[:,:,2]
	binary_output = np.zeros_like(s_channel)
	binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
	return binary_output


def combined_thresh(img):
	hls_bin = hls_thresh(img, thresh=(50, 255))

	img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	abs_bin = AbsSobelThresh(img, orient='x', thresh_min=50, thresh_max=255)
	mag_bin = mag_thresh(img, sobel_kernel=3, mag_thresh=(50, 255))
	dir_bin = dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))

	combined = np.zeros_like(dir_bin)
	combined[(abs_bin == 1 | ((mag_bin == 1) & (dir_bin == 1))) | hls_bin == 1] = 1

	scale_factor = np.max(combined) / 255
	combined = (combined / scale_factor).astype(np.uint8)
	return combined # DEBUG



def PerspectiveTransform(img):

    src = np.float32(
		[[200, 720],
		 [1100, 720],
		 [595, 450],
		 [685, 450]])
    dst = np.float32(
		[[300, 720],
		 [980, 720],
		 [300, 0],
		 [980, 0]])
    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)
    img_size=(img.shape[1],img.shape[0])
    warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)
    #unwarped = cv2.warpPerspective(warped, m_inv, (warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG
    return warped, m, m_inv

# if __name__=="__main__":
# 	img=cv2.imread("drive.jpg")
# 	img_thresh=combined_thresh(img)
#
# 	warped,unwarped,m,m_inv=PerspectiveTransform(img_thresh)
# 	cv2.imshow("img",img)
# 	cv2.imshow("img_thresh",img_thresh)
#
# 	print img_thresh.shape
# 	print warped.shape
# 	histogram = np.sum(warped[int(warped.shape[0] / 2):, :], axis=0)
# 	print histogram.shape
# 	cv2.imshow("warped",warped)
# 	print warped
# 	cv2.imshow("unwarped",unwarped)
#
# 	cv2.waitKey()
#   cv2.destroyAllWindows()


