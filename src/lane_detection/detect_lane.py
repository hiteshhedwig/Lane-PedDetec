import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
from warped_image import warp
from image_pipeline import pipeline
from fit_line import fit_polynomial
from lane_pixels import find_lane_pixels

# Read in the saved matrix and distortion coefficient

file= 'camera_cali.p'
with open(file, mode='rb') as f:
  dist_pickle= pickle.load(f)
mtx = dist_pickle["matrix"]
dist = dist_pickle["dist_c"]

# Read in an image
#img = mpimg.imread('straight_lines1.jpg')
cap=cv2.VideoCapture('harder_challenge_video.mp4')
#gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

#cap.set(CV_CAP_PROP_FOURCC, CV_FOURCC('H', '2', '6', '4'))
size=(width, height)
print("size", size)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('testing4.avi',fourcc, 20, size)


src= np.float32(
        [[712,466],
         [1035,662],
         [307,662],         
         [569, 469]])
dst= np.float32([
		[842,0],
		[842,664],
		[400,664],
		[400,0]],)

#VIDEO
while(1):
 _,img=cap.read()    
 #print("yehh",img.shape[0])
 """
 def measure_curvature_real():
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Start by generating our fake example data
    ploty, left_fit_cr, right_fit_cr = generate_data(ym_per_pix, xm_per_pix)
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return left_curverad, right_curverad
    """

 undistorted=cv2.undistort(img, mtx, dist, None, mtx) 
 result = pipeline(undistorted)
 ysize=img.shape[0]
 xsize=img.shape[1]
 region_select= np.copy(img)
 region_of_interest_vertices = [
    (0, 720),
    (1280/2, 720/2),
    (1280, 720),
    ]   
 roi= np.array([region_of_interest_vertices],np.int32)
 mask=np.zeros_like(result)
 match_mask_color= 255
 cv2.fillPoly(mask, roi, match_mask_color)
 masked_image= cv2.bitwise_and(result, mask)
 
 warped_img= warp(result)
 #bird_eye= warp(img)
 try:
    out_img = fit_polynomial(warped_img, undistorted)
    print('Writing down result')
    out.write(out_img)

 except TypeError:
     out.write(undistorted)
 #orig_img=warp(bird_eye,2)
 #cv2.imshow('feed',out_img)
 #warped_img1= np.dstack((warped_img, warped_img, warped_img))*255

 
 if cv2.waitKey(1) & 0xFF == ord('q'):
 
    break

cap.release()
print("writing down the file")
out.release()
cv2.destroyAllWindows()
 #plt.imshow(out_img)
 #plt.show()
