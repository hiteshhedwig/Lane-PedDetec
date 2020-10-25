import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
#DEFINING PLOTING
'''
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None 
'''
#f, (ax1, ax2, ax3,ax4) = plt.subplots(4)
#f.tight_layout()

# Read in the saved matrix and distortion coefficient
file= 'camera_cali.p'
with open(file, mode='rb') as f:
  dist_pickle= pickle.load(f)
mtx = dist_pickle["matrix"]
dist = dist_pickle["dist_c"]

# Read in an image
#img = mpimg.imread('straight_lines1.jpg')
cap=cv2.VideoCapture('challenge_video.mp4')
#gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

#cap.set(CV_CAP_PROP_FOURCC, CV_FOURCC('H', '2', '6', '4'))
size=(width, height)
print("size", size)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('interim_data.avi',fourcc, 20, size)


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

 def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint-150])
    rightx_base = np.argmax(histogram[midpoint+100:]) + midpoint+100

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 15
    # Set the width of the windows +/- margin
    margin = 70
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        #cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        #(win_xleft_high,win_y_high),(0,255,0), 2) 
        #cv2.rectangle(out_img,(win_xright_low,win_y_low),
        #(win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

 def fit_polynomial(binary_warped, undistorted):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    img_size= (undistorted.shape[1], undistorted.shape[0])

    # Fit a second order polynomial to each using `np.polyfit`

    #bird_eye= np.copy(bird_eye)


    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    #ax3.plot(left_fitx, ploty, color='yellow')
    #ax3.plot(right_fitx, ploty, color='yellow')
    #return out_img
    ##THIS IS SECOND SUB_BLOCK
    #change bird eye to out_img for binary result
    margin = 100
    #left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')  

    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    #print(warp_zero.shape)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    #print(color_warp.shape) 
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    M_ulta = cv2.getPerspectiveTransform(dst,src)
    newwarp= cv2.warpPerspective(color_warp,M_ulta,img_size,flags=cv2.INTER_LINEAR)
    result1 = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)
    

    
    return result1

 def measure_curvature_real():
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Start by generating our fake example data
    # Make sure to feed in your real data instead in your project!
    ploty, left_fit_cr, right_fit_cr = generate_data(ym_per_pix, xm_per_pix)
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return left_curverad, right_curverad


#PIPELINE TO DO IMAGE PROCESSING
 def pipeline(img, s_thresh=(170, 255), sx_thresh=(50, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    #gray=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    #v_channel= gray[:,:,2]
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h_channel = hls[:,:,0]
    l_channel= hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary

#Wrapedimage
 def warp(img, unit=1):
	img_size= (img.shape[1], img.shape[0])

	#source points
	'''
	src= np.float32(
        [[422,500],
         [830,500],
         [1022,617],
         [307,625]])

	dst= np.float32([
		[0,0],
		[1280,0],
		[1280,720],
		[0,720]])
	'''
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

	if unit==2:
		M_ulta = cv2.getPerspectiveTransform(dst,src)
		warped= cv2.warpPerspective(img,M_ulta,img_size,flags=cv2.INTER_LINEAR)

	if unit==1:
		M = cv2.getPerspectiveTransform(src,dst)
		warped= cv2.warpPerspective(img,M,img_size,flags=cv2.INTER_LINEAR)
		#print(warped.shape)
	return warped

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
 #cv2.imshow('feed',out_img)
 warped_img1= np.dstack((warped_img, warped_img, warped_img))*255
 out.write(warped_img1)
 
 if cv2.waitKey(1) & 0xFF == ord('q'):
 
    break

cap.release()
print("writing down the file")
out.release()
cv2.destroyAllWindows()
 #plt.imshow(out_img)
 #plt.show()
