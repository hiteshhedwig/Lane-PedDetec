import cv2
import numpy as np
import config

#Wrapedimage
def warp(img, unit=1):
	img_size= (img.shape[1], img.shape[0])


	if unit==2:
		M_ulta = cv2.getPerspectiveTransform(config.dst,config.src)
		warped= cv2.warpPerspective(img,M_ulta,img_size,flags=cv2.INTER_LINEAR)

	if unit==1:
		M = cv2.getPerspectiveTransform(config.src,config.dst)
		warped= cv2.warpPerspective(img,M,img_size,flags=cv2.INTER_LINEAR)
		
	return warped
