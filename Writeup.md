# **Uadcity Self Driving Car Nanodegree Program: Finding Lane Lines on the Road**
## This project requires us to detect lanes lines in a video file using basic computer vision. In the video, the lines on the road show us where the lanes to serve as  constant reference for the vehicle to self steer. This project aims at automatically detecting lane lines using an algorithm. 

The tools used are:
1.	Python3
2.	OpenCV
3.	Numpy
4.	Jupyter Notebook

The following steps were used to develop and test the above algorithm:
•	Import Initial Images
•	Create Helper Functions
•	Load Test Images
•	Convert Images to Grayscale
•	Apply Gaussian Smoothing
•	Apply Canny Transform
•	Apply Region of interest
•	Apply Hough Transform
•	Merge Original Image with Lines
•	Video Test

Approach:
First the necessary libraries were imported:
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

To test whether necessary library were successfully imported, the image test_images/solidWhiteRight.jpg was read then outputted. 

To do intermediate calculations a set of Helper methods were implemented. They will essentially like perp, moving average and segment intersection. 
Another helped function that was implemented was to convert the images have the lanes to grayscale. This is very important especially since we will be using Canny Edge Detector of OpenCV. What we are basically doing is collapsing the 3 channels of RGB into a single channel with a pixel range of [0,255] using this code:
return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
Then the Canny edge detection function was used to detect the edges of line using
return cv2.Canny(img, low_threshold, high_threshold)

Then the Gaussian blur function was  implemented which  helps in getting rid of noisy parts of the image which makes the next steps more reliable.

Then the Region of Interest method was implemented. The region of interest is defines the region where the the car will be focusing on.. Everything outside the ROI will be blacked out to zero

Next the function to define the most dominant line from a set of detected lines from image and video files was implemented
The function implemented was 
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):

Then the algorithm was first tried on images, Five images files as given below was fed into the algorithm. They were kept in the “/test_images” folder inside the working directory
solidWhiteRight
solidYellowCurve
solidYellowCurve2
solidYellowLeft
whiteCarLaneSwitch

Associated images with the detected lanes were outputted in “/test_images_out” inside the working directory

Once the lanes were detected , the final test was done two videos files below kept in “/video” folder inside the working folder
solidWhiteRight.mp4
solidYellowLeft.mp4

The function used was fl_image which modifies the images of a video clip by replacing then images using the driver image processing function “process_image”

The annotated video output was outputted in working folder as white.mp4 and yellow.mp4




