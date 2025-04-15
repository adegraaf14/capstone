import numpy as np
import cv2 as cv
import argparse
import time

#display resizing
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600

#image resize so that it fits in the screen properly
def resize_image(image, width=DISPLAY_WIDTH, height = DISPLAY_HEIGHT):
        h, w = image.shape[:2]
        scale = min(width / w, height / h)
        return cv.resize(image, (int(w * scale), int(h * scale)))
    
#argument parse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image file")
ap.add_argument("-r", "--radius", type = int, help = "radius of Gaussian blue; must be odd")
args = vars(ap.parse_args())

cap = cv.VideoCapture(args["image"])

color = np.random.randint(0, 255, (100, 3))

ret, first_frame = cap.read()
if not ret:
    print("Failed to capture video")
    cap.release()
    exit()
    
mask = np.zeros_like(first_frame)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print('No frames grabbed.')
        break
    
    
    #convert current frame to grayscale and apply Gaussian blur
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.GaussianBlur(frame_gray, (args["radius"], args["radius"]), 0)

    
    #finding the brightest point in the image
    (_,maxVal,_,maxLoc) = cv.minMaxLoc(frame_gray)
    
    #draw line
    if 'prev_maxLoc' in locals():
        mask = cv.line(mask, prev_maxLoc, maxLoc, color[0].tolist(), 2)
    
    frame = cv.circle(frame, maxLoc, 5, color[0].tolist(), -1)
    
    img = cv.add(frame, mask)
    
    #cv.imshow('frame', resize_image(img))
    cv.imshow('frame', resize_image(mask))
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    
    #update previous
    prev_maxLoc = maxLoc
    
cap.release()
cv.destroyAllWindows()
    
    
    
    
    
    
    
    