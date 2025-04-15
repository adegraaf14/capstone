import cv2 as cv
from picamera2 import Picamera2
import numpy as np
import time
from datetime import datetime

RADIUS = 31
#display resizing
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600

#image resize so that it fits in the screen properly
def resize_image(image, width=DISPLAY_WIDTH, height = DISPLAY_HEIGHT):
        h, w = image.shape[:2]
        scale = min(width / w, height / h)
        return cv.resize(image, (int(w * scale), int(h * scale)))
    
def generate_filename():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return "mask_%s.png" %timestamp


mask = None
color = (255,255,255) #white
fill_color = (0, 0, 255) #red
debug_color = (255, 0, 0) #blue
picam2 = Picamera2()
picam2.start()


first_frame = picam2.capture_array()
if first_frame is None:
    print("Failed to capture initial frame.")
    exit()
    
mask = np.zeros_like(first_frame)
combined_mask = np.zeros_like(mask)
reference_frame = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
last_reset_time = time.time()
prev_maxLoc = None


while True:
    
    frame = picam2.capture_array()
    #new_reference_frame = cv.cvtColor(picam2.capture_array(), cv.COLOR_BGR2GRAY)
    #reference_frame = new_reference_frame
    cv.imshow("ref frame", resize_image(reference_frame))
    
    #converting to grayscale
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #frame_gray = cv.GaussianBlur(frame_gray, (RADIUS, RADIUS), 0)
    
    #reference difference frame
    frame_diff = cv.absdiff(reference_frame, frame_gray)
    _, thresholded = cv.threshold(frame_diff, 150, 255, cv.THRESH_BINARY) #adjust threshold as needed

    (_, maxVal, _, maxLoc) = cv.minMaxLoc(thresholded)
    
    if prev_maxLoc is not None:
        frame = cv.line(frame, prev_maxLoc, maxLoc, color, 2)
        mask = cv.line(mask, prev_maxLoc, maxLoc, color, 2)
        
    prev_maxLoc = maxLoc
    
    
    #find contours
    contours, _ = cv.findContours(thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    max_brightness = -1
    brightest_contour = None
    maxLoc = None
    
    for contour in contours:
        mask_temp = np.zeros_like(frame_gray)
        cv.drawContours(mask_temp, [contour], -1, debug_color, thickness=cv.FILLED)
              
    img = cv.add(frame, mask)
    cv.imshow("Thresholded Differences", resize_image(thresholded))
    cv.imshow("frame diff", resize_image(frame_diff))
    cv.imshow("mask", resize_image(mask))
    cv.imshow("img", resize_image(img))
    
    
    
    current_time = time.time()
    if current_time - last_reset_time >= 5:
        new_reference_frame = cv.cvtColor(picam2.capture_array(), cv.COLOR_BGR2GRAY)
        exclusion_radius = 15
        
        if prev_maxLoc is not None:
            mask_exclusion = np.zeros_like(new_reference_frame)
            cv.circle(mask_exclusion, prev_maxLoc, exclusion_radius, (255), -1)
            new_reference_frame[mask_exclusion == 255] = 0
            
            
        reference_frame = new_reference_frame
        mask = np.zeros_like(frame)
        last_reset_time = current_time
    
    
    
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    
cv.destroyAllWindows()

