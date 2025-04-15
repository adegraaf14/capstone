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


while True:
    
    frame = picam2.capture_array()
    
    #converting to grayscale
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.GaussianBlur(frame_gray, (RADIUS, RADIUS), 0)
    
    #reference difference frame
    frame_diff = cv.absdiff(reference_frame, frame_gray)
    _, thresholded = cv.threshold(frame_diff, 30, 255, cv.THRESH_BINARY) #adjust threshold as needed
    
    #find contours
    contours, _ = cv.findContours(thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    max_brightness = -1
    brightest_contour = None
    maxLoc = None
    
    for contour in contours:
        mask_temp = np.zeros_like(frame_gray)
        cv.drawContours(mask_temp, [contour], -1, 255, thickness=cv.FILLED)
        
        
        #mean intensity
        mean_val = cv.mean(frame_gray, mask=mask_temp)[0]
        if mean_val > max_brightness:
            max_brightness = mean_val
            brightest_contour = contour
    
    if brightest_contour is not None:
        (x, y), radius = cv.minEnclosingCircle(brightest_contour)
        maxLoc = (int(x), int(y))
        frame = cv.circle(frame, maxLoc, 5, color, -1)
        mask = cv.line(mask, prev_maxLoc, maxLoc, color, 2) if 'prev_maxLoc' in locals() else mask
        
        mask_gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        #combined_mask = cv.add(mask, thresholded)
        
        prev_maxLoc = maxLoc
        
    cv.imshow("frame_diff", resize_image(frame_diff))
    cv.imshow("Thresholded Differences", resize_image(thresholded))
    cv.imshow("Combined Mask", resize_image(combined_mask))
    
    
    
    current_time = time.time()
    if current_time - last_reset_time >= 5:
        reference_frame = cv.cvtColor(picam2.capture_array(), cv.COLOR_BGR2GRAY)
        reference_frame = cv.GaussianBlur(reference_frame, (RADIUS, RADIUS), 0)
        mask = np.zeros_like(frame)
        last_reset_time = current_time
    
    prev_maxLoc = maxLoc if maxLoc else prev_maxLoc
    
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    
cv.destroyAllWindows()
