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


last_reset_time = time.time()
background = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

while True:
    
    frame = picam2.capture_array()
    
    #converting to grayscale
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.GaussianBlur(frame_gray, (RADIUS, RADIUS), 0)
    
    #subtracting background and normalizing
    frame_gray = cv.subtract(frame_gray, background)
    frame_gray = cv.normalize(frame_gray, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    
    #thresholding
    _, thresholded = cv.threshold(frame_gray, 200, 255, cv.THRESH_BINARY)
    
    #find contours
    contours, _ = cv.findContours(thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    min_area = 10
    max_area = 50   
    valid_contours = [c for c in contours if min_area <= cv.contourArea(c) <= max_area]
    
    if valid_contours:
        
        brightest_contour = max(valid_contours, key=lambda c: cv.minEnclosingCircle(c)[1])
        
        #coords
        (x,y), _ = cv.minEnclosingCircle(brightest_contour)
        maxLoc = (int(x), int(y))
        
        #drawing the contour
        frame = cv.circle(frame, maxLoc, 5, color, -1)
        
        if 'prev_maxLoc' in locals():
            mask = cv.line(mask, prev_maxLoc, maxLoc, color, 2)
            mask_gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
            contours, _ = cv.findContours(mask_gray, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            
            fill_mask = np.zeros_like(mask)
            for contour in contours:
                if cv.contourArea(contour) > 10:
                    is_closed = cv.isContourConvex(contour)
                    if is_closed:
                            cv.drawContours(fill_mask, [contour], -1, fill_color, thickness=cv.FILLED)
            combined_mask = cv.add(mask, fill_mask)
    else:
        maxLoc = None #no valid object found
        
    img = cv.add(frame, mask)
    
    cv.imshow("img", resize_image(img))
    cv.imshow("Combined_Mask", resize_image(combined_mask))
    
    current_time = time.time()
    if current_time - last_reset_time >= 5:
        mask = np.zeros_like(frame)
        last_reset_time = current_time
    
    prev_maxLoc = maxLoc if maxLoc else prev_maxLoc
    
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    
cv.destroyAllWindows()