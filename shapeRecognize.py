import cv2 as cv
from picamera2 import Picamera2
import numpy as np
import time
from datetime import datetime
from matplotlib import pyplot as plt


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
while True:
    
    frame = picam2.capture_array()
    #cv2.imshow("frame", image)
    
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.GaussianBlur(frame_gray, (RADIUS, RADIUS), 0)
    
    (_,maxVal,_,maxLoc) = cv.minMaxLoc(frame_gray)
    
    if 'prev_maxLoc' in locals():
        mask = cv.line(mask, prev_maxLoc, maxLoc, color, 2)
        #mask = cv.line(mask, prev_maxLoc, maxLoc, color[0].tolist(), 2)
        mask_gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        contours, _ = cv.findContours(mask_gray, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        
        fill_mask = np.zeros_like(mask)
        
        for contour in contours:
            #cv.drawContours(mask, [contour], -1, debug_color, 2)
            if cv.contourArea(contour) > 10: #ignoring small contours
                is_closed = cv.isContourConvex(contour)
                if is_closed:
                    cv.drawContours(fill_mask, [contour], -1, fill_color, thickness=cv.FILLED)
        combined_mask = cv.add(mask, fill_mask)
            
 
        
    frame = cv.circle(frame, maxLoc, 5, color, -1)
    
    img = cv.add(frame, mask)

    
    #cv.imshow("Mask", resize_image(mask))
    cv.imshow("Combined_Mask", resize_image(combined_mask))
    cv.imshow(fill_mask)
    
    
    #used for resetting the mask every 5 seconds, comment to remove
    current_time = time.time()
    if current_time - last_reset_time >= 5:
        #filename = generate_filename()
        #cv.imwrite(filename, img)
        #cv.fillConvexPoly(resize_image(frame), resize_image(mask), color=(0,255,255))
        #sleep(5)
        mask = np.zeros_like(frame)
        last_reset_time = current_time
    
    prev_maxLoc = maxLoc
    
    
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    
cv.destroyAllWindows()


