import cv2 as cv
from picamera2 import Picamera2
import numpy as np
import time
from datetime import datetime

RADIUS = 31
#display resizing
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600
CONSISTENCY_FRAMES = 5
DISTANCE_THRESHOLD = 20

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
fill_layer = np.zeros_like(first_frame)
combined_mask = np.zeros_like(mask)
prev_frame = picam2.capture_array()
prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
last_reset_time = time.time()
prev_maxLoc = None
consistency_counter = 0


while True:
    
    frame = picam2.capture_array()
    #new_reference_frame = cv.cvtColor(picam2.capture_array(), cv.COLOR_BGR2GRAY)
    #reference_frame = new_reference_frame
    #cv.imshow("ref frame", resize_image(reference_frame))
    
    #converting to grayscale
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #frame_gray = cv.GaussianBlur(frame_gray, (RADIUS, RADIUS), 0)
    
    #reference difference frame
    frame_diff = cv.absdiff(prev_gray, frame_gray)
    _, motion_mask = cv.threshold(frame_diff, 50, 255, cv.THRESH_BINARY)
    
    motion_regions = cv.bitwise_and(frame_gray, frame_gray, mask=motion_mask)
    _, maxVal, _, maxLoc = cv.minMaxLoc(motion_regions)
    
    if maxVal > 200:
        if prev_maxLoc is not None:
            mask = cv.line(mask, prev_maxLoc, maxLoc, color, 2)
        prev_maxLoc = maxLoc
        
    mask_gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    
    contours, _ = cv.findContours(mask_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        epsilon = 0.01 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)
        
        
        if cv.contourArea(contour) > 300:
            cv.drawContours(fill_layer, [contour], -1, fill_color, -1)
           
    fill_layer = cv.subtract(fill_layer, mask)
        
    output_frame = cv.add(frame, mask)
    output_frame = cv.add(output_frame, fill_layer)
    
    
    
    cv.imshow("Fill layer", resize_image(fill_layer))
    #cv.imshow("Motion mask", resize_image(motion_mask))
    #cv.imshow("Motion regions", resize_image(motion_regions))
    cv.imshow("Tracked output", resize_image(output_frame))
    
    prev_gray = frame_gray.copy()
    
    
    current_time = time.time()
    if current_time - last_reset_time >= 5:
        #filename1 = generate_filename()
        #filename2 = generate_filename()
        #cv.imwrite(filename1, output_frame)
        #cv.imwrite(filename2, fill_layer)
        mask = np.zeros_like(frame)
        fill_layer = np.zeros_like(frame)
        last_reset_time = current_time
        prev_maxLoc = None
    
    
    
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    
cv.destroyAllWindows()


#Current version being used 1/21/25
