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
    timestamp = datetime.now().strftime("%Y%m%d_%H:%M")
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
    
last_reset_time = time.time()

while True:
    
    frame = picam2.capture_array()
    cv.imshow("Frame", resize_image(frame))
    
    current_time = time.time()
    if current_time - last_reset_time >= 5:
        #filename = generate_filename()
        #cv.imwrite(filename, frame)
        last_reset_time = time.time()
    
    
    
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    
cv.destroyAllWindows()


#Current version being used 1/21/25





























































































