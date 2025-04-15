import cv2 as cv
from picamera2 import Picamera2
import numpy as np
import time
from datetime import datetime
import board
import neopixel


RADIUS = 31
#display resizing
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600
AREA_THRESHOLD = 500



#pixel stuff=======================================================================
LED_COUNT = 36
PIN = board.D21
ORDER = neopixel.GRB

#global vars
mask = None
color = (255, 255, 255) #white
fill_color = (0, 0, 255) #red
debug_color = (255, 0, 0) #blue
picam2 = None
pixels = None
prev_gray = None
prev_maxLoc = None
fill_layer = None
last_reset_time = None

def initialize():
    global picam2, pixels, mask, fill_layer, prev_gray, prev_maxLoc, last_reset_time
    
    picam2 = Picamera2()
    picam2.start()

    pixels = neopixel.NeoPixel(PIN, LED_COUNT, brightness=0.3, auto_write=False)
    
    first_frame = picam2.capture_array()
    if first_frame is None:
        print("Failed to capture initial frame.")
        return False
        
    mask = np.zeros_like(first_frame)
    fill_layer = np.zeros_like(first_frame)
    
    prev_frame = picam2.capture_array()
    prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
    last_reset_time = time.time()
    prev_maxLoc = None
    
    return True

def cleanup():
    global picam2
    
    if picam2 is not None:
        picam2.stop()
        
    clear()
    cv.destroyAllWindows()

def xy_to_index(x,y):
    return y * 6 + x

def clear():
    pixels.fill((0,0,0))
    pixels.show()
    
def square():
    clear()
    square_indices = [xy_to_index(x, 0) for x in range(6)] + \
                     [xy_to_index(x, 5) for x in range(6)] + \
                     [xy_to_index(0, y) for y in range(6)] + \
                     [xy_to_index(5, y) for y in range(6)]
    for i in set(square_indices):
        pixels[i] = (255,0,0)
    pixels.show()
    
def smiley():
    clear()
    smiley_indices = [
        xy_to_index(1, 1), xy_to_index(4, 1),
        xy_to_index(1, 4), xy_to_index(2, 5), xy_to_index(3, 5), xy_to_index(4, 4)
    ]
    for i in smiley_indices:
        pixels[i] = (255, 255, 0)
    pixels.show()
    
def triangle():
    clear()
    triangle_indices = [
        xy_to_index(2, 0), xy_to_index(3, 0),
        xy_to_index(1, 1), xy_to_index(4, 1),
        xy_to_index(0, 2), xy_to_index(5, 2),
        xy_to_index(1, 3), xy_to_index(4, 3),
        xy_to_index(2, 4), xy_to_index(3, 4),
        xy_to_index(2, 5), xy_to_index(3, 5)
    ]
    for i in triangle_indices:
        pixels[i] = (0, 0, 255)
    pixels.show()
    
def growing_diamond():
    clear()
    
    diamond_layers = [
        [xy_to_index(2, 2), xy_to_index(3, 2),
         xy_to_index(2, 3), xy_to_index(3, 3)], #center
        
        [xy_to_index(1, 2), xy_to_index(4, 2),
         xy_to_index(1, 3), xy_to_index(4, 3),
         xy_to_index(2, 1), xy_to_index(3, 1),
         xy_to_index(2, 4), xy_to_index(3, 4)], #2nd layer
        
        [xy_to_index(0, 2), xy_to_index(5, 2),
         xy_to_index(0, 3), xy_to_index(5, 3),
         xy_to_index(1, 1), xy_to_index(4, 1),
         xy_to_index(1, 4), xy_to_index(4, 4)], #3rd layer
        
        [xy_to_index(0, 1), xy_to_index(5, 1),
         xy_to_index(0, 4), xy_to_index(5, 4),
         xy_to_index(1, 0), xy_to_index(4, 0),
         xy_to_index(1, 5), xy_to_index(4, 5)], #4th layer
        
        [xy_to_index(0, 0), xy_to_index(5, 0),
         xy_to_index(0, 5), xy_to_index(5, 5)]
        
        
    ]
    
    for layer in diamond_layers:
        for i in layer:
            pixels[i] = (75, 0, 130)
        pixels.show()
        time.sleep(0.3)
        
    time.sleep(1)
    clear()
    
#end  of led pixel stuff===================================================================

#image resize so that it fits in the screen properly
def resize_image(image, width=DISPLAY_WIDTH, height = DISPLAY_HEIGHT):
        h, w = image.shape[:2]
        scale = min(width / w, height / h)
        return cv.resize(image, (int(w * scale), int(h * scale)))
    
def generate_filename():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return "mask_%s.png" %timestamp

def detect_shapes(fill_layer):
    fill_gray = cv.cvtColor(fill_layer, cv.COLOR_BGR2GRAY)
    contours, _ = cv.findContours(fill_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv.contourArea(contour)
        if area > AREA_THRESHOLD:
            epsilon = 0.02 * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, True)
            vertices = len(approx)
            
            #classification
            if vertices == 3:
                print("Triangle detected")
                triangle()
                time.sleep(5)
                clear()
            elif vertices == 4:
                x, y, w, h = cv.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 0.9 <= aspect_ratio <= 1.1:
                    print("Square detected")
                    square()
                    time.sleep(5)
                    clear()
                else:
                    print("Rectangle detected")
                    growing_diamond()
                    time.sleep(5)
                    clear()
            else:
                perimeter = cv.arcLength(contour, True)
                circularity = 4 * np.pi * (area / (perimeter ** 2))
                if 0.8 <= circularity <= 1.2:
                    print("Circle detected")
                    smiley()
                    time.sleep(5)
                    clear()
                else:
                    print("Unknown shape detected")
                    
            
            return



def motion_detected():
    SENSOR_PIN = 17
    
    try:
        return GPIO.input(SENSOR_PIN) == GPIO.HIGH
    except:
        return false

def process_frame():
    
    global mask, fill_layer, prev_gray, prev_maxLoc, last_reset_time
    
    frame = picam2.capture_array()
    
    
    #converting to grayscale
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    
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
    
    detect_shapes(fill_layer)
        
    output_frame = cv.add(frame, mask)
    output_frame = cv.add(output_frame, fill_layer)
    
    
    
    cv.imshow("Fill layer", resize_image(fill_layer))
    cv.imshow("Motion regions", resize_image(motion_regions))
    cv.imshow("Tracked output", resize_image(output_frame))
    
    prev_gray = frame_gray.copy()
    
    
    current_time = time.time()
    if current_time - last_reset_time >= 5:
        mask = np.zeros_like(frame)
        fill_layer = np.zeros_like(frame)
        last_reset_time = current_time
        prev_maxLoc = None
    
    
    
    k = cv.waitKey(30) & 0xff
    return k
    
def main__loop():
    initialize()
    
    try:
        while True:
            key = process_frame()
            if key == 27:
                break
    finally:
        cleanup()
        
if __name__ == "__main__":
    main_loop()


#Current version being used 1/21/25



