#imports
import numpy as np
import argparse
import cv2 as cv

DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600

#image resize so that it fits in the screen properly
def resize_image(image, width=DISPLAY_WIDTH, height = DISPLAY_HEIGHT):
        h, w = image.shape[:2]
        scale = min(width / w, height / h)
        return cv2.resize(image, (int(w * scale), int(h * scale)))


#argument parse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image file")
ap.add_argument("-r", "--radius", type = int, help = "radius of Gaussian blue; must be odd")
args = vars(ap.parse_args())

cap = cv.VideoCapture(args["image"])
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
if not ret:
    print("Failed to capture video")
    cap.release()
    exit()
    
#convert first frame to grayscale and apply Gaussian blur
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
old_gray = cv.GaussianBlur(old_gray, (args["radius"], args["radius"]), 0) #might need to change args.radius

#Finding the brightest point
(_,_,_, maxLoc) = cv.minMaxLoc(old_gray)
p0 = np.array([[maxLoc]], dtype=np.float32) #might need to figure out what kind of variable p0 is compared to before

mask = np.zeros_like(old_frame)


while True:
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    
    #convert current frame to grayscale and apply Gaussian blur
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.GaussianBlur(frame_gray, (args["radius"], args["radius"]), 0) #might need to change args.radius
    
    #calculate optical flow to track the brightest point
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    #update if successfully tracked
    if p1 is not None and st[0][0] == 1:
        new_loc = p1[0][0]
        old_loc = p0[0][0]
        
        #draw line and circle for tracking
        mask = cv.line(mask, (int(new_loc[0]), int(new_loc[1])), (int(old_loc[0]), int(old_loc[1])), color[0].tolist(), 2)
        frame = cv.circle(frame, (int(new_loc[0]), int(new_loc[1])), 5, color[0].list(), -1)
        
        #combine
        img = cv.add(frame, mask)
        
        cv.imshow('Tracked brightest point', resize_image(img))
        
        
        #update for next iteration
        old_gray = frame_gray.copy()
        p0 = p1
    else:
        print("Point lost, recalculating the brightest point in the current frame.")
        (_,_,_, maxLoc) = cv.minMaxLoc(frame_gray)
        p0 = np.array([[maxLoc]], dtype = np.float32)
        
        
        
cap.release()
cv.destroyAllWindows()
    
                                          
                                          
                                          
                                          
                                          
                                          
                                          
                                          
                                          
                                          
                                          
                                          





