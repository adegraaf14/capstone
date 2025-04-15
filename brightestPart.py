#packages
import numpy as np
import argparse
import cv2

DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600

def resize_image(image, width=DISPLAY_WIDTH, height = DISPLAY_HEIGHT):
        h, w = image.shape[:2]
        scale = min(width / w, height / h)
        return cv2.resize(image, (int(w * scale), int(h * scale)))

#argument parse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image file")
ap.add_argument("-r", "--radius", type = int, help = "radius of Gaussian blue; must be odd")
args = vars(ap.parse_args())

#load image and conver to grayscale
image = cv2.imread(args["image"])
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#apply Guassian blur then find brightest region
gray = cv2.GaussianBlur(gray, (args["radius"], args["radius"]), 0)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
image = orig.copy()
cv2.circle(image, maxLoc, args["radius"], (255, 0, 0), 2)

#display the image
cv2.imshow("Robust", resize_image(image))
cv2.waitKey(0)
