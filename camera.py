from picamzero import Camera
from time import sleep

cam = Camera()
cam.start_preview()
sleep(2)
cam.take_photo("~/Desktop/image.jpg")
cam.stop_preview()