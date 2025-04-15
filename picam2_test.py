from picamera2 import Picamera2, Preview
from libcamera import Transform
import time
import subprocess
import os



picam2 = Picamera2()

#picam2.start_preview(Preview.QTGL, x=100, y=200, width=800, height=600, transform=Transform(hflip=1))
video_config = picam2.create_video_configuration()
picam2.configure(video_config)
picam2.start()
time.sleep(3)

output_file = "ir_vid2.h264"

picam2.start_and_record_video(output_file, duration=8)

#time.sleep(8)

picam2.stop_recording()
picam2.stop()
print("Video recording completed")


h264_file = "ir_vid2.h264"
mp4_file =  "ir_vid2.mp4"

subprocess.run(["ffmpeg", "-framerate","30", "-i", h264_file, "-c", "copy", mp4_file])
print("Video conversion complete")