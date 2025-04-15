import io
import time
import picamera2

def outputs():
    stream = io.BytesIO()
    tup=()
    for i in range(40):
        stream.seek(0)
        img = Image.open(stream)
        np_image=np.array(img)
        tup = tup+(np_img,)
        stream.seek(0)
        stream.truncate()
        final_image=np.dstack(tup)
        
with picamera2.PiCamera2() as camera:
    camera.resolution = (640, 480)
    camera.framerate = 80
    time.sleep(2)
    start = time.time()
    camera.capture_sequence(outputs(), 'jpeg', use_video_port=True)
    finish = time.time()
    print('Captured 40 images')