import time
import board
import neopixel

LED_COUNT = 6
LED_PIN = board.D21
ORDER = neopixel.GRB

pixels = neopixel.NeoPixel(LED_PIN, LED_COUNT, brightness=0.6, auto_write=False, pixel_order=ORDER)

def color_wipe(color, wait=0.1):
    for i in range(LED_COUNT):
        pixels[i] = color
        pixels.show()
        time.sleep(wait)
        
color_wipe((255,0,0))
time.sleep(1)
color_wipe((0,255,0))
time.sleep(1)
color_wipe((0,0,255))
time.sleep(1)
           
pixels.fill((0,0,0))
pixels.show()