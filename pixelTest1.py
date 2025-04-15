import board
import neopixel
import time

pixels = neopixel.NeoPixel(board.D21, 36, brightness=0.5, auto_write=False)

pixels.fill((255, 0, 0))
pixels.show()

time.sleep(5)

pixels.fill((0,0,0))
pixels.show()
