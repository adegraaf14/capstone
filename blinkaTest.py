import board
import digitalio
import busio
import neopixel

print("Hello world")

pin = digitalio.DigitalInOut(board.D4)
print("Digital IO ok")

i2c= busio.I2C(board.SCL, board.SDA)
print("I2C ok")

pixel_pin = board.D18

num_pixels = 6

ORDER = neopixel.GRB

pixels = neopixel.NeoPixel(
	pixel_pin, num_pixels, brightness=0.2, auto_write-False, pixel_order=ORDER
)

def rainbow_cycle(wait):
	for j in range(255):
		pixel_index = (i * 256 // num_pixels) + j
		pixels[i] = 
