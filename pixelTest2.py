import board
import neopixel
import time

LED_COUNT = 36
PIN = board.D21
ORDER = neopixel.GRB

pixels = neopixel.NeoPixel(PIN, LED_COUNT, brightness=0.3, auto_write=False)

def xy_to_index(x,y):
    return y * 6 + x

def clear():
    pixels.fill((0,0,0))
    pixels.show()
    
def square_animation():
    """Display a square pattern animation with rainbow colors"""
    clear()
    
    # Define the square perimeter indices
    # Bottom row (right to left): 1, 2, 3, 4, 5, 6
    # Left column (bottom to top): 6, 12, 18, 24, 30, 36
    # Top row (left to right): 36, 35, 34, 33, 32, 31
    # Right column (top to bottom): 31, 25, 19, 13, 7, 1
    
    square_indices = [
        # Bottom row (right to left)
        1, 2, 3, 4, 5, 6,
        # Left column (bottom to top) - exclude corners already included
        12, 18, 24, 30,
        # Top row (left to right) - exclude corners already included
        36, 35, 34, 33, 32, 31,
        # Right column (top to bottom) - exclude corners already included
        25, 19, 13, 7
    ]
    
    # Define rainbow colors
    rainbow_colors = [
        (255, 0, 0),     # Red
        (255, 127, 0),   # Orange
        (255, 255, 0),   # Yellow
        (0, 255, 0),     # Green
        (0, 0, 255),     # Blue
        (75, 0, 130),    # Indigo
        (143, 0, 255)    # Violet
    ]
    
    # First animation: Moving rainbow around the square
    for offset in range(len(square_indices) * 2):  # Run through twice
        clear()
        for i, idx in enumerate(square_indices):
            color_idx = (i + offset) % len(rainbow_colors)
            # LED indices in your array are 1-indexed, but Python is 0-indexed
            # So we subtract 1 from the LED index
            pixels[idx - 1] = rainbow_colors[color_idx]
        pixels.show()
        time.sleep(0.1)

def flick_animation():
    """Display a chaotic explosion animation with rapid color flashes that grows over time"""
    clear()
    
    # Define explosion colors
    explosion_colors = [
        (255, 0, 0),      # Red
        (255, 127, 0),    # Orange
        (255, 255, 0),    # Yellow
        (255, 69, 0),     # Red-Orange
        (255, 165, 0)     # Darker Orange
    ]
    
    # Number of flashes in the explosion
    num_flashes = 8
    
    # Create a chaotic explosion effect that grows
    for flash in range(num_flashes):
        clear()
        
        # For first 4 flashes, we'll create a growing effect
        growth_stage = flash
        if growth_stage > 3:
            growth_stage = 3  # Cap at 3 for full display
        
        # Calculate which LEDs to include based on growth stage
        active_leds = []
        
        # Generate the base pattern
        for i in range(36):
            row = i // 6
            col = i % 6
            
            # Skip LEDs based on position for asymmetric effect
            # Also apply growth pattern for first 4 flashes
            if flash < 4:
                # Growth stages:
                # 0: Just the center 2x2
                # 1: Inner 4x4 area
                # 2: Almost full matrix with corners missing
                # 3: Full matrix with random holes
                
                # Center coordinates
                center_row = 2.5
                center_col = 2.5
                
                # Calculate distance from center
                distance = max(abs(row - center_row), abs(col - center_col))
                
                # Only include LEDs within the current growth radius
                if growth_stage == 0 and distance > 1.0:
                    continue
                elif growth_stage == 1 and distance > 1.5:
                    continue
                elif growth_stage == 2 and distance > 2.5:
                    continue
            
            # Add chaos by skipping some LEDs
            if flash % 4 == 0:
                if i % 7 == 0 or (i > 20 and i % 3 == 0):
                    continue
            elif flash % 4 == 1:
                if (i % 7 == 1) or (i + i // 6) % 5 == 2:
                    continue
            elif flash % 4 == 2:
                if i % 13 == 0:
                    continue
            else:
                if i % 9 == 0 or i % 11 == 0:
                    continue
            
            active_leds.append(i)
        
        # Assign colors to the active LEDs
        for i in active_leds:
            # Use different color selection methods for different flashes
            if flash < 3:
                # More red and orange early in explosion
                color_index = flash % 3
            elif flash < 6:
                # More yellow in the middle
                color_index = 2 + (i % 3) % len(explosion_colors)
            else:
                # Mix of colors at the end
                color_index = (i + flash) % len(explosion_colors)
                
            pixels[i] = explosion_colors[color_index]
        
        pixels.show()
        
        # Vary the flash duration
        # Shorter duration as explosion progresses
        flash_duration = 0.12 - (flash * 0.01)
        time.sleep(flash_duration)
    
    # Final bright flash
    bright_leds = []
    for i in range(36):
        # Skip a few random LEDs for chaotic look
        if i % 17 == 0 or i % 13 == 0:
            continue
        bright_leds.append(i)
    
    for i in bright_leds:
        # Bright yellow-white flash
        pixels[i] = (255, 255, 150)
    
    pixels.show()
    time.sleep(0.15)
    
    # Quick fade out
    for brightness in [0.7, 0.4, 0.1]:
        for i in bright_leds:
            color = pixels[i]
            pixels[i] = (int(color[0] * brightness), int(color[1] * brightness), int(color[2] * brightness))
        pixels.show()
        time.sleep(0.05)
    
    clear()

def circle_animation():
    """Display a spiral animation that expands from center with green gradient"""
    clear()
    
    # Define spiral path from center outward
    # Starting with the center LEDs (using 0-indexed positions)
    # The pattern spirals clockwise from the center
    spiral_path = [
        # Center (2x2 center area)
        
        15, 16, 
        22, 21,
        
        # First ring around center (moving clockwise from top-left of center)
        20, 14, 8, 9,
        10, 11, 17, 23,
        29, 28, 27, 26,
        
        # Outer ring (moving clockwise from top-left)
        25, 19, 13, 7, 1,
        2, 3, 4, 5, 6,
        12, 18, 24, 30, 36,
        35, 34, 33, 32, 31
    ]
    
    # Define the color gradient - from deep green to light green
    # Starting color: Deep green
    start_color = (0, 100, 0)  # Deep green
    # Ending color: Light green
    end_color = (150, 255, 150)  # Light green
    
    # Light up LEDs one by one in spiral pattern
    for i, led_index in enumerate(spiral_path):
        # Calculate color for this position in the spiral
        # We'll interpolate between start_color and end_color based on position
        progress = i / len(spiral_path)  # 0.0 to 1.0
        
        # Linear interpolation between colors
        r = int(start_color[0] + (end_color[0] - start_color[0]) * progress)
        g = int(start_color[1] + (end_color[1] - start_color[1]) * progress)
        b = int(start_color[2] + (end_color[2] - start_color[2]) * progress)
        
        # Set the LED color
        pixels[led_index -1] = (r, g, b)
        
        # Show the updated pattern
        pixels.show()
        
        # Delay - slightly faster for the outer rings to create acceleration effect
        #if i < 4:  # Center
        #    delay = 0.15
        #elif i < 18:  # Middle ring
        #    delay = 0.1
        #else:  # Outer ring
        #    delay = 0.05
            
        delay = 0.15 - 0.004 * i
        
        time.sleep(delay)
    
    # Hold the final spiral pattern
    time.sleep(0.5)
    
    # Optional: Add a pulsing effect after the spiral is complete
    for pulse in range(3):  # Pulse 3 times
        # Pulse down (dim the lights)
        for brightness in range(100, 30, -5):  # 100% to 30%
            scale = brightness / 100.0
            for i, led_index in enumerate(spiral_path):
                progress = i / len(spiral_path)
                r = int((start_color[0] + (end_color[0] - start_color[0]) * progress) * scale)
                g = int((start_color[1] + (end_color[1] - start_color[1]) * progress) * scale)
                b = int((start_color[2] + (end_color[2] - start_color[2]) * progress) * scale)
                pixels[led_index - 1] = (r, g, b)
            pixels.show()
            time.sleep(0.02)
        
        # Pulse up (brighten the lights)
        for brightness in range(30, 101, 5):  # 30% to 100%
            scale = brightness / 100.0
            for i, led_index in enumerate(spiral_path):
                progress = i / len(spiral_path)
                r = int((start_color[0] + (end_color[0] - start_color[0]) * progress) * scale)
                g = int((start_color[1] + (end_color[1] - start_color[1]) * progress) * scale)
                b = int((start_color[2] + (end_color[2] - start_color[2]) * progress) * scale)
                pixels[led_index - 1] = (r, g, b)
            pixels.show()
            time.sleep(0.02)
    
    # Fade out
    for brightness in range(100, -1, -5):  # 100% to 0%
        scale = brightness / 100.0
        for i, led_index in enumerate(spiral_path):
            progress = i / len(spiral_path)
            r = int((start_color[0] + (end_color[0] - start_color[0]) * progress) * scale)
            g = int((start_color[1] + (end_color[1] - start_color[1]) * progress) * scale)
            b = int((start_color[2] + (end_color[2] - start_color[2]) * progress) * scale)
            pixels[led_index - 1] = (r, g, b)
        pixels.show()
        time.sleep(0.02)
    
    clear()


def triangle_animation():
    """Display a right triangle that rotates and bounces around the LED matrix"""
    clear()
    
    # Define right triangle shapes in different positions and orientations
    # Each triangle is a list of LED indices (0-indexed)
    triangles = [
        [1, 2, 7],
        [2, 8, 9],
        [10, 16, 15],
        [17, 11, 10],
        [18, 17, 23],
        [29, 28, 22],
        [28, 34, 33],
        [27, 21, 20],
        [25, 20, 19],
        [21, 20, 14],
        [16, 15, 10],
        [17, 11, 10],
        [12, 6, 5]
        
        
    ]
    
    # Define colors for the triangle
    colors = [
        (0, 0, 255),    # Blue
        (0, 127, 255),  # Light blue
        (127, 0, 255),  # Purple
        (255, 0, 127)   # Pink
    ]
    
    # Number of animation cycles
    num_cycles = 2
    
    # Bounce and rotate the triangle
    for cycle in range(num_cycles):
        for t_idx, triangle in enumerate(triangles):
            clear()
            
            # Current color for this triangle
            color = colors[(cycle + t_idx) % len(colors)]
            
            # Light up the triangle
            for led in triangle:
                pixels[led - 1] = color
            pixels.show()
            
            # Add a subtle pulsing effect while this triangle is displayed
            for brightness in [0.8, 0.9, 1.0, 0.9, 0.8]:
                for led in triangle:
                    r, g, b = color
                    pixels[led - 1] = (int(r * brightness), int(g * brightness), int(b * brightness))
                pixels.show()
                time.sleep(0.05)
            
            # Add motion blur effect during transition
            next_triangle = triangles[(t_idx + 1) % len(triangles)]
            
            # Fade out current triangle while fading in next triangle
            steps = 5
            for step in range(steps):
                # Calculate blend factor
                blend = step / (steps - 1)  # 0.0 to 1.0
                
                # Clear for this transition frame
                clear()
                
                # Blend current triangle (fading out)
                for led in triangle:
                    r, g, b = color
                    # Reduce brightness based on blend factor
                    brightness = 1.0 - blend
                    if brightness > 0:
                        pixels[led - 1] = (int(r * brightness), int(g * brightness), int(b * brightness))
                
                # Blend next triangle (fading in)
                next_color = colors[(cycle + t_idx + 1) % len(colors)]
                for led in next_triangle:
                    r, g, b = next_color
                    # Increase brightness based on blend factor
                    brightness = blend
                    if brightness > 0:
                        # If this LED is already lit in the current triangle, blend the colors
                        if led in triangle:
                            curr_r, curr_g, curr_b = pixels[led]
                            pixels[led - 1] = (
                                max(curr_r, int(r * brightness)),
                                max(curr_g, int(g * brightness)),
                                max(curr_b, int(b * brightness))
                            )
                        else:
                            pixels[led - 1] = (int(r * brightness), int(g * brightness), int(b * brightness))
                
                pixels.show()
                time.sleep(0.04)

    
    # Fade out
    for brightness in [0.8, 0.6, 0.4, 0.2, 0]:
        # Get current state of all pixels
        current_state = [pixels[i] for i in range(36)]
        
        # Apply brightness reduction
        for i, color in enumerate(current_state):
            if color != (0, 0, 0):  # Skip already off pixels
                r, g, b = color
                pixels[i] = (int(r * brightness), int(g * brightness), int(b * brightness))
        
        pixels.show()
        time.sleep(0.05)
    
    clear()


square_animation()
flick_animation()
circle_animation()
triangle_animation()