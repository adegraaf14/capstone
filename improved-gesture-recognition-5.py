import cv2 as cv
from picamera2 import Picamera2
import numpy as np
import time
from datetime import datetime
import board
import neopixel
from collections import deque
import math

# Constants
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600
AREA_THRESHOLD = 300
MOTION_THRESHOLD = 50
RESET_TIME = 5  # seconds before resetting the drawing
PATH_LENGTH = 64  # number of points to track for gesture recognition
MIN_PATH_POINTS = 20  # minimum points needed to attempt recognition
MIN_GESTURE_SIZE = 100  # minimum bounding box size (pixels) for a gesture to be recognized
CONFIDENCE_THRESHOLD = 40  # minimum confidence score to recognize a gesture (0-100)

# LED configuration
LED_COUNT = 36
PIN = board.D21
ORDER = neopixel.GRB
BRIGHTNESS = 0.3

# Initialize NeoPixel strip
pixels = neopixel.NeoPixel(PIN, LED_COUNT, brightness=BRIGHTNESS, auto_write=False)

# Function to convert 2D coordinates to LED index in 6x6 grid
def xy_to_index(x, y):
    return y * 6 + x

# LED animation functions
def clear():
    pixels.fill((0, 0, 0))
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

# Image processing helper functions
def resize_image(image, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT):
    """Resize image to fit display while maintaining aspect ratio"""
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    return cv.resize(image, (int(w * scale), int(h * scale)))

def generate_filename():
    """Generate a timestamped filename for saving images"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"gesture_{timestamp}.png"

# Gesture recognition functions
def calculate_features(points):
    """Calculate features from a path for gesture recognition"""
    if len(points) < MIN_PATH_POINTS:
        return None
    
    # Simplify the path to reduce noise and minor hand tremors
    # Using Ramer-Douglas-Peucker algorithm to simplify the curve
    epsilon = 5.0  # Tolerance parameter - higher means more simplification
    simplified_points = np.array(points, dtype=np.float32)
    simplified_points = cv.approxPolyDP(simplified_points, epsilon, False)
    simplified_points = [tuple(p[0]) for p in simplified_points]
    
    # If we have too few points after simplification, use original points
    if len(simplified_points) < MIN_PATH_POINTS * 0.5:
        simplified_points = points
    
    # Convert points list to numpy array
    points_array = np.array(simplified_points)
    
    # Find centroid
    centroid = np.mean(points_array, axis=0)
    
    # Calculate distances from centroid
    distances = np.sqrt(np.sum((points_array - centroid) ** 2, axis=1))
    avg_distance = np.mean(distances)
    std_distance = np.std(distances)
    
    # Calculate the total path length
    path_length = 0
    for i in range(1, len(simplified_points)):
        path_length += np.linalg.norm(np.array(simplified_points[i]) - np.array(simplified_points[i-1]))
    
    # Calculate the convex hull
    if len(simplified_points) >= 3:  # Need at least 3 points for convex hull
        hull = cv.convexHull(np.array(simplified_points, dtype=np.float32))
        hull_area = cv.contourArea(hull)
        hull_perimeter = cv.arcLength(hull, True)
    else:
        hull_area = 1  # Avoid division by zero
        hull_perimeter = 0
    
    # Calculate the bounding box
    x, y, w, h = cv.boundingRect(np.array(simplified_points, dtype=np.float32))
    aspect_ratio = float(w) / (h if h > 0 else 1)
    
    # Calculate direction changes with higher tolerance
    direction_changes = 0
    if len(simplified_points) > 2:
        # Use a sliding window approach to smooth out the direction calculation
        window_size = 3
        angles = []
        
        for i in range(window_size, len(simplified_points)):
            # Get points at start and end of window
            start_point = np.array(simplified_points[i-window_size])
            end_point = np.array(simplified_points[i])
            
            # Calculate direction vector
            vector = end_point - start_point
            
            if np.linalg.norm(vector) > 0:
                # Calculate angle in radians
                angle = np.arctan2(vector[1], vector[0])
                angles.append(angle)
        
        # Count significant direction changes (using a higher threshold)
        for i in range(1, len(angles)):
            angle_diff = abs(angles[i] - angles[i-1])
            # Normalize angle difference to be between 0 and pi
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            angle_diff = abs(angle_diff)
            
            # Only count significant direction changes (45 degrees or more)
            if angle_diff > np.pi/4:
                direction_changes += 1
    
    # Line detection - check if the gesture is close to a straight line
    straightness = 0
    if len(simplified_points) >= 2:
        # Get first and last point
        first_point = np.array(simplified_points[0])
        last_point = np.array(simplified_points[-1])
        
        # Calculate the length of the direct line between start and end
        direct_distance = np.linalg.norm(last_point - first_point)
        
        # Calculate straightness as the ratio of direct distance to path length
        # A perfect line would have a ratio of 1.0
        straightness = direct_distance / path_length if path_length > 0 else 0
    
    # Calculate the average speed along the path (if timestamps available)
    # Here we use point density as a proxy
    point_density = len(simplified_points) / path_length if path_length > 0 else 0
    
    # Check vertical direction of flick (looking at first and last point)
    start_end_direction = None
    if len(simplified_points) >= 2:
        start_point = np.array(simplified_points[0])
        end_point = np.array(simplified_points[-1])
        
        # Calculate vertical difference
        y_diff = end_point[1] - start_point[1]
        
        # Determine if motion was mostly down (positive y diff in image coords) or up (negative)
        if abs(y_diff) > 20:  # Only consider significant vertical movement
            start_end_direction = "down" if y_diff > 0 else "up"
    
    # Calculate triangle-specific metrics (if hull has 3-4 vertices after approximation)
    is_triangle_like = False
    if len(simplified_points) >= 3:
        # Approximate the hull to see if it's triangle-like
        hull_approx = cv.approxPolyDP(hull, 0.04 * cv.arcLength(hull, True), True)
        # Check if the approximated hull has 3 or 4 vertices (allowing for some imprecision)
        if len(hull_approx) == 3 or len(hull_approx) == 4:
            is_triangle_like = True
    
    return {
        'std_distance': std_distance / (avg_distance if avg_distance > 0 else 1),
        'compactness': (hull_perimeter ** 2) / (4 * np.pi * hull_area) if hull_area > 0 else 0,
        'aspect_ratio': aspect_ratio,
        'direction_changes': direction_changes,
        'path_length': path_length,
        'boundingbox_area': w * h,
        'point_count': len(simplified_points),
        'straightness': straightness,
        'point_density': point_density,
        'vertical_direction': start_end_direction,
        'is_triangle_like': is_triangle_like
    }

def calculate_confidence_scores(features):
    """Calculate confidence scores for each gesture type"""
    # Initialize confidence scores
    confidence = {
        'circle': 0,
        'square': 0,
        'triangle': 0,
        'flick': 0
    }
    
    # Basic qualification check - if we don't have enough points or area, return zeros
    if (features['point_count'] < MIN_PATH_POINTS or 
        features['boundingbox_area'] < MIN_GESTURE_SIZE or 
        features['path_length'] < MIN_GESTURE_SIZE * 0.5):
        return confidence
    
    # Calculate circle confidence
    # Ideal circle has low std_distance, compactness close to 1, aspect ratio close to 1
    std_distance_score = max(0, 1 - features['std_distance'] * 3)  # Lower is better
    compactness_score = max(0, 1 - abs(features['compactness'] - 1.0) / 0.8)  # Closer to 1 is better
    aspect_ratio_score = max(0, 1 - abs(features['aspect_ratio'] - 1.0) / 0.5)  # Closer to 1 is better
    confidence['circle'] = int((std_distance_score * 0.4 + compactness_score * 0.4 + aspect_ratio_score * 0.2) * 100)
    
    # Calculate square confidence
    # Ideal square has ~4 direction changes, compactness close to 1, aspect ratio close to 1
    direction_score = max(0, 1 - abs(features['direction_changes'] - 4) / 2)  # Closer to 4 is better
    square_compactness = max(0, 1 - abs(features['compactness'] - 1.0) / 0.5)  # Closer to 1 is better
    square_aspect = max(0, 1 - abs(features['aspect_ratio'] - 1.0) / 0.5)  # Near 1 for square
    confidence['square'] = int((direction_score * 0.5 + square_compactness * 0.3 + square_aspect * 0.2) * 100)
    
    # Calculate triangle confidence
    # Ideal triangle has 3 direction changes, compactness ~1.3, is_triangle_like=True
    triangle_direction = max(0, 1 - abs(features['direction_changes'] - 3) / 2)  # Closer to 3 is better
    triangle_compactness = max(0, 1 - abs(features['compactness'] - 1.3) / 0.7)  # ~1.3 is ideal for triangle
    triangle_shape = 0.8 if features['is_triangle_like'] else 0.2  # Big boost if hull approximates to triangle
    confidence['triangle'] = int((triangle_direction * 0.4 + triangle_compactness * 0.3 + triangle_shape * 0.3) * 100)
    
    # Calculate flick confidence
    # Ideal flick has vertical aspect ratio, low direction changes, straight, quick motion
    flick_aspect = max(0, 1 - features['aspect_ratio']) * 2 if features['aspect_ratio'] < 0.5 else 0  # Lower is better
    flick_direction = max(0, 1 - features['direction_changes'] / 3) if features['direction_changes'] <= 3 else 0  # Fewer is better
    flick_straightness = features['straightness'] if features['straightness'] > 0.7 else 0  # Straighter is better
    flick_speed = max(0, 1 - features['point_density']) if features['point_density'] < 0.3 else 0  # Lower density = faster
    
    # Check if there was significant vertical movement
    vertical_movement = 1.0 if features['vertical_direction'] is not None else 0.2
    
    confidence['flick'] = int((flick_aspect * 0.3 + flick_direction * 0.2 + 
                          flick_straightness * 0.2 + flick_speed * 0.2 + 
                          vertical_movement * 0.1) * 100)
    
    return confidence

def recognize_gesture(features):
    """Recognize gesture based on calculated features and confidence scores"""
    if features is None:
        return "none", {}
    
    # Calculate confidence scores for each gesture
    confidence = calculate_confidence_scores(features)
    
    # Find the gesture with the highest confidence
    max_confidence = 0
    max_gesture = "none"
    
    for gesture, score in confidence.items():
        if score > max_confidence:
            max_confidence = score
            max_gesture = gesture
    
    # Only recognize if confidence is above threshold
    if max_confidence >= CONFIDENCE_THRESHOLD:
        return max_gesture, confidence
    else:
        return "none", confidence

# Main program
def main():
    # Initialize camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (1280, 720)})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # Allow camera to warm up
    
    # Grab first frame
    first_frame = picam2.capture_array()
    if first_frame is None:
        print("Failed to capture initial frame.")
        exit()
    
    # Initialize variables
    mask = np.zeros_like(first_frame)
    prev_frame = picam2.capture_array()
    prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
    last_reset_time = time.time()
    
    # Points deque for tracking IR LED path
    points = deque(maxlen=PATH_LENGTH)
    last_gesture_time = time.time()
    last_gesture = "none"
    last_confidence = {}
    
    print("Gesture Recognition System Started")
    print("Press ESC to exit")
    
    while True:
        # Capture frame
        frame = picam2.capture_array()
        
        # Convert to grayscale
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Compute frame difference for motion detection
        frame_diff = cv.absdiff(prev_gray, frame_gray)
        _, motion_mask = cv.threshold(frame_diff, MOTION_THRESHOLD, 255, cv.THRESH_BINARY)
        
        # Find brightest point in motion areas (IR LED)
        motion_regions = cv.bitwise_and(frame_gray, frame_gray, mask=motion_mask)
        _, max_val, _, max_loc = cv.minMaxLoc(motion_regions)
        
        current_time = time.time()
        
        # Only track if point is bright enough (IR LED)
        if max_val > 150:
            # Only add points at a certain interval to prevent too many close points
            # This helps with better gesture recognition by making points more significant
            if len(points) == 0 or (len(points) > 0 and 
                                    np.linalg.norm(np.array(max_loc) - np.array(points[0])) > 3):
                points.appendleft(max_loc)
            
            # Draw line connecting points
            for i in range(1, len(points)):
                if points[i - 1] is None or points[i] is None:
                    continue
                cv.line(mask, points[i - 1], points[i], (255, 255, 255), 2)
        
        # Only display current tracking point if bright enough
        if max_val > 150:
            # Draw a circle at the current tracking point
            cv.circle(frame, max_loc, 5, (0, 255, 0), -1)
        
        # Recognize gesture if we have enough points and 1.5 seconds passed since last recognition
        if len(points) >= MIN_PATH_POINTS and current_time - last_gesture_time > 1.5:
            features = calculate_features(list(points))
            
            if features:  # Make sure features were calculated properly
                gesture, confidence = recognize_gesture(features)
                
                if gesture != "none":
                    # Print gesture and confidence
                    print(f"Gesture detected: {gesture}")
                    for g, score in confidence.items():
                        print(f"  {g}: {score}%")
                    
                    # Trigger LED animation based on gesture
                    if gesture == "circle":
                        circle_animation()
                    elif gesture == "square":
                        square_animation()
                    elif gesture == "triangle":
                        triangle_animation()
                    elif gesture == "flick":
                        flick_animation()
                    
                    # Save the gesture and confidence
                    last_gesture = gesture
                    last_confidence = confidence
                    last_gesture_time = current_time
                    
                    # Reset points after recognition
                    points.clear()
                    mask = np.zeros_like(frame)
        
        # Reset if tracking timeout reached
        if current_time - last_reset_time >= RESET_TIME:
            mask = np.zeros_like(frame)
            last_reset_time = current_time
        
        # Create output display
        output_frame = cv.add(frame, mask)
        
        # Add gesture name to display
        cv.putText(
            output_frame, 
            f"Last Gesture: {last_gesture.capitalize()}", 
            (10, 30), 
            cv.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        # Draw current path
        path_display = np.zeros_like(frame)
        for i in range(1, len(points)):
            if points[i - 1] is None or points[i] is None:
                continue
            cv.line(path_display, points[i - 1], points[i], (0, 255, 0), 2)
            
        # Draw bounding box of current path to visualize gesture size
        if len(points) >= MIN_PATH_POINTS:
            points_array = np.array(list(points))
            x, y, w, h = cv.boundingRect(points_array)
            cv.rectangle(path_display, (x, y), (x + w, y + h), (0, 165, 255), 2)
            
            # Show the area size on screen to help with debugging
            area_text = f"Area: {w * h} (Min: {MIN_GESTURE_SIZE})"
            cv.putText(path_display, area_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            # Show confidence scores if we have a recent gesture
            if len(last_confidence) > 0:
                y_offset = 90
                for gesture, score in last_confidence.items():
                    cv.putText(
                        path_display,
                        f"{gesture.capitalize()}: {score}%",
                        (10, y_offset),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255) if gesture != last_gesture else (0, 255, 0),
                        1
                    )
                    y_offset += 30
        
        # Display
        cv.imshow("IR Wand Tracking", resize_image(output_frame))
        cv.imshow("Current Path", resize_image(path_display))
        
        # Update previous frame
        prev_gray = frame_gray.copy()
        
        # Check for exit key
        k = cv.waitKey(30) & 0xff
        if k == 27:  # ESC
            break
    
    # Clean up
    cv.destroyAllWindows()
    picam2.stop()
    clear()  # Turn off LEDs

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Program stopped by user")
        clear()  # Ensure LEDs are off
    finally:
        cv.destroyAllWindows()
