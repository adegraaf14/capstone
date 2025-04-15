import RPi.GPIO as GPIO
import time
import os
import subprocess
import threading
import signal

import realTimeShape_Pixel_mod as shape_detector

SENSOR_PIN = 17
DEBOUNCE_TIME = 1.0
INACTIVITY_TIMEOUT = 300 #5 minutes
SLEEP_COMMAND = "sudo /opt/vc/bin/tvservice -o && sudo systemctl stop lightdm"
WAKE_COMMAND = "sudo /opt/vc/bin/tvservice -p && sudo systemctl start lightdm"

GPIO.setmode(GPIO.BCM)
GPIO.setup(SENSOR_PIN, GPIO.IN)

system_sleeping = True
last_trigger_time = 0
last_activity_time = 0
app_thread = None
app_running = False
stop_app_event = threading.Event()
inactivity_timer_thread = None
exit_flag = threading.Event()

def start_application():
    global app_thread, app_running, stop_app_event
    
    if app_running:
        return
    
    if not shape_detector.initialize():
        print("Failed to initialize shape detection code")
        return
    
    stop_app_event.clear()
    
    app_thread = threading.Thread(target=run_appliation)
    app_thread.daemon = True
    app_thread.start()
    app_running = True
    print("Main application started")
    
def stop_application():
    
    global app_running, stop_app_event
    
    if not app_running:
        return
    
    stop_app_event.set()
    
    if app_thread and app_thread.is_alive():
        app_thread.join(timeout=3.0)
        
    shape_detector.cleanup()
        
    app_running = False
    print("Main application stopped")
    
def run_application():
    global last_activity_time
    
    try:
        while not stop_app_event.is_set():
            key = shape_detector.process_frame()
            
            if shape_detector.motion_detected(): #need to implement motion detected still
                last_activity_time = time.time()
                
                
            if key == 27:
                break
    except Exception as e:
        print(f"Error in application: {e}")
    finally:
        print("Application thread exiting")
        
        
def wake_up_system():
    global system_sleeping, last_activity_time
    
    current_time = time.time()
    if current_time - last_trigger_time < DEBOUNCE_TIME:
        return
    
    last_trigger_time = current_time
    
    if system_sleeping:
        #wake up
        print("IR signal detected! Waking up...")
        os.system(WAKE_COMMAND)
        
        os.system("sudo cpufreq-set -g performance")
        os.system("echo 1 | sudo tee /sys/class/leds/led0/brightness")
        
        system_sleeping = False
        last_activity_time = time.time()
        
        start_application()
        
        start_inactivity_timer()
        
def go_to_sleep():
    global system_sleeping
    
    if not system_sleeping:
        print("Going to sleep due to inactivity...")
        
        stop_application()
        
        os.system("sudo cpufreq-set -g powersave")
        os.system("echo 0 | sudo tee /sys/class/leds/led0/brightness")
        
        os.system(SLEEP_COMMAND)
        
        system_sleeping = True
        
def inactivity_timer():
    global last_activity_time
    
    while not exit_flag.is_set():
        if not system_sleeping:
            current_time = time.time()
            time_since_activity = current_time - last_activity_time
            
            if time_since_activity >= INACTIVITY_TIMEOUT:
                go_to_sleep()
                
        time.sleep(1)
        
def start_inactivity_timer():
    global inactivity_timer_thread
    
    if inactivity_timer_thread is None or not inactivity_timer_thread.is_alive():
        inactivity_timer_thread = threading.Thread(target=inactivity_timer)
        inactivity_timer_thread.daemon = True
        inactivity_timer_thread.start()
        
def check_ir_state():
    
    global system_sleeping, last_trigger_time
    
    while not exit_flag.is_set():
        if system_sleeping:
            current_time = time.time()
            
            ir_high = GPIO.input(SENSOR_PIN) == GPIO.HIGH
            
            if ir_high and (current_time - last_trigger_time > DEBOUNCE_TIME):
                last_trigger_time = current_time
                wake_up_system()
                
        time.sleep(0.1)

def signal_handler(sig, frame):
    print("Exiting...")
    exit_flag.set()
    stop_application()
    if inactivity_timer_thread and inactivity_timer_thread.is_alive():
        inactivity_timer_thread.join(timeout=1.0)
    if ir_monitor_thread and ir_monitor_thread.is_alive():
        ir_monitor_thread.join(timeout=1.0)
    GPIO_cleanup()
    exit(0)
    
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    try:
        print("IR Sleep controller running.")
        print("System starting in sleep mode. Wave wand to wake up.")
        
        system_sleeping = True
        ir_monitor_thread = threading.Thread(target=check_ir_state)
        ir_monitor_thread.daemon = True
        ir_monitor_thread.start()
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Exiting...")
        exit_flag.set()
        stop_application()
        
    finally:
        GPIO.cleanup()
    
    
    
    
    
    
    
    
    
    
    