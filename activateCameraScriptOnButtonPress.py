import RPi.GPIO as GPIO
import subprocess
import time

# -------- CONFIG --------
TRIGGER_PIN = 27  # change to your GPIO pin (BCM numbering)
SCRIPT_TO_RUN = "/home/pi/me522Project/tf-env/bin/python /home/pi/me522Project/cameraInterface.py"
# ------------------------

def run_camera_interface(channel):
    print("GPIO signal detected! Running cameraInterface.py...")

    # Run the script
    subprocess.Popen(SCRIPT_TO_RUN, shell=True)

    # optional debounce delay
    time.sleep(1)

def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(TRIGGER_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

    # detect rising edge (LOW -> HIGH signal)
    GPIO.add_event_detect(TRIGGER_PIN, GPIO.RISING, callback=run_camera_interface, bouncetime=500)

    print(f"Listening on GPIO {TRIGGER_PIN}... Press Ctrl+C to exit")

    try:
        while True:
            time.sleep(0.5)  # keep program alive
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    main()