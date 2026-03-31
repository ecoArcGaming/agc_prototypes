import cv2
import time
import numpy as np
import sounddevice as sd
import Jetson.GPIO as GPIO

# implement VLM logic here
from vlm import process_data 

# --- Configuration ---
BUTTON_PIN = 15          # Physical Board Pin 15 (adjust to your wiring)
VIDEO_DEVICE = 0         # /dev/video0
SAMPLE_RATE = 44100      # Standard audio sample rate
RECORD_SECONDS = 5       # Duration to record


def setup_gpio():
    """Configures the GPIO pins."""
    GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
    # Setup pin with an internal pull-up resistor. 
    # Wire your button to connect Pin 15 to GND when pressed.
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def capture_frame():
    """Captures a single frame from the capture card."""
    print("Capturing frame...")
    cap = cv2.VideoCapture(VIDEO_DEVICE)
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return None
    
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def record_audio(duration, fs):
    """Records audio from the I2S microphone."""
    # Record mono audio (channels=1) as 16-bit integers
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait() # Block execution until recording finishes
    print("Recording complete.")
    return recording

def play_audio(audio_data, fs):
    """Plays audio through the I2S amplifier."""
    print("Playing response audio...")
    sd.play(audio_data, samplerate=fs)
    sd.wait() # Block execution until playback finishes

def main():
    setup_gpio()
    print("System ready. Waiting for button press...")
    
    try:
        while True:
            # Because of the pull-up resistor, the pin reads LOW when pressed (connected to GND)
            if GPIO.input(BUTTON_PIN) == GPIO.LOW:
                print("\n--- Button Pressed! ---")
                
                # 1. Take a frame
                frame = capture_frame()
                if frame is None:
                    print("Skipping cycle due to camera error.")
                    time.sleep(1)
                    continue
                
                # 2. Record 5 seconds of audio
                audio_data = record_audio(RECORD_SECONDS, SAMPLE_RATE)
                
                # 3. Call the external function
                print("Sending data to external function...")
                # Assuming process_data takes the frame and audio, and returns audio
                response_audio = process_data(frame, audio_data, SAMPLE_RATE) 
                
                # 4. Play response
                if response_audio is not None:
                    play_audio(response_audio, SAMPLE_RATE)
                else:
                    print("No audio returned from external function.")
                
                print("\nSequence complete. Waiting for next press...")
                time.sleep(0.5) # Debounce delay so holding the button doesn't spam it
                
            # Small sleep to prevent the while loop from consuming 100% CPU
            time.sleep(0.05) 
            
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    finally:
        GPIO.cleanup() # Always clean up GPIO states on exit

if __name__ == '__main__':
    main()