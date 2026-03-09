import sounddevice as sd
import numpy as np

# Callback function to process audio chunks in real-time
def audio_callback(indata, frames, time, status):
    # Print hardware warnings if any exist
    if status:
        print(f"Status Error: {status}")
    
    # Calculate the absolute maximum amplitude of the first channel
    volume = np.max(np.abs(indata[:, 0]))
    
    # Multiply by a larger number (500) to make the bar more sensitive
    bar_length = int(volume * 500) 
    print("|" + "*" * bar_length)

print("Starting live stream... Press Ctrl+C in the terminal to stop.")

# Open the continuous audio stream
with sd.InputStream(device=6, channels=4, samplerate=16000, callback=audio_callback):
    # Keep the main program alive while the background thread works
    while True:
        sd.sleep(1000)