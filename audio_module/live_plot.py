import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import queue

# Configuration constants
DEVICE_ID = 6
CHANNELS = 4
SAMPLE_RATE = 16000
WINDOW_SIZE = 8000  # Display 0.5 seconds of audio

# Create a thread-safe queue to pass data between the audio thread and the GUI
audio_queue = queue.Queue()

# 1. Audio Callback
def audio_callback(indata, frames, time, status):
    # Put a copy of the new audio chunk into the queue
    audio_queue.put(indata.copy())

# 2. GUI Setup 
fig, axes = plt.subplots(CHANNELS, 1, figsize=(8, 6))
lines = []

# Initialize the 4 lines with arrays of zeros
for i in range(CHANNELS):
    line, = axes[i].plot(np.zeros(WINDOW_SIZE))
    axes[i].set_ylim([-0.5, 0.5])
    axes[i].set_ylabel(f"Mic {i+1}")
    lines.append(line)

# This matrix holds the current visual window
plot_data = np.zeros((WINDOW_SIZE, CHANNELS))

# --- 3. Animation Update (The Slow Worker) ---
def update_plot(frame):
    global plot_data
    
    # Read all available new data from the quepython live_plot.pyue
    while True:
        try:
            data = audio_queue.get_nowait()
        except queue.Empty:
            break
        
        # Shift the old data to the left and insert the new data on the right
        shift = len(data)
        if shift < WINDOW_SIZE:
            plot_data = np.roll(plot_data, -shift, axis=0)
            plot_data[-shift:] = data
            
    # Update the visual lines on the graph
    for i in range(CHANNELS):
        lines[i].set_ydata(plot_data[:, i])
        
    return lines

# --- 4. Main Execution ---
print("Starting live stream... Close the plot window to stop.")

# Open the audio stream
stream = sd.InputStream(device=DEVICE_ID, channels=CHANNELS, samplerate=SAMPLE_RATE, callback=audio_callback)

with stream:
    # Start the animation loop that refreshes the screen every 30 milliseconds
    ani = FuncAnimation(fig, update_plot, interval=30, blit=True)
    plt.show()
    