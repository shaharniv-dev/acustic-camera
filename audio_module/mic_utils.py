import sounddevice as sd
import numpy as np

# Function to test the 4-channel microphone recording
def test_recording(device_id=6, duration=3, sample_rate=16000):
    print("Starting short test recording...")
    
    # Record from the 4 microphones
    audio_data = sd.rec(int(duration * sample_rate), 
                        samplerate=sample_rate, 
                        channels=4, 
                        device=device_id)
    
    # Block until done
    sd.wait()
    
    print("Test finished successfully!")
    print("The shape of the audio data is:", audio_data.shape)
    
    # Return the matrix so we can use it later if needed
    return audio_data

def get_ps3eye_index():
    # Get all devices and all host APIs
    devices = sd.query_devices()
    host_apis = sd.query_hostapis()
    
    for i, dev in enumerate(devices):
        # 1. Check for 4 channels
        if dev['max_input_channels'] >= 4:
            # 2. Get the name of the Host API for this device
            api_idx = dev['hostapi']
            api_name = host_apis[api_idx]['name']
            
            # 3. Check if it's WASAPI or DirectSound
            if "WASAPI" in api_name or "DirectSound" in api_name:
                print(f"[*] Detected 4-channel device: '{dev['name']}'")
                print(f"[*] Using Host API: {api_name} at index {i}")
                return i
                
    print("[!] Error: Could not find a 4-channel microphone on WASAPI/DirectSound.")
    return None