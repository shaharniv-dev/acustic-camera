import sounddevice as sd
import numpy as np
import cv2 as cv
import threading
import time

# --- Audio Modules ---
from audio_module.DOA import generate_srpphat_lookup_table, srp_phat_localization, VAD
from audio_module.mic_utils import get_ps3eye_index
from config import CHUNK_SIZE, MIC_POSITIONS, SPEED_OF_SOUND as c, SAMPLE_RATE as fs

# --- Camera Modules ---
from camera_module.camera_io import CameraIO
from camera_module.spatial_math import SpatialMath
from camera_module.renderer import OverlayRenderer

# ==========================================
# Thread-Safe Shared State
# ==========================================
class AcousticState:
    def __init__(self):
        self.lock = threading.Lock()
        self.is_active = False
        self.angles = None
        self.srp_powers = None

shared_state = AcousticState()

# ==========================================
# Audio Thread Callback
# ==========================================
def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[!] Audio Status Error: {status}")
    
    # Voice Activity Detection (VAD)
    if not VAD(indata):
        with shared_state.lock:
            shared_state.is_active = False
        return
        
    # 2. Compute Spatial Power Spectrum (SRP-PHAT)
    # Assuming lookup_table, mic_pairs, and precomputed_angles are strictly available in global scope
    global lookup_table, mic_pairs, precomputed_angles
    estimated_angle, srp_powers = srp_phat_localization(indata, lookup_table,precomputed_angles)
    
    # 3. Safely update the shared state for the video thread
    with shared_state.lock:
        shared_state.is_active = True
        shared_state.angles = precomputed_angles
        shared_state.srp_powers = srp_powers

# ==========================================
# Main Video Thread & Application Loop
# ==========================================
def main():
    print("[*] Initializing System Components...")
    
    # 1. Initialize Audio Structures
    global lookup_table, mic_pairs, precomputed_angles
    lookup_table, precomputed_angles = generate_srpphat_lookup_table()
    audio_idx = get_ps3eye_index()
    if audio_idx is None:
        print("[!] Exiting: No valid microphone found.")
        exit(1)

    # 2. Initialize Optical & Graphics Pipelines
    try:
        cam = CameraIO() # Will automatically find hardware
        mapper = SpatialMath(fov_h_deg=75.0, width=cam.actual_w, height=cam.actual_h)
        renderer = OverlayRenderer(mapper=mapper, alpha=0.6, threshold_ratio=0.75)
    except Exception as e:
        print(f"[!] Hardware initialization failed: {e}")
        exit(1)

    print("\n[*] Starting synchronized live stream... Press 'q' to exit.")

    # 3. Launch Background Audio Thread
    audio_stream = sd.InputStream(device=audio_idx, channels=4, samplerate=fs, 
                                  blocksize=CHUNK_SIZE, dtype=np.float32, 
                                  callback=audio_callback)
    
    with audio_stream:
        try:
            # 4. Main Video Loop
            while True:
                # Capture visual frame
                frame = cam.get_frame()
                frame = cv.flip(frame, 1) # Mirror for natural interaction
                if frame is None:
                    print("[!] Frame dropped.")
                    continue

                # Safely copy the latest acoustic data
                with shared_state.lock:
                    is_active = shared_state.is_active
                    if is_active:
                        angles_data = np.copy(shared_state.angles)
                        powers_data = np.copy(shared_state.srp_powers)

                # Process and Overlay if sound is present
                if is_active and powers_data is not None:
                    # Mathematical projection (1D Acoustic -> 2D Optical Matrix)
                    heatmap_2d = mapper.map_srp_to_image(angles_data, powers_data)
                    
                    # Graphical rendering (Matrix -> Color Mask -> Alpha Blending)
                    display_frame = renderer.render(frame, heatmap_2d, angles_data, powers_data)
                else:
                    display_frame = frame

                # Display Output
                cv.imshow("Acoustic Camera (Live)", display_frame)
                
                # Check for termination command
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\n[*] Graceful shutdown initiated via KeyboardInterrupt.")
        finally:
            print("[*] Releasing hardware resources...")
            cam.release()
            cv.destroyAllWindows()

if __name__ == "__main__":
    main()