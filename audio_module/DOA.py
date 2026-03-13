import sys
import os
import numpy as np

# Add the parent directory to Python's system path so it can find config.py
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Now it will successfully find and import config from the parent folder
import config

# --- CORE MATH FUNCTION (Used by both GCC and SRP) ---
def get_correlation_array(fft1, fft2,dist):
    """
    Computes the GCC-PHAT cross-correlation array between two FFT signals with anti-aliasing and interpolation.
    This is a helper function that returns the entire cross-correlation array, which can then be searched for peaks.
    """
    n = len(fft1)
    n_interpulated = n * config.PED_FACTOR
  
    # 1. Cross-power spectrum
    cross_power = fft1 * np.conj(fft2)
    
    # 2. PHAT weighting
    cross_power_phat = cross_power / (np.abs(cross_power) + 1e-10) 
    
    # 3. Anti-aliasing filter (Spatial Nyquist for 6cm)
    max_freq = config.SPEED_OF_SOUND / (2 * dist)  # Nyquist frequency based on mic pair distance
    freqs = np.fft.rfftfreq(n, d=1.0/config.SAMPLE_RATE)
    cross_power_phat[freqs > max_freq] = 0.0

    # 4. Interpolate to higher resolution (10x)
    padded_phat=np.zeros(n_interpulated//2 + 1, dtype=complex)

    # Copy the original PHAT values into the padded array
    padded_phat[:len(cross_power_phat)] = cross_power_phat

    # 5. IFFT
    cross_correlation = np.fft.irfft(padded_phat, n_interpulated)
    cross_correlation_shifted = np.fft.fftshift(cross_correlation)  
    
    return cross_correlation_shifted

def get_all_correlations_vectorized(fft_signals):
    """
    This is the current implementation.
    Computes the cross-correlation arrays for all microphone pairs in a fully vectorized manner.
    This function assumes that the input fft_signals is a 2D array of shape (num_mics, num_bins) 
    and that the global config variables for mic pairs and frequency masking are properly defined.
    """
    
    # Extract signals for all pairs simultaneously 
    # Resulting shapes: (6, config.NUM_BINS)
    sig1_matrix = fft_signals[config.mic1_idx, :]
    sig2_matrix = fft_signals[config.mic2_idx, :]
    
    # Compute cross-power spectrum for all pairs
    cross_powers = sig1_matrix * np.conj(sig2_matrix)
    
    # Apply PHAT weighting
    magnitudes = np.abs(cross_powers)
    # Prevent division by zero
    magnitudes[magnitudes == 0] = 1e-8 
    phat_cross_powers = cross_powers / magnitudes
    
    # Apply the precomputed dynamic low-pass filter (spatial anti-aliasing)
    filtered_cross_powers = phat_cross_powers * config.freq_mask
    
    # Inverse FFT for all pairs at once 
    # Output shape: (6, config.N_FFT)
    all_correlations = np.fft.irfft(filtered_cross_powers, n=config.N_FFT, axis=1)
    
    return all_correlations


def compute_gcc_phat(fft1, fft2):
    
    n = len(fft1)
    n_interpulated = n * config.PED_FACTOR
    effective_sample_rate = config.SAMPLE_RATE * config.PED_FACTOR
    center_idx = n_interpulated // 2  

    # Call the helper function to get the array instead of calculating it here
    cross_correlation_shifted = get_correlation_array(fft1, fft2)

    # Find the RAW index of the maximum correlation peak
    max_window = 6*config.PED_FACTOR  # +/- 6 samples at original rate, scaled by interpolation factor  
    start_idx = center_idx - max_window
    end_idx = center_idx + max_window + 1

    search_window = cross_correlation_shifted[start_idx:end_idx]
    shift_idx = np.argmax(search_window) + start_idx  

    # # --- SUB-SAMPLE INTERPOLATION (Parabolic Fit) for GCC without interpulating ---
    """"
    if 0 < shift_idx < len(cross_correlation_shifted) - 1:
        y_minus1 = cross_correlation_shifted[shift_idx - 1]
        y_0 = cross_correlation_shifted[shift_idx]
        y_plus1 = cross_correlation_shifted[shift_idx + 1]
        
        denominator = y_minus1 - 2 * y_0 + y_plus1
        
        if denominator == 0:
            fraction = 0.0
        else:
            fraction = 0.5 * (y_minus1 - y_plus1) / denominator
    else:
        fraction = 0.0
        
    exact_shift_idx = shift_idx + fraction
    """
    # 6. NOW subtract the center to get the true fractional delay
    delay_samples_padded = (shift_idx - center_idx) // config.PED_FACTOR # Scale back to original sample rate units
    
    # 7. Convert to seconds
    delay_seconds = delay_samples_padded / effective_sample_rate
    
    return delay_seconds, delay_samples_padded
   
def generate_srpphat_lookup_table():
    """
     Generates a lookup table for SRP-PHAT that maps each angle 
     to the corresponding sample delays for all microphone pairs.
     This is precomputed to allow for fast steering during localization.
    """
    degrees=np.arange(-89.5, 89.5, 0.5)  # From -89.5 to +89.5 degrees in 0.5 degree steps
    delay_degree_sample = np.zeros((len(degrees), len(config.MIC_PAIRS)), dtype=int)
    for idx, angle in enumerate(degrees):
        theta_rad = np.radians(angle)
        for pair_idx, (i, j) in enumerate(config.MIC_PAIRS):
            d = config.PAIR_DISTANCES[pair_idx]
            delay = (d * np.sin(theta_rad)) / config.SPEED_OF_SOUND  # Time delay in seconds
            delay_degree_sample[idx, pair_idx] = int(np.floor(delay * config.SAMPLE_RATE * config.PED_FACTOR + 0.5))  # Convert delay from seconds to samples at 16 kHz
    return delay_degree_sample, degrees 

def srp_phat_localization(audio_frame, lookup_table, angles):
    """
    Performs SRP-PHAT localization on a given audio frame using the precomputed lookup table.
    This function computes the spatial power spectrum for all angles and returns the angle with the highest power
    along with the power spectrum itself.
    """
    # 1. Vectorized FFT on original chunks
    fft_signals = np.fft.rfft(audio_frame, axis=0).T
    
    # 2. Extract pairs for cross-power calculation
    sig1_matrix = fft_signals[config.mic1_idx, :]
    sig2_matrix = fft_signals[config.mic2_idx, :]
    
    # 3. Compute cross-power and apply PHAT weighting
    cross_powers = sig1_matrix * np.conj(sig2_matrix)
    magnitudes = np.abs(cross_powers)
    magnitudes[magnitudes == 0] = 1e-8 
    phat_cross_powers = cross_powers / magnitudes
    
    # 4. Apply dynamic low-pass filter (spatial anti-aliasing)
    filtered_cross_powers = phat_cross_powers * config.freq_mask
    
    # 5. Frequency Domain ZERO-PADDING to increase time resolution for IFFT
    padded_cross_powers = np.zeros((config.num_pairs, config.PADDED_BINS), dtype=complex)
    padded_cross_powers[:, :config.ORIG_BINS] = filtered_cross_powers
    
    # 6. Vectorized IFFT on the padded array
    raw_correlations = np.fft.irfft(padded_cross_powers, n=config.PADDED_N_FFT, axis=1)
    
    # Shift zero-delay to the center to match lookup table 
    correlations = np.fft.fftshift(raw_correlations, axes=1)
    
    # 7. Vectorized Delay-and-Sum (SRP) Steering
    center_idx = correlations.shape[1] // 2
    
    # lookup_table shape is (num_angles, num_pairs)
    # target_indices shape becomes (num_angles, num_pairs)
    target_indices = center_idx - lookup_table
    
    # pair_indices shape: (num_pairs,)  np.newaxis makes it (1, num_pairs)
    pair_indices = np.arange(config.num_pairs)
    
    # Advanced indexing: broadcasts row indices across all angles to extract 
    # the exact steered value for every angle and pair simultaneously.
    steered_correlations = correlations[pair_indices[np.newaxis, :], target_indices]
    
    # Sum across pairs to get the final spatial power spectrum
    srp_powers = np.sum(steered_correlations, axis=1)

    # 8. Find the Best Angle
    best_angle_idx = np.argmax(srp_powers)
    best_angle = angles[best_angle_idx]

   
    return best_angle, srp_powers


def VAD(audio_frame, energy_threshold=0.01): 
    """
    Simple energy-based Voice Activity Detection (VAD).
    Returns True if voice activity is detected, False otherwise.
    """   
 # VAD - Voice Activity Detection
    energy1 = np.mean(audio_frame[:, 0]**2)
    energy2 = np.mean(audio_frame[:, 1]**2)
    energy3 = np.mean(audio_frame[:, 2]**2)
    energy4 = np.mean(audio_frame[:, 3]**2)
    
    if energy1 < (energy_threshold**2) and energy2 < (energy_threshold**2) and energy3 < (energy_threshold**2) and energy4 < (energy_threshold**2):
        return False  # No voice activity detected
    return True  # Voice activity detected
    

if __name__ == "__main__":
    import time
    
    print("--- Testing Vectorized SRP-PHAT Accuracy ---")
    
    # 1. Generate the lookup table
    print("Generating lookup table...")
    # Make sure this matches how your function actually returns values
    lookup_table, angles = generate_srpphat_lookup_table()
    
    # 2. Define a known target angle for our simulated source
    TARGET_ANGLE = 30.0  # degrees
    theta_rad = np.radians(TARGET_ANGLE)
    
    # 3. Create a simulated audio frame (broadband noise)
    # We will artificially delay this noise to mimic a physical sound wave
    source_signal = np.random.randn(config.CHUNK_SIZE)
    fft_source = np.fft.rfft(source_signal)
    freqs = np.fft.rfftfreq(config.CHUNK_SIZE, 1/config.SAMPLE_RATE)
    
    mock_audio = np.zeros((config.CHUNK_SIZE, len(config.MIC_POSITIONS)))
    
    print(f"\n--- Simulating Source at {TARGET_ANGLE} degrees ---")
    for i, pos in enumerate(config.MIC_POSITIONS):
        # Calculate exact time delay for this mic relative to the center of the array (x=0)
        delay_sec = (pos * np.sin(theta_rad)) / config.SPEED_OF_SOUND
        
        # Apply exact fractional delay in the frequency domain
        phase_shift = np.exp(-1j * 2 * np.pi * freqs * delay_sec)
        mock_audio[:, i] = np.fft.irfft(fft_source * phase_shift)
        
        print(f"Mic {i} (pos {pos}m): absolute delay = {delay_sec*1000:.3f} ms")
        
    print("\n--- Theoretical Pairwise Delays (Your Lookup Table Logic) ---")
    for pair_idx, (i, j) in enumerate(config.MIC_PAIRS):
        d = config.PAIR_DISTANCES[pair_idx]
        pair_delay_sec = (d * np.sin(theta_rad)) / config.SPEED_OF_SOUND
        
        # This is your exact formula for padded samples:
        pair_delay_samples = int(np.floor(pair_delay_sec * config.SAMPLE_RATE * config.PED_FACTOR + 0.5))
        
        print(f"Pair ({i}, {j}): dist = {d:.2f}m | delay = {pair_delay_sec*1000:.3f} ms | padded shift = {pair_delay_samples} samples")
        
    # 4. Run the localization function
    print("\n--- Running srp_phat_localization... ---")
    start_time = time.perf_counter()
    
    best_angle, srp_powers = srp_phat_localization(mock_audio, lookup_table, angles)
    
    end_time = time.perf_counter()
    execution_time_ms = (end_time - start_time) * 1000
    
    # 5. Print final validation results
    print("\n--- Results ---")
    print(f"Target simulated angle:   {TARGET_ANGLE} degrees")
    print(f"Estimated measured angle: {best_angle} degrees")
    
    # Calculate error
    error = abs(TARGET_ANGLE - best_angle)
    print(f"Absolute Error:           {error:.2f} degrees")
    
    # We allow a tiny margin of error because your grid resolution is 1.0 degree steps
    if error <= 1.0:  
        print("Status: SUCCESS! The math, matrices, and indexing are perfectly aligned.")
    else:
        print("Status: FAILED. The estimated angle is too far from the target.")
        
    print(f"Execution time: {execution_time_ms:.2f} ms")
   