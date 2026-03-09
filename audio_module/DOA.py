import numpy as np
import itertools
from config import SAMPLE_RATE as fs, PED_FACTOR as pf, SPEED_OF_SOUND as c, MIC_POSITIONS

# --- CORE MATH FUNCTION (Used by both GCC and SRP) ---
def get_correlation_array(sig1, sig2):
    n = len(sig1)
    n_interpulated = n * pf
    
    # 1. FFT
    fft1 = np.fft.rfft(sig1)
    fft2 = np.fft.rfft(sig2)

    # 2. Cross-power spectrum
    cross_power = fft1 * np.conj(fft2)
    
    # 3. PHAT weighting
    cross_power_phat = cross_power / (np.abs(cross_power) + 1e-10) 
    
    # 4. Anti-aliasing filter (Spatial Nyquist for 6cm)
    max_freq = 2800.0
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    cross_power_phat[freqs > max_freq] = 0.0

    # 5. Interpolate to higher resolution (10x)
    padded_phat=np.zeros(n_interpulated//2 + 1, dtype=complex)

    # Copy the original PHAT values into the padded array
    padded_phat[:len(cross_power_phat)] = cross_power_phat

    # 6. IFFT
    cross_correlation = np.fft.irfft(padded_phat, n_interpulated)
    cross_correlation_shifted = np.fft.fftshift(cross_correlation)  
    
    return cross_correlation_shifted

def compute_gcc_phat(sig1, sig2):
    n = len(sig1)
    n_interpulated = n * pf
    effective_sample_rate = fs * pf

    center_idx = n_interpulated // 2  
    
    # Call the helper function to get the array instead of calculating it here
    cross_correlation_shifted = get_correlation_array(sig1, sig2)

    # Find the RAW index of the maximum correlation peak
    max_window = 6*pf  # +/- 6 samples at original rate, scaled by interpolation factor  
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
    delay_samples_padded = (shift_idx - center_idx) // pf # Scale back to original sample rate units
    
    # 7. Convert to seconds
    delay_seconds = delay_samples_padded / effective_sample_rate
    
    return delay_seconds, delay_samples_padded
   
def generate_srpphat_lookup_table():
    # Generate all 6 possible unique pairs of microphones (indices 0 to 3)
    MIC_PAIRS = list(itertools.combinations(range(len(MIC_POSITIONS)), 2))

    # Create a dictionary or list to store the physical distance for each pair
    PAIR_DISTANCES = []
    for i, j in MIC_PAIRS:
        # The physical distance 'd' between microphone i and microphone j
        distance = abs(MIC_POSITIONS[i] - MIC_POSITIONS[j])
        PAIR_DISTANCES.append(distance)


    degrees_of_interest=np.arange(-89.5, 89.5, 1.0)  # From -89.5 to +89.5 degrees in 1 degree steps
    delay_degree_sample = np.zeros((len(degrees_of_interest), len(MIC_PAIRS)), dtype=int)
    for idx, angle in enumerate(degrees_of_interest):
        theta_rad = np.radians(angle)
        for pair_idx, (i, j) in enumerate(MIC_PAIRS):
            d = PAIR_DISTANCES[pair_idx]
            delay = (d * np.sin(theta_rad)) / c  # Time delay in seconds
            delay_degree_sample[idx, pair_idx] = int(np.floor(delay * fs * pf + 0.5))  # Convert delay from seconds to samples at 16 kHz
    return delay_degree_sample, degrees_of_interest, MIC_PAIRS

def srp_phat_localization(audio_frame, lookup_table, mic_pairs, angles):

    #compute correlation for each pair
    correlations = []
    for i, j in mic_pairs:
        sig1 = audio_frame[:, i]
        sig2 = audio_frame[:, j]
        corr_array = get_correlation_array(sig1, sig2)
        correlations.append(corr_array)
    # find the middle index of the correlation array (zero delay)
    center_idx = len(correlations[0]) // 2
    # Array to store the energy score for each angle
    srp_powers = np.zeros(len(angles))
    # compute the SRP-PHAT score for each angle in the lookup table
    
    for angle_idx in range(len(angles)):
        score = 0
        for pair_idx in range(len(mic_pairs)):
            delay_offset = lookup_table[angle_idx, pair_idx]
            target_idx = center_idx + delay_offset
            score+= correlations[pair_idx][target_idx]
        srp_powers[angle_idx] = score
    # Find the angle with the maximum SRP-PHAT score
    best_angle_idx = np.argmax(srp_powers)
    best_angle = angles[best_angle_idx]
            
    return best_angle, srp_powers
def VAD(audio_frame, energy_threshold=0.01):    
 # VAD - Voice Activity Detection
    energy1 = np.mean(audio_frame[:, 0]**2)
    energy2 = np.mean(audio_frame[:, 1]**2)
    energy3 = np.mean(audio_frame[:, 2]**2)
    energy4 = np.mean(audio_frame[:, 3]**2)
    
    if energy1 < (energy_threshold**2) and energy2 < (energy_threshold**2) and energy3 < (energy_threshold**2) and energy4 < (energy_threshold**2):
        return False  # No voice activity detected
    return True  # Voice activity detected
    
def fractional_delay(signal, delay_in_samples):
    """ Shifts a signal by a fractional number of samples using FFT """
    N = len(signal)
    fft_sig = np.fft.rfft(signal)
    # Get the frequencies 
    freqs = np.fft.rfftfreq(N)
    # Apply phase shift: e^(-j * 2 * pi * f * delay)
    shifted_fft = fft_sig * np.exp(-1j * 2 * np.pi * freqs * delay_in_samples)
    return np.fft.irfft(shifted_fft, N)

if __name__ == "__main__":
    print("--- Running SRP-PHAT Sanity Check ---")

    # 1. Generate the spatial map (Lookup Table)
    lookup_table, angles, mic_pairs = generate_srpphat_lookup_table()

    # 2. Simulate a sound source coming from +20 degrees
    true_angle = 20.0
    theta_rad = np.radians(true_angle)

    # Generate random white noise as our "voice" (4000 samples)
    source_signal = np.random.randn(4000)

    # Create the empty 4-channel audio frame
    audio_frame = np.zeros((4000, 4))

    # Apply the exact delays to each microphone based on physics
    for mic_idx in range(4):
        dist = MIC_POSITIONS[mic_idx]
        delay_sec = (dist * np.sin(theta_rad)) / c
        
        # Calculate EXACT fractional delay in samples
        exact_delay_samples = delay_sec * fs
        
        # Shift the signal perfectly, even if it's 0.478 samples!
        audio_frame[:, mic_idx] = fractional_delay(source_signal, exact_delay_samples)

    # 3. Feed the fake audio into our SRP-PHAT function
    estimated_angle, powers = srp_phat_localization(audio_frame, lookup_table, fs, mic_pairs, angles)

    print(f"Target True Angle: {true_angle} degrees")
    print(f"SRP-PHAT Estimated Angle: {estimated_angle} degrees")
   