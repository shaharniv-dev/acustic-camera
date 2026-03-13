
import itertools
import numpy as np

SAMPLE_RATE = 16000
CHUNK_SIZE = 4000 
PED_FACTOR = 30
SPEED_OF_SOUND = 343.0
MIC_POSITIONS = [-0.03, -0.01, 0.01, 0.03]

# Generate all 6 possible unique pairs of microphones (indices 0 to 3)
MIC_PAIRS = list(itertools.combinations(range(len(MIC_POSITIONS)), 2))

# Create a dictionary or list to store the physical distance for each pair
PAIR_DISTANCES = []
for i, j in MIC_PAIRS:
    # The physical distance 'd' between microphone i and microphone j
    distance = abs(MIC_POSITIONS[i] - MIC_POSITIONS[j])
    PAIR_DISTANCES.append(distance)

ORIG_BINS = CHUNK_SIZE // 2 + 1
# Calculate N_FFT based on chunk size and zero-padding factor
PADDED_N_FFT = CHUNK_SIZE * PED_FACTOR
PADDED_BINS = PADDED_N_FFT // 2 + 1

# Convert distances to a NumPy array for vectorized operations
pair_distances_np = np.array(PAIR_DISTANCES)

# Calculate spatial Nyquist frequency (cutoff) for each pair
cutoff_freqs = SPEED_OF_SOUND / (2 * pair_distances_np)

freq_bins = np.fft.rfftfreq(CHUNK_SIZE, d=1.0/SAMPLE_RATE)
freq_mask = freq_bins < cutoff_freqs[:, np.newaxis]

# Extract indices for vectorized slicing
mic1_idx = [pair[0] for pair in MIC_PAIRS]
mic2_idx = [pair[1] for pair in MIC_PAIRS]
num_pairs = len(MIC_PAIRS)