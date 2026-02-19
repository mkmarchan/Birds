import numpy as np
import soundfile as sf
import scipy.signal
from smstools.models import sineModel as sm
import matplotlib.pyplot as plt

from sinetracks import SineTrack

def main():
    file_path_a = '/Users/mkmarchan/Downloads/crow_crop_long.wav'
    data_a, sample_rate_a = sf.read(file_path_a)

    # Analysis settings a
    frame_size_a = 1024
    hop_size_a = int(frame_size_a / 4)
    window_a = scipy.signal.windows.dpss(frame_size_a, 5 * np.pi)
    sine_threshold_db_a = -60
    max_num_sines_a = 100
    min_sine_dur_a = 0.03
    freq_dev_offset_a = 100
    freq_dev_slope_a = 0.1
    min_freq_a = 400

    file_path_b = '/Users/mkmarchan/Downloads/car_horn_crop.wav'
    data_b, sample_rate_b = sf.read(file_path_b)

    # Analysis settings b
    frame_size_b = 4096
    hop_size_b = int(frame_size_b / 4)
    window_b = scipy.signal.windows.dpss(frame_size_b, 5 * np.pi)
    sine_threshold_db_b = -90
    max_num_sines_b = 100
    min_sine_dur_b = 0.2
    freq_dev_offset_b = 20

    freqs_a, mags_a, _ = sm.sineModelAnal(
        data_a,
        sample_rate_a,
        window_a,
        frame_size_a,
        hop_size_a,
        sine_threshold_db_a,
        max_num_sines_a,
        min_sine_dur_a,
        freq_dev_offset_a,
        freq_dev_slope_a,
        )
    
    freqs_a[freqs_a < min_freq_a] = 0
    
    amps_a = 10 ** (mags_a / 20)
    
    freqs_b, mags_b, _ = sm.sineModelAnal(
        data_b,
        sample_rate_b,
        window_b,
        frame_size_b,
        hop_size_b,
        sine_threshold_db_b,
        max_num_sines_b,
        min_sine_dur_b,
        freq_dev_offset_b)
        
    amps_b = 10 ** (mags_b / 20)

    # Stuff good values into our amplitudes and magnitudes
    amps_a[freqs_a == 0] = 0
    mags_a[freqs_a == 0] = -200
    amps_b[freqs_b == 0] = 0
    mags_b[freqs_b == 0] = -200

    # Morph settings
    b_morph_frame = 6
    morph_pct = np.minimum(1, np.arange(freqs_a.shape[0]) / (freqs_a.shape[0] / 1.2)) # np.ones(freqs_a.shape[0]) 

    a_tracks = SineTrack.extract_tracks(freqs_a, mags_a)
    b_tracks = SineTrack.extract_tracks(freqs_b, mags_b)

    # # Plot a_tracks
    # for track in a_tracks:
    #     frames = np.arange(track.start_frame, track.end_frame)
    #     plt.plot(frames, track.freqs)

    # plt.show()

    # # Plot b_tracks
    # for track in b_tracks:
    #     frames = np.arange(track.start_frame, track.end_frame)
    #     plt.plot(frames, track.freqs)

    # plt.show()

    fundamental_idx_b = get_fundamental_idx(freqs_b[b_morph_frame], mags_b[b_morph_frame])
    freq_ratios_b = freqs_b[b_morph_frame] / freqs_b[b_morph_frame, fundamental_idx_b]

    freqs_int = np.zeros((freqs_a.shape[0], max_num_sines_a * max_num_sines_b))
    mags_int = np.zeros_like(freqs_int)

    for frame_idx in range(freqs_a.shape[0]):
        # Get the fundamental frequency of the A sound for this frame
        fundamental_idx_a = get_fundamental_idx(freqs_a[frame_idx], mags_a[frame_idx])
        fundamental_a = freqs_a[frame_idx, fundamental_idx_a]

        # Rescale the b frequencies to have the same fundamental
        rescaled_freqs_b = freq_ratios_b * fundamental_a

        freq_weights = np.zeros((max_num_sines_a, max_num_sines_b))

        for freq_idx_a in range(freqs_a.shape[1]):
            for freq_idx_b in range(rescaled_freqs_b.shape[0]):
                # If either frequency is 0, then don't use it
                if freqs_a[frame_idx, freq_idx_a] == 0 or freqs_b[b_morph_frame, freq_idx_b] == 0:
                    freq_weights[freq_idx_a, freq_idx_b] = 0
                    continue

                # Calculate the frequency weight for this pair of frequencies
                freq_weights[freq_idx_a, freq_idx_b] = np.abs(np.log(freqs_a[frame_idx, freq_idx_a]) - np.log(freqs_b[b_morph_frame, freq_idx_b]))
                # Scale the weight by how close the frequencies are in magnitude
                freq_weights[freq_idx_a, freq_idx_b] *= 10 ** (-1 * np.abs(mags_a[frame_idx, freq_idx_a] - mags_b[b_morph_frame, freq_idx_b]) / 2)

        # Normalize our frequency weights. We want both our rows and columns
        # to sum to 1. This is likely impossible, so just approximate it by iteratively
        # normalizing
        normalized_freq_weights = freq_weights
        for iteration in range(100):
            # Sum the weights for a given a idx
            freq_weight_a_sums = np.sum(normalized_freq_weights, axis = -1, keepdims = True)
            # Avoid division by 0
            freq_weight_a_sums[freq_weight_a_sums == 0] = 1

            normalized_freq_weights = normalized_freq_weights / freq_weight_a_sums

            # Sum the weights for a given a idx
            freq_weight_b_sums = np.sum(normalized_freq_weights, axis = 0, keepdims = True)
            # Avoid division by 0
            freq_weight_b_sums[freq_weight_b_sums == 0] = 1

            normalized_freq_weights = normalized_freq_weights / freq_weight_b_sums

        for freq_idx_a in range(freqs_a.shape[1]):
            start_freq_idx = max_num_sines_b * freq_idx_a
            end_freq_idx = start_freq_idx + max_num_sines_b
            if freqs_a[frame_idx, freq_idx_a] == 0:
                freqs_int[frame_idx, start_freq_idx:end_freq_idx] = 0
                mags_int[frame_idx, start_freq_idx:end_freq_idx] = -200
                continue

            freqs_int[frame_idx, start_freq_idx:end_freq_idx] = 2 ** (np.log2(freqs_a[frame_idx, freq_idx_a]) * (1 - morph_pct[frame_idx]) + np.log2(freqs_b[b_morph_frame]) * morph_pct[frame_idx]) # freqs_b[b_morph_frame]
            freqs_int[frame_idx, start_freq_idx:end_freq_idx][freqs_b[b_morph_frame] == 0] = 0
            mags_int[frame_idx, start_freq_idx:end_freq_idx] = mags_a[frame_idx, freq_idx_a] * (1 - morph_pct[frame_idx]) + mags_b[b_morph_frame] * morph_pct[frame_idx] + 20 * np.log10(normalized_freq_weights[freq_idx_a]) # 20 * np.log10(amps_a[frame_idx, freq_idx_a] * normalized_freq_weights[freq_idx_a]) # 20 * np.log10(amps_b[b_morph_frame] * np.sum(amps_a[frame_idx] ** 2) / np.sum(amps_b[b_morph_frame] ** 2))
            mags_int[frame_idx, start_freq_idx:end_freq_idx][freqs_b[b_morph_frame] == 0] = 0
        
    # output = sm.sineModelSynth(freqs_a, mags_a, np.array([]), frame_size_a, hop_size_a, sample_rate_a)
    output = sm.sineModelSynth(freqs_int, mags_int, np.array([]), frame_size_a, hop_size_a, sample_rate_a)

    sf.write("test.wav", output, sample_rate_a)

def get_fundamental_idx(freqs, mags):
    # For now, our algorithm for finding the fundamental frequency will be the naive
    # approach of just choosing the loudest freq
    return np.argmax(mags)

if __name__ == '__main__':
    main()