import numpy as np
import soundfile as sf
import scipy.signal
from smstools.models import sineModel as sm
import matplotlib.pyplot as plt

def main():
    file_path_a = '/Users/mkmarchan/Downloads/ferry_crop.wav'
    data_a, sample_rate_a = sf.read(file_path_a)

    file_path_b = '/Users/mkmarchan/Downloads/common_loon_crop.wav'
    data_b, sample_rate_b = sf.read(file_path_b)

    frame_size = 4096
    hop_size = 1024
    window = scipy.signal.windows.dpss(frame_size, 5 * np.pi)
    sine_threshold_db = -60
    max_num_sines = 100
    min_sine_dur = 0.01

    freqs_a, mags_a, _ = sm.sineModelAnal(
        data_a,
        sample_rate_a,
        window,
        frame_size,
        hop_size,
        sine_threshold_db,
        max_num_sines,
        min_sine_dur,
        )
    
    amps_a = 10 ** (mags_a / 20)
    
    freqs_b, mags_b, _ = sm.sineModelAnal(
        data_b,
        sample_rate_b,
        window,
        frame_size,
        hop_size,
        sine_threshold_db,
        max_num_sines,
        min_sine_dur)
    
    amps_b = 10 ** (mags_b / 20)

    # Stuff good values into our amplitudes and magnitudes
    amps_a[freqs_a == 0] = 0
    mags_a[freqs_a == 0] = -200
    amps_b[freqs_b == 0] = 0
    mags_b[freqs_b == 0] = -200
            
    # interpolate
    shortest_num_frames = min(freqs_a.shape[0], freqs_b.shape[0])
    weighting = (1 - (np.arange(shortest_num_frames) / (shortest_num_frames - 1))) ** 0.25
    # weighting = weighting.reshape((-1, 1))
    # freqs_int = freqs_a[:shortest_num_frames] * (1 - weighting) + freqs_b[:shortest_num_frames] * weighting
    # mags_int = mags_a[:shortest_num_frames] * (1 - weighting) + mags_b[:shortest_num_frames] * weighting

    freqs_int = np.zeros((shortest_num_frames, max_num_sines ** 2))
    amps_int = np.zeros((shortest_num_frames, max_num_sines ** 2))
    mags_int = np.zeros((shortest_num_frames, max_num_sines ** 2))

    for frame in range(shortest_num_frames):
        freq_weights = np.zeros((max_num_sines, max_num_sines))
        for a_index in range(max_num_sines):
            for b_index in range(max_num_sines):
                # If either frequency is 0, then don't use it
                if freqs_a[frame, a_index] == 0 or freqs_b[frame, b_index] == 0:
                    freq_weights[a_index, b_index] = 0
                    continue

                # Calculate the frequency weight for this pair of frequencies
                freq_weights[a_index, b_index] = np.abs(np.log(freqs_a[frame, a_index]) - np.log(freqs_b[frame, b_index]))
                # Scale the weight by how close the frequencies are in magnitude
                freq_weights[a_index, b_index] *= 10 ** (-1 * np.abs(mags_a[frame, a_index] - mags_b[frame, b_index]) / 2)
        
        squared_freq_weights = freq_weights ** 2
        for a_index in range(max_num_sines):
            freqs_start_idx = a_index * max_num_sines
            freqs_end_idx = freqs_start_idx + max_num_sines
            # Square the weights because we want to normalize based on energy, not amplitude
            squared_weight_sum = np.sum(squared_freq_weights[a_index])

            if squared_weight_sum == 0:
                # Don't interpolate if this doesn't map to any other frequency.
                # Instead, just control its magnitude
                freqs_int[frame, freqs_start_idx] = freqs_a[frame, a_index]
                if freqs_int[frame, freqs_start_idx] == 0:
                    amps_int[frame, freqs_start_idx] = 0
                    continue

                amps_int[frame, freqs_start_idx] = amps_a[frame, a_index] * (1 - weighting[frame])
                continue

            normalized_weights = squared_freq_weights[a_index] / squared_weight_sum
            freqs_int[frame, freqs_start_idx:freqs_end_idx] = freqs_a[frame, a_index] * (1 - weighting[frame]) + freqs_b[frame] * weighting[frame]
            amps_int[frame, freqs_start_idx:freqs_end_idx] = amps_a[frame, a_index] * normalized_weights * (1 - weighting[frame]) + amps_b[frame] * normalized_weights * weighting[frame]

        # Now that we've distributed our frequencies from a to b,
        # we need to now re-normalize such that we maintain the relative amplitude relationships
        # between b's frequencies based on the current morph weighting
        # TODO: Need to do two things
        #   1. Overall energy of the frame should match the interpolated energy between A and B
        #   2. The relative energy ratios of the b freqs should match
        for b_index in range(max_num_sines):
            amps_b_int = amps_int[frame, b_index::max_num_sines]
            
            b_freq_scalars = amps_b[frame] / amps_b_int
            b_freq_scalars[amps_b_int == 0] = 0

            # Interpolate between not using the scalars and fully using the scalars
            b_freq_scalars = (1 - weighting[frame]) + b_freq_scalars * weighting[frame]
            amps_int[frame, b_index::max_num_sines] = b_freq_scalars * amps_b_int

        amp_sum_a = np.sum(amps_a[frame])
        amp_sum_b = np.sum(amps_b[frame])
        amp_sum_int = np.sum(amps_int[frame])
        target_amp = amp_sum_a * (1 - weighting[frame]) + amp_sum_b * weighting[frame]

        if amp_sum_int == 0:
            amps_int[frame] = 0
            mags_int[frame] = -200
            continue

        energy_scalar = target_amp / amp_sum_int
        amps_int[frame] *= energy_scalar
        mags_int[frame] = 20 * np.log10(amps_int[frame])


    # a_tracks = extract_tracks(freqs_a, mags_a)
    # b_tracks = extract_tracks(freqs_b, mags_b)

    # cost_matrix = np.zeros((len(a_tracks), len(b_tracks)))
    # for a_index, a_track in enumerate(a_tracks):
    #     for b_index, b_track in enumerate(b_tracks):
    #         print("Hello")

    # plt.plot(freqs_a)
    # plt.show()

    # plt.plot(freqs_b[:, :10])
    # plt.show()

    # plt.plot(amps_a[:, 0])
    # plt.plot(np.sum(amps_int[:, :max_num_sines], axis = -1))
    # plt.show()

    # plt.plot(mags_a[:, 0])
    # plt.plot(20 * np.log10(np.sum(amps_int[:, :max_num_sines], axis = -1)))
    # plt.show()
    
    # plt.plot(freqs_a[:, 0])
    # # plt.plot(freqs_b[:, 0])
    # plt.plot(freqs_int[:, 0])
    # plt.show()

    # plt.plot(mags_a[:, 0])
    # # plt.plot(freqs_b[:, 0])
    # plt.plot(mags_int[:, 0])
    # plt.show()

    # TODO: there is jittering here because of the sines having destructive interference.
    #       Tried to compensate by normalizing based on energy but that only helps a bit
    output = sm.sineModelSynth(freqs_int, mags_int, np.array([]), frame_size, hop_size, sample_rate_a)

    sf.write("test.wav", output, sample_rate_a)

if __name__ == '__main__':
    main()