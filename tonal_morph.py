import numpy as np
import soundfile as sf
import scipy.signal
from smstools.models import sineModel as sm
import matplotlib.pyplot as plt

from sinetracks import SineTrack

def main():
    file_path = '/Users/mkmarchan/Downloads/crow_crop_long.wav'
    data, sample_rate = sf.read(file_path)

    # Analysis settings a
    frame_size = 1024
    hop_size = int(frame_size / 4)
    window = scipy.signal.windows.dpss(frame_size, 5 * np.pi)
    sine_threshold_db = -70
    max_num_sines = 200
    min_sine_dur = 0.01
    freq_dev_offset = 200
    freq_dev_slope = 0.01
    min_freq = 200

    freqs, mags, _ = sm.sineModelAnal(
        data,
        sample_rate,
        window,
        frame_size,
        hop_size,
        sine_threshold_db,
        max_num_sines,
        min_sine_dur,
        freq_dev_offset,
        freq_dev_slope,
        )
    
    freqs[freqs < min_freq] = 0
    
    amps = 10 ** (mags / 20)

    # Stuff good values into our amplitudes and magnitudes
    amps[freqs == 0] = 0
    mags[freqs == 0] = -200

    tracks = SineTrack.extract_tracks(freqs, mags)

    # # Plot a_tracks
    # for track in tracks:
    #     frames = np.arange(track.start_frame, track.end_frame)
    #     plt.plot(frames, track.freqs)

    # Morph settings
    morph_pct = np.minimum(1, np.arange(freqs.shape[0]) / (freqs.shape[0] / 2))
    # morph_pct = np.arange(freqs.shape[0]) / (freqs.shape[0] - 1)
    # morph_pct = np.ones(freqs.shape[0])

    # plt.show()

    for octave in list([0.5]):
        for semitone in list([0, 4, 7, 11, 12, 14, 21, 24, 31]):
            # Target settings
            target_reference_freq = 440
            target_semitones = np.array([semitone])
            # target_semitones = np.concatenate([target_semitones, target_semitones + 24, target_semitones + 48])
            target_freqs = target_reference_freq * (2 ** (target_semitones / 12))
            target_freqs = np.concatenate(target_freqs[np.newaxis, :] * np.arange(1, 25)[:, np.newaxis])

            freqs_int = np.zeros_like(freqs)

            for frame_idx in range(freqs.shape[0]):
                target_freq_dists = np.abs(freqs[frame_idx, :, np.newaxis] - target_freqs[np.newaxis, :])
                nearest_target_freq_idx = np.argmin(target_freq_dists, axis = -1)
                log_freq_int = np.log2(freqs[frame_idx]) * (1 - morph_pct[frame_idx]) + np.log2(target_freqs[nearest_target_freq_idx] * octave) * morph_pct[frame_idx]
                freqs_int[frame_idx] = 2 ** (log_freq_int)
                freqs_int[frame_idx, freqs[frame_idx] == 0] = 0
                
            # output = sm.sineModelSynth(freqs, mags, np.array([]), frame_size, hop_size, sample_rate)
            output = sm.sineModelSynth(freqs_int, mags, np.array([]), frame_size, hop_size, sample_rate)

            sf.write(f"test_{octave}_{semitone}.wav", output, sample_rate)



if __name__ == '__main__':
    main()