import numpy as np

class SineTrack():
    def __init__(self, start_frame, end_frame, freqs, mags):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.freqs = freqs
        self.mags = mags

    @staticmethod
    def extract_tracks(freqs, mags):
        tracks = list()
        for freq_index in range(freqs.shape[1]):
            start_frame = 0
            track_freqs = list()
            track_mags = list()
            cur_freq = 0
            for frame in range(freqs.shape[0]):
                if freqs[frame, freq_index] != 0:
                    if cur_freq == 0:
                        track_freqs = list()
                        track_mags = list()
                        start_frame = frame

                    cur_freq = freqs[frame, freq_index]
                    track_freqs.append(cur_freq)
                    track_mags.append(mags[frame, freq_index])
                else:
                    if cur_freq != 0:
                        tracks.append(SineTrack(start_frame, frame, track_freqs, track_mags))
                        cur_freq = 0

        return tracks
    
    @staticmethod
    def track_list_to_numpy(track_list):
        max_frame = 0
        for track in track_list:
            max_frame = max(max_frame, track.end_frame)

        num_tracks = len(track_list)
        output_freqs = np.zeros((max_frame, num_tracks))
        output_mags = np.zeros((max_frame, num_tracks))

        for track_idx, track in enumerate(track_list):
            output_freqs[track.start_frame, track.end_frame, track_idx] = track.freqs
            output_mags[track.start_frame, track.end_frame, track_idx] = track.mags

        return output_freqs, output_mags
