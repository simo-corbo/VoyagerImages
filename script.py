import scipy
import numpy as np
import sys
sys.setrecursionlimit(2**30)

def load_wav(file):
    return scipy.io.wavfile.read(file)



def main():
    sample_rate, data = load_wav('resources/voyager_images_double.wav')
    # there are 2 channels in the audio file
    # the images are in both channels

    for channel in data.transpose():
        # normalise the data
        channel = channel / max(abs(channel))
        # extract the first 15 seconds of the audio
        first_30s = channel[:sample_rate * 30]
        # Find the end of the audio start tone.
        start_tone, _ = scipy.signal.find_peaks(-first_30s, height = np.max(-first_30s) - 0.2)
        



if __name__ == '__main__':
    
    main()