import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft
import numpy as np
rate, data = wav.read('audio_files/CantinaBand3.wav')
fft_out = fft(data)
freqs = np.fft.fftfreq(n=len(data),d = 1/rate)
plt.plot(data, np.abs(fft_out))
plt.show()
plt.plot(freqs,abs(np.fft.fft(data)))
plt.show()
