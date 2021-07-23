import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
import struct
import scipy
import math
from scipy.io import wavfile as wav
import freq_gen as fg
import filter as ft

def live_run(seconds):
    chunk = int(220500/2)
    sample_format = pyaudio.paInt16 
    chanels = 1
    fs = 22050
    filename = "audio_files/recording1.wav"
    pa = pyaudio.PyAudio() 
    stream1 = pa.open(format=sample_format, channels=chanels,
                    rate=fs, input=True,frames_per_buffer=chunk)
    stream2 = pa.open(format = pyaudio.paInt16,
                    channels = 1,
                    rate = fs,
                    output = True)
    
    print('Recording...')

    for i in range(seconds*10):
        data = stream1.read(chunk)
        #process data to the integer
        samples = []
        for i in data:
            samples.append(int(i))
        samples = np.array(samples).astype(np.int16)
        filtered_samples = ft.butterworth(samples,4,0.1)
        filtered_samples = fg.scale_255(filtered_samples)
        fg.play(filtered_samples,fs)
        # bin_final = bytes(list(filtered_samples))      
        # stream2.write(bin_final)
        
    stream1.stop_stream()
    stream1.close()
    stream2.stop_stream()
    stream2.close()
    pa.terminate()

live_run(10)