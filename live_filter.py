import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
import struct
import scipy
from math import ceil
from scipy.io import wavfile as wav
import freq_gen as fg
import filter as ft

def live_run(seconds,save_file = False):
    chunk = int(220500/2)
    sample_format = pyaudio.paInt16 
    chanels = 1
    fs = 22050
    filename = "audio_files/recording1.wav"
    #first find the filter to run
    filter_num,arg_list = ft.filter_finder()
    #visualize the filter
    ft.visualize_filter(filter_num,arg_list)
    pa = pyaudio.PyAudio() 
    stream1 = pa.open(format=sample_format, channels=chanels,
                    rate=fs, input=True,frames_per_buffer=chunk)
    stream2 = pa.open(format = pyaudio.paInt16,
                    channels = 1,
                    rate = fs,
                    output = True)
    tot_samples = np.array([])
    tot_filtered_samples = np.array([])

    

    print('Recording...')
    for i in range(int(ceil(seconds/5))):
        data = stream1.read(chunk)
        #process data to the integer
        samples = []

        samples = np.frombuffer(data, dtype='int16')
                
        if(not save_file):
            filtered_samples = ft.filter_runner(samples,filter_num,arg_list)
            filtered_samples = filtered_samples.astype(np.int16)
            stream2.write(filtered_samples.tobytes())
        else:
            tot_samples = np.append(tot_samples,samples)

    
    if(save_file):
        print("Recording completed")
        print("Applying filter ....")
        tot_filtered_samples = ft.filter_runner(tot_samples,filter_num,arg_list)
        fg.play(tot_samples,fs,True,'audio_files/original.wav')
        fg.play(tot_filtered_samples,fs,True,'audio_files/filtered.wav')
        print("Done")



    stream1.stop_stream()
    stream1.close()
    stream2.stop_stream()
    stream2.close()
    pa.terminate()



# def record(seconds):

save_file = False
save_file = input('Enter "yes" to save the recording and filtered recording or "no" to run a live filter\n') == 'yes'
print(save_file)
live_run(10,save_file)