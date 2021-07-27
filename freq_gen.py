from matplotlib.colors import same_color
import pyaudio
import numpy as np
import math
import wave
import filter as ft
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav

def scale_255(samples):
    samples = samples.astype(np.int16)
    if(np.min(samples) < 0):
        samples -= np.min(samples)
    if(np.max(samples)>255):
        samples = samples/np.max(samples)
        samples *= 255
        samples = samples.round().astype(np.int16)
    return samples



def f_gen(freq,samp_rate):
    seconds = 10
    no_samples = seconds*samp_rate
    time = []
    for i in range(0,no_samples):
        time.append(i/samp_rate)
    final_arr = np.zeros(no_samples)
    time = np.array(time)
    for i in freq:
        final_arr = np.add(final_arr,255*np.sin(time*2*math.pi*i))
    return final_arr





def play(samples,samp_rate,save_file=False,filename = 'audio_files/output.wav'):
    pa = pyaudio.PyAudio()    
    if(not save_file):
        stream = pa.open(format = pyaudio.paInt16,
                    channels = 1,
                    rate = samp_rate,
                    output = True)
        samples = np.array(scale_255(samples))
        bin_final = bytes(list(samples))
        stream.write(bin_final)
        stream.stop_stream()
        stream.close()
        pa.terminate()
    
    else:
        samples = samples.astype(np.int16)
        bin_final = samples.tobytes()
        sf = wave.open(filename, 'wb')
        sf.setnchannels(1)
        sf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        sf.setframerate(samp_rate)
        sf.writeframes(bin_final)
        sf.close()
    


def low_pass_filter(samples,fc,fs,plot_g = False):
    freqs = np.fft.fftfreq(n=len(samples),d = 1/(fs))

    samples_fft = np.fft.fft(samples)
    
    low_pass_samples_fft = []
    for i in range(0,len(samples_fft)):
        if(abs(freqs[i])<=fc):
            low_pass_samples_fft.append(samples_fft[i])
        else:
            low_pass_samples_fft.append(0)
    low_pass_samples_fft = np.array(low_pass_samples_fft)

    if(plot_g):
        plt.plot(freqs,abs(samples_fft))
        plt.show()
        plt.plot(freqs,abs(low_pass_samples_fft))
        plt.show()
    return np.fft.ifft(low_pass_samples_fft).real

def file_samples(filename): 
    rate, data = wav.read(filename)
    return rate,data



if(__name__ == "__main__"):
    fs = 22050
    freqs = np.fft.fftfreq(n=len(f_gen([200,400,2000,2300],fs)),d = 1/(fs))
    samples =f_gen([200,400,2000,2300],fs) 
    fs,samples = file_samples('audio_files/CantinaBand3.wav')
    plt.plot(samples[:1000])
    plt.show()
    
    # low_pass_samples = low_pass_filter(samples,10000,fs,True)
    butter_samples = ft.butterworth(samples,4,0.1)
    # play(low_pass_samples,fs,True)
    # play(butter_samples,fs,True)
    # play(scale_255(f_gen([200,400,2000,2300],22050)),22050)