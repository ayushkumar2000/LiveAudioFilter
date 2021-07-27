from matplotlib.colors import same_color
import pyaudio
import numpy as np
import math
import wave
import filter as ft
import matplotlib.pyplot as plt

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
    



def file_samples(filename): 
    rate, data = wav.read(filename)
    return rate,data



if(__name__ == "__main__"):
    fs = 22050
    freqs = np.fft.fftfreq(n=len(f_gen([200,400,2000,2300],fs)),d = 1/(fs))
    samples =f_gen([200],fs) 
    noise = 255*np.random.normal(0, .2, samples.shape)
    sample_noise = samples+noise
    # fs,samples = file_samples('audio_files/CantinaBand3.wav')
    plt.plot(sample_noise[:1000])
    plt.show()
    plt.plot(freqs, abs(np.fft.fft(sample_noise)))
    plt.show()
    samples_filtered = ft.low_pass_filter(sample_noise,500,fs)
    plt.plot(samples_filtered[:1000])
    plt.show()
    plt.plot(freqs, abs(np.fft.fft(samples_filtered)))
    plt.show()
    
    
    