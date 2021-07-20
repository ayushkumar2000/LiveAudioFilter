import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
import struct
import scipy
import math
import sys
def record():
    chunk = 1024 
    sample_format = pyaudio.paInt16 
    chanels = 1
    smpl_rt = 44100
    seconds = 4
    filename = "audio_files/recording1.wav"
    pa = pyaudio.PyAudio() 
    stream = pa.open(format=sample_format, channels=chanels,
                    rate=smpl_rt, input=True,
                    frames_per_buffer=chunk)
    
    print('Recording...')
    frames = []
    for i in range(0, int(smpl_rt / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    pa.terminate()
    fr2 = frames
    print('Done !!! ')
    sf = wave.open(filename, 'wb')
    sf.setnchannels(chanels)
    sf.setsampwidth(pa.get_sample_size(sample_format))
    sf.setframerate(smpl_rt)
    sf.writeframes(b''.join(frames))
    sf.close()
    return fr2
def record2():
    chunk = 1 
    sample_format = pyaudio.paInt8
    chanels = 1
    smpl_rt = 44100
    seconds = 2
    filename = "audio_files/recording1.wav"
    pa = pyaudio.PyAudio() 
    stream = pa.open(format=sample_format, channels=chanels,
                    rate=smpl_rt, input=True)
    
    print('Recording...')
    frames = []
    for i in range(0, int(smpl_rt / chunk * seconds)):
        data = stream.read(1)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    pa.terminate()
    fr2 = frames
    print('Done !!! ')
    sf = wave.open(filename, 'wb')
    sf.setnchannels(chanels)
    sf.setsampwidth(pa.get_sample_size(sample_format))
    sf.setframerate(smpl_rt)
    sf.writeframes(b''.join(frames))
    sf.close()
    frames = np.array(frames)
    np.save('frames/frames1.npy',frames)
    return fr2

def play(filename):
    chunk = 1024 
    af = wave.open(filename, 'rb')
    pa = pyaudio.PyAudio()
    stream = pa.open(format = pa.get_format_from_width(af.getsampwidth()),
                    channels = af.getnchannels(),
                    rate = af.getframerate(),
                    output = True)
    rd_data = af.readframes(chunk)
    frames_tot=[]
    for i in range(0,70):
        # stream.write(rd_data)
        rd_data = af.readframes(chunk)
        frames_tot.append(rd_data)
    stream.stop_stream()
    stream.close()
    pa.terminate()
    # print(len(b''.join(frames_tot)))
    return b''.join(frames_tot)

def play_frames(frames): 
    pa = pyaudio.PyAudio()
    stream = pa.open(format = pyaudio.paInt16,
                    channels = 1,
                    rate = 44100,
                    output = True)
    for i in frames:
        stream.write(i)
    stream.stop_stream()
    stream.close()
    pa.terminate()
def play_fr(frames): 
    pa = pyaudio.PyAudio()
    stream = pa.open(format = pyaudio.paInt16,
                    channels = 1,
                    rate = 44100,
                    output = True)
    
    stream.write(frames)
    stream.stop_stream()
    stream.close()
    pa.terminate()    

def window_avg(arr,size):
    res = []
    curr = 0
    while(curr<len(arr)):
        res.append(sum(arr[curr:curr+size])/len(arr[curr:curr+size]))
        curr+=1
    return res
def np2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

def run_live_filter():
    chunk = 1024 
    sample_format = pyaudio.paInt16 
    chanels = 1
    smpl_rt = 44100
    seconds = 4
    filename = "audio_files/recording1.wav"
    pa = pyaudio.PyAudio() 
    stream1 = pa.open(format=sample_format, channels=chanels,
                    rate=smpl_rt, input=True,
                    frames_per_buffer=chunk)
    stream2 = pa.open(format = pyaudio.paInt16,
                    channels = 1,
                    rate = 44100,
                    output = True)
    
    print('Recording...')
    frames = []
    for i in range(0, int(smpl_rt / chunk * seconds)):
        data = stream1.read(chunk*100)
        #process data to the integer
        final_arr = []
        for i in data:
            final_arr.append(int(i))
        #do the fft and other operations on the final_arr
        final_arr1 = np.array(final_arr)
        plt.plot(np.arange(final_arr1.shape[-1]),final_arr1.real,np.arange(final_arr1.shape[-1]),final_arr1.imag)
        plt.show()
        final_arr_fft = np.fft.rfft(final_arr)
        plt.plot(np.arange(final_arr_fft.shape[-1]),final_arr_fft.real,np.arange(final_arr_fft.shape[-1]),final_arr_fft.imag)
        plt.show()
        low_pass = np.concatenate( (np.zeros((5,)), np.ones((final_arr_fft.shape[-1]-5,))))
        
        final_arr2 = np.fft.irfft(final_arr_fft*low_pass).round().astype(np.int16)
        plt.plot(np.arange(final_arr2.shape[-1]),final_arr2.real,np.arange(final_arr2.shape[-1]),final_arr2.imag)
        plt.show()
        mx, mn = np.max(final_arr2), np.min(final_arr2)
        final_arr2 = (final_arr2 - mn) // (mx-mn)
        final_arr2 *= 255
        # # final_arr2 += np.abs(np.min(final_arr2))
        # final_arr2 = [max(min(i,255),0) for i in final_arr2]
        #pack back to frames to play it    
        bin_final = bytes(list(final_arr2))
        stream2.write(bin_final)
        

    stream1.stop_stream()
    stream1.close()
    stream2.stop_stream()
    stream2.close()
    pa.terminate()

def run_file_filter():
    pa = pyaudio.PyAudio() 
    fs = 22050
    stream = pa.open(format = pyaudio.paInt16,
                    channels = 1,
                    rate = 22050,
                    output = True)
    for k in range(0, 1):
        #process data to the integer samples load it into final_arr
        final_arr = [int(i) for i in play('audio_files/CantinaBand3.wav')]
        print('finished reading from file')
        
        # #first making the signal zero mean
        mean_final_arr = sum(final_arr)/len(final_arr)
        final_arr = [i - mean_final_arr for i in final_arr]


        #plot the actual signal
        final_arr1 = np.array(final_arr)
        # plt.plot(np.arange(final_arr1.shape[-1]),final_arr1.real,np.arange(final_arr1.shape[-1]),final_arr1.imag)
        # plt.show()
        
        
        
        #do the fft and other operations on the final_arr
        
        final_arr_fft = np.fft.rfft(final_arr,np2(len(final_arr)))
        # final_arr_fft_shifted = np.fft.fftshift(final_arr_fft)
        freqs = np.fft.fftfreq(n=len(final_arr_fft),d = 1/(fs))
        final_arr_fft_pos =  final_arr_fft[:int(len(final_arr_fft)/2)+1]

        #the fft results
        # plt.plot(freqs,np.abs(final_arr_fft_shifted))
        # plt.show()
        plt.plot(freqs[freqs>=0],np.abs(final_arr_fft_pos))
        plt.show()


        #filters
        low_pass = np.concatenate( (np.ones((len(final_arr_fft)//4,)), np.zeros((final_arr_fft.shape[-1]-len(final_arr_fft)//4,))))
        high_pass = np.concatenate( (np.zeros((1000,)), np.ones((final_arr_fft.shape[-1]-1000,))))

        # plt.plot(freqs[freqs>=0],np.abs(final_arr_fft_pos*low_pass))
        # plt.show()
        
        # low_pass = np.concatenate( (np.ones((50000,)), np.zeros((final_arr_fft.shape[-1]-50000,))))
        
        final_arr2 = np.fft.irfft(final_arr_fft*low_pass).round().astype(np.int16)

        plt.plot(np.arange(final_arr2.shape[-1]),final_arr2.real,np.arange(final_arr2.shape[-1]),final_arr2.imag)
        plt.show()

        #adjusting result of filter to make it between (0,255)
        final_arr2 += np.abs(np.min(final_arr2))
        mx, mn = np.max(final_arr2), np.min(final_arr2)
        final_arr2 = (final_arr2)/mx
        final_arr2 *=255
        final_arr2 = final_arr2.round().astype(np.int16)
        plt.plot(np.arange(final_arr2.shape[-1]),final_arr2.real,np.arange(final_arr2.shape[-1]),final_arr2.imag)
        plt.show()
        # final_arr2 = np.clip(final_arr2,0,255)
        
        
        #pack back to frames to play it    
        bin_final = bytes(list(final_arr2))
        stream.write(bin_final)
        
    stream.stop_stream()
    stream.close()
    pa.terminate()




if(__name__ == "__main__"): 
    # record()
    run_file_filter()
