import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
import struct
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
    for i in range(0,10000):
        # stream.write(rd_data)
        rd_data = af.readframes(chunk)
        frames_tot.append(rd_data)
    stream.stop_stream()
    stream.close()
    pa.terminate()
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

def run_live_filter2():
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
                    rate = 22050,
                    output = True)
    
    print('Recording...')
    frames = []
    for i in range(0, 1):
        #process data to the integer
        final_arr = [int(i) for i in play('audio_files/CantinaBand3.wav')]
        print('finished reading')
        #do the fft and other operations on the final_arr
        final_arr1 = np.array(final_arr)
        plt.plot(np.arange(final_arr1.shape[-1]),final_arr1.real,np.arange(final_arr1.shape[-1]),final_arr1.imag)
        plt.show()
        final_arr_fft = np.fft.rfft(final_arr)
        plt.plot(np.arange(final_arr_fft.shape[-1]),final_arr_fft.real,np.arange(final_arr_fft.shape[-1]),final_arr_fft.imag)
        plt.show()
        low_pass = np.concatenate( (np.ones((10000,)), np.zeros((final_arr_fft.shape[-1]-10000,))))
        
        final_arr2 = np.fft.irfft(final_arr_fft*low_pass).round().astype(np.int16)
        plt.plot(np.arange(final_arr2.shape[-1]),final_arr2.real,np.arange(final_arr2.shape[-1]),final_arr2.imag)
        plt.show()
        # final_arr2 += np.abs(np.min(final_arr2))
        mx, mn = np.max(final_arr2), np.min(final_arr2)
        final_arr2 = (final_arr2 - mn) #// (mx-mn)
        final_arr2 = np.clip(final_arr2,0,255)
        #pack back to frames to play it    
        bin_final = bytes(list(final_arr2))
        stream2.write(bin_final)
        

    stream1.stop_stream()
    stream1.close()
    stream2.stop_stream()
    stream2.close()
    pa.terminate()




if(__name__ == "__main__"): 
    # record()
    run_live_filter2()
    sys.exit(0)
    
    frames1 = record()
    linear_arr = b''.join(frames1)
    count1 = 0
    # final_arr = window_avg(linear_arr,int(10))
    # mean_arr = sum(linear_arr)/len(linear_arr)
    #print(linear_arr,file = open('testfile2.txt','w'))
    final_arr = []
    for i in linear_arr:
        final_arr.append(int(i))
    # final_arr = [i-mean_arr for i in linear_arr]
    
    plt.plot(final_arr)
    plt.show()
    # plt.plot(np.fft.fft(final_arr))
    # plt.show()
    print(len(final_arr))
    #pack back to frames to play it    
    bin_final = bytes(final_arr)
    
    # for val in final_arr:
    #     bin_final += struct.pack('<H',val)
    play_fr(bin_final)




# play_frames(frames1)