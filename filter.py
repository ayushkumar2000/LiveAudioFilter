'''Contains functions to apply filters to vector'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def butterworth(samples,order,cutoff,plot_g=False):
    b,a = signal.butter(order,cutoff)
    filtered_samples = signal.lfilter(b,a,samples)
    if(plot_g):
        plt.plot(samples[:1000])
        plt.plot(filtered_samples[:1000])
        plt.show()
    return filtered_samples
def genfilter(samples,zeros,poles):
    pass

if( __name__ == '__main__'):
    #visualize any filter using the impulse response for a dirac delta
    fs = 10000 #sampling rate = 10000hz
    samples = np.concatenate((np.ones(1),np.zeros(1000)))

    #give the filter
    filtered_samples = butterworth(samples,2,0.1)
    
    #generate fft and plot the graph
    filtered_samples_fft = np.fft.fft(filtered_samples)
    filtered_samples_freqs = np.fft.fftfreq(len(filtered_samples),d = (1/fs))
    plt.plot(filtered_samples_freqs,abs(filtered_samples_fft))
    plt.show()

    






