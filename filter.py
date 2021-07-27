'''Contains functions to apply filters to vector'''
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from sympy.utilities.lambdify import lambdify

z = Symbol('z')

def filter_finder(fs):
    '''Runs the filter based on user needs'''
    #find out: 1 - lowpass(), 2-highpass(), 3-lccde coeffs, 4 - poles,zeros genfilter() 
    
    #first get the required params for these filters
    #second get visualize the filter
    #third check if we want to save the file or not
    #run live_filter

    print('Enter type of filter you want 1. Low pass 2. High pass 3. lccde coefficients 4. Poles Zeros')
    filter_num = int(input())
    arg_list = []
    if(filter_num == 1):
        print('Enter cutoff')
        arg_list.append(float(input()))
        arg_list.append(fs)
    if(filter_num == 2):
        print('Enter cutoff')
        arg_list.append(float(input()))
        arg_list.append(fs)
    if(filter_num == 3):
        print('Enter numerator coefficients separated by space (b[0] * z^0 + b[1]*z^-1 ....)')
        num = [float(i) for i in input().split(' ')]
        print('Enter denominator separated by space (a[0] * z^0 + a[1]*z^-1 ....)')
        den = [float(i) for i in input().split(' ')] 
        arg_list.append(num)
        arg_list.append(den)
    
    if(filter_num == 4):
        print('Enter zeros separated by space')
        zeros = [float(i) for i in input().split(' ')]
        print('Enter poles separated by space')
        poles = [float(i) for i in input().split(' ')] 
        arg_list.append(zeros)
        arg_list.append(poles)
    
    return filter_num,arg_list 
    
        
        



def filter_runner(samples,filter_num,arg_list):
    '''this runs the filters based on the number'''
    if(filter_num == 1):
        return low_pass_filter(samples,arg_list[0],arg_list[1])
    if(filter_num == 2):
        return high_pass_filter(samples,arg_list[0],arg_list[1])
    if(filter_num == 3):
        return signal_filter(samples,arg_list[0],arg_list[1])
    if(filter_num == 4):
        return genfilter(samples,arg_list[0],arg_list[1])






def low_pass_filter(samples,fc,fs,plot_g = False):
    '''This runs a low pass filter based on cutoff frequency fc'''
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
def high_pass_filter(samples,fc,fs,plot_g = False):
    '''This runs a highpass filter based on cutoff frequency fc'''

    freqs = np.fft.fftfreq(n=len(samples),d = 1/(fs))

    samples_fft = np.fft.fft(samples)
    
    low_pass_samples_fft = []
    for i in range(0,len(samples_fft)):
        if(abs(freqs[i])>=fc):
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


def genfilter(samples,zeros,poles,plot_g=False):
    '''This runs the pole-zero filter'''
    zb = Symbol('z')
    whole_b =1
    for root in zeros:
        whole_b *=(zb-root)

    whole_a =1
    for root in poles:
        whole_a *=(zb-root)
    if(plot_g):
        print('b(x) =',whole_b.expand())
        print('a(x) =',whole_a.expand())
    b = poly(whole_b).all_coeffs()
    a = poly(whole_a).all_coeffs()

    return signal_filter(samples,b,a)


def evalHz(Hz,w):
    func = lambdify(z,Hz,'numpy')
    return func(np.exp(-1*1j*w))
def signal_filter(samples,b,a):
    '''This is generic implementation of lfilter using fft'''
    whole_b =0
    for i in range(0,len(b)):
        whole_b += (z**i)*b[i]
    whole_a =0
    for i in range(0,len(a)):
        whole_a += (z**i)*a[i]
    whole_exp = whole_b/whole_a
    samples_fft = np.fft.fft(samples)
    N = len(samples_fft)
    #evaluate the H[k] by putting w = 2*pi*k/N in H(e^jw)
    H = evalHz(whole_exp,2* np.pi * np.array(range(0,N))/N)
    #applying the filter by doing samples_fft X H
    filtered_samples = samples_fft * H
    return np.fft.ifft(filtered_samples).real
    
def visualize_filter(filer_num,arg_list,fs = 50000):
       #visualize any filter using the impulse response for a dirac delta
    samples = np.concatenate((np.ones(1),np.zeros(1000)))

    #give the filter
    # filtered_samples = butterworth(samples,10,0.1)
    filtered_samples = filter_runner(samples,filer_num,arg_list)
    
    #generate fft and plot the graph
    filtered_samples_fft = np.fft.fft(filtered_samples)
    filtered_samples_freqs = np.fft.fftfreq(len(filtered_samples),d = (1/fs))
    plt.plot(filtered_samples_freqs,abs(filtered_samples_fft))
    plt.xlabel('Frequency in Hz')
    plt.ylabel('|H(z)|')
    plt.show()




if( __name__ == '__main__'):
    # print(signal_filter([1,0,0,0,0,0,0,0,0,0,0,0],[1,-1],[1,-0.5]))
    visualize_filter(2,[[1],[0.5]])




