'''Contains functions to apply filters to vector'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sympy import *


def filter_finder():
    '''Runs the filter based on user needs'''
    #find out: 1 - butterworth(), 2 - lowpass(), 3 -highpass(), 4 - lccde coeffs, 5 - poles,zeros genfilter() 
    
    #first get the required params for these filters
    #second get visualize the filter
    #third check if we want to save the file or not
    #run live_filter

    print('Enter type of filter you want 1. Butterworth 2. Low pass 3. High pass 4. lccde coefficients 5. Poles Zeros')
    filter_num = int(input())
    arg_list = []
    if(filter_num == 1):
        print('Enter order and cutoff separated by space')
        arg_list.append([float(i) for i in input().split(' ')])
    if(filter_num == 2):
        print('Enter cutoff')
        arg_list.append(float(input()))
    if(filter_num == 3):
        print('Enter cutoff')
        arg_list.append(float(input()))
    if(filter_num == 4):
        print('Enter numerator coefficients separated by space')
        arg_list.append(float(input()))
    if(filter_num == 5):
        print('Enter zeros separated by space')
        zeros = [float(i) for i in input().split(' ')]
        print('Enter poles separated by space')
        poles = [float(i) for i in input().split(' ')] 
        arg_list.append(zeros,poles)
    
    return filter_num,arg_list 
    
        
        



def filter_runner(samples,filter_num,arg_list):
    if(filter_num == 1):
        return butterworth(samples,arg_list[0][0],arg_list[0][1])
    if(filter_num == 2):
        return lowpass(samples,arg_list[0])
    if(filter_num == 3):
        return highpass(samples,arg_list[0])
    if(filter_num == 4):
        return lccde(samples,arg_list[0])
    if(filter_num == 5):
        return genfilter(samples,arg_list[0],arg_list[1])





def butterworth(samples,order,cutoff,plot_g=False):
    b,a = signal.butter(order,cutoff)
    filtered_samples = signal.lfilter(b,a,samples)
    if(plot_g):
        plt.plot(samples[:1000])
        plt.plot(filtered_samples[:1000])
        plt.show()
    return filtered_samples
def genfilter(samples,zeros,poles,plot_g=False):
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

    return signal.lfilter(b,a,samples)
    
def visualize_filter(filer_num,arg_list):
       #visualize any filter using the impulse response for a dirac delta
    fs = 10000 #sampling rate = 10000hz
    samples = np.concatenate((np.ones(1),np.zeros(1000)))

    #give the filter
    # filtered_samples = butterworth(samples,10,0.1)
    filtered_samples = filter_runner(samples,filer_num,arg_list)
    
    #generate fft and plot the graph
    filtered_samples_fft = np.fft.fft(filtered_samples)
    filtered_samples_freqs = np.fft.fftfreq(len(filtered_samples),d = (1/fs))
    plt.plot(filtered_samples_freqs,abs(filtered_samples_fft))
    plt.show()




if( __name__ == '__main__'):
    visualize_filter()





