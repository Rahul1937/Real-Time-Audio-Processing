import pyaudio
import os
import struct
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft
import time


# to display in separate Tk window
#matplotlib.use('TKAgg')

# constants
CHUNK = 1024 * 2             # samples per frame
FORMAT = pyaudio.paInt16     # audio format (bytes per sample?)
CHANNELS = 1                 # single channel for microphone
RATE = 44100                 # samples per second
Fs=RATE

filter_length=512

def IIR_filter_cutoff(cutoff_freq, filter_order=3, filter_type="lowpass"):
    try:
        w_c=2*cutoff_freq/Fs #cutoff freq in rad/s

        [num,den]= signal.butter(int(filter_order), w_c, btype= filter_type)

        z=np.poly1d(num).roots
        p=np.poly1d(den).roots
        #freq response
        [w,H]=signal.freqz(num, den, worN=filter_length)
        w=Fs*(w)/(2*np.pi)

        h=np.real(np.fft.ifft(H))               #impulse response

        #filtered = signal.convolve(input_signal, h, mode='same')

        return (h,H,w,z,p)
    except:
        print("Invalid Input")

def IIR_filter_zpk(zeros, poles, gain=1):
    try:
        #freq response
        [w,H]=signal.freqz_zpk(zeros, poles, gain, worN=filter_length)
        w=Fs*(w)/(2*np.pi)
        
        h=np.real(np.fft.ifft(H))               #impulse response

        #filtered = signal.convolve(input_signal, h, mode='same')

        return (h,H,w,zeros,poles)
    except:
        print("invalid input")

def IIR_filter_rational(numerator, denominator=[1]):
    #freq response
    try:
        [w,H]=signal.freqz(numerator, denominator, worN=filter_length)
        w=Fs*(w)/(2*np.pi)
        z=np.poly1d(numerator).roots
        p=np.poly1d(denominator).roots
        h=np.real(np.fft.ifft(H))               #impulse response

        #filtered = signal.convolve(input_signal, h, mode='same')

        return (h,H,w,z,p)
    except:
        print("invalid input")

def FIR_filter_cutoff(cutoff_freq, num_taps, window_type="hamming", filter_type="lowpass"):
    try:
        w_c=2*cutoff_freq/Fs #cutoff freq in rad/s

        t=signal.firwin(int(num_taps),w_c,window=window_type, pass_zero=filter_type) #taps
        z=np.poly1d(t).roots
        p=[]
        #freq response
        [w,H]=signal.freqz(t, worN=filter_length)
        w=Fs*(w)/(2*np.pi)

        h=np.real(np.fft.ifft(H))               #impulse response

        #filtered = signal.convolve(input_signal, h, mode='same')

        return (h,H,w,z,p)
    except:
        print("invalid input")

def FIR_filter_coefficients(coeff):
    try:
        #freq response
        [w,H]=signal.freqz(coeff, worN=filter_length)
        w=Fs*(w)/(2*np.pi)
        z=np.poly1d(coeff).roots
        p=[]
        h=np.real(np.fft.ifft(H))               #impulse response

        #filtered = signal.convolve(input_signal, h, mode='same')

        return (h,H,w,z,p)
    except:
        print("invalid input")

def filter(input_signal, filter_impulse_response):
    try:
        filtered = np.convolve(input_signal, filter_impulse_response, mode='same')
        return filtered
    except:
        print("Not able to filter")
        out=np.zeros(len(input_signal))
        return out

def plot_filter(h,H,w,z,p):
    #plot_filter_impulse_response(h)
    #plot_filter_mag_response(H,w)
    #plot_filter_phase_response(H,w)
    
    fig1, ax=plt.subplots(2,2)
    n=np.linspace(0,len(h),filter_length)
    ax[0,0].plot(n,h)
    ax[0,0].set_title('Impulse Response')
    ax[0,0].set_xlabel('Samples')
    ax[0,0].set_ylabel('Response')

    w_r=2*np.pi*w/Fs
    H_arg=np.angle(H, deg=False)

    ax[0,1].plot(w_r, H_arg)
    ax[0,1].set_title('Phase Response')
    ax[0,1].set_xlabel('Frequency (in rad)')
    ax[0,1].set_ylabel('Phase (in rad)')
    ax[0,1].grid('on')
    
    H_db=20*np.log10(abs(H))

    ax[1,0].plot(w,H_db)
    ax[1,0].set_title('Magnitude Response')
    ax[1,0].set_xlabel('Frequency (in Hz)')
    ax[1,0].set_ylabel('Magnitude (in dB)')
    ax[1,0].grid('on')

    theta=np.linspace(0,2*np.pi, 100)
    ejtheta=np.exp(1j*theta)
    ax[1,1].plot(np.real(ejtheta), np.imag(ejtheta))
    for pt in z:
        ax[1,1].plot(np.real(pt), np.imag(pt), 'ro')
    for pt in p:
        ax[1,1].plot(np.real(pt), np.imag(pt), 'rx')
    ax[1,1].set_title('Pole-Zero Plot')
    ax[1,1].grid()
    ax[1,1].set_aspect('equal', adjustable="datalim")

    plt.tight_layout()
    
    #plt.figure()


def let_user_pick(options):
    for idx, element in enumerate(options):
        print("{}) {}".format(idx+1,element))
    i = input("Enter number: ")
    try:
        if 0 < int(i) <= len(options):
            return int(i)
    except:
        pass
    return 1

def askForNumber():
    while True:
        try:
            return float(input('Please enter a number: '))
        except ValueError:
            pass

def askForArray():
    print("Give input as array (type elements with spaces)!")
    arr=input()
    arr2=arr.split()
    return np.array(arr2, dtype="complex")
    

#t = np.linspace(0, 1,Fs)
#u = (1*np.cos(2*np.pi*400*t) + 0.4*np.sin(2*np.pi*5000*t) + 0.01*np.cos(2*np.pi*10000*t))

filter_models=['IIR with cutoff frequency', 'IIR with zeros, poles and gain', 'IIR with numerator and denominator coefficients', 'FIR with cutoff frequency and window type', 'FIR with taps']
filter_types=['Lowpass', 'Highpass', 'Bandpass', 'Bandstop']
fir_windows=['Hamming', 'Hann', 'Blackman', 'Cosine']

#print("Choose type of filter and filter inputs:\n 1) IIR with cutoff frequency \n2) IIR with zeros, poles and gain \n3) IIR with numerator and denominator coefficients\n4) FIR with cutoff frequency and window type\n5) FIR with taps")
print("----------------------------Choose one method of filter design:----------------------------")
input_type=let_user_pick(filter_models)

if(input_type==1):
    #print("Choose Type of Filter:\n 1) lowpass\n 2) highpass\n 3) bandpass\n 4) bandstop")
    print("----------------------------Choose Type of Filter:----------------------------")
    type_filter=let_user_pick(filter_types)
    if(type_filter==1):
        print("Input cutoff frequency (in Hz):----------------------------")
        fc=askForNumber()
        print("Input filter order:----------------------------")
        order=askForNumber()
        filter_impulse_response, filter_freq_resp, frequencies, zeros, poles=IIR_filter_cutoff(fc, order, filter_type="lowpass")
    elif(type_filter==2):
        print("Input cutoff frequency (in Hz):----------------------------")
        fc=askForNumber()
        print("Input filter order:----------------------------")
        order=askForNumber()
        filter_impulse_response, filter_freq_resp, frequencies, zeros, poles=IIR_filter_cutoff(fc, order, filter_type="highpass")
    elif(type_filter==3):
        print("Input lower cutoff frequency (in Hz):----------------------------")
        fc1=askForNumber()
        print("Input upper cutoff frequency (in Hz):----------------------------")
        fc2=askForNumber()
        print("Input filter order:----------------------------")
        order=askForNumber()
        filter_impulse_response, filter_freq_resp, frequencies, zeros, poles=IIR_filter_cutoff(np.array([fc1,fc2]), order, filter_type="bandpass")
    elif(type_filter==4):
        print("Input lower cutoff frequency (in Hz):----------------------------")
        fc1=askForNumber()
        print("Input upper cutoff frequency (in Hz):----------------------------")
        fc2=askForNumber()
        print("Input filter order:----------------------------")
        order=askForNumber()
        filter_impulse_response, filter_freq_resp, frequencies, zeros, poles=IIR_filter_cutoff(np.array([fc1,fc2]), order, filter_type="bandstop")   

elif(input_type==2):
    print("Input zeros:----------------------------")
    z=askForArray()
    print("Input poles:----------------------------")
    p=askForArray()
    print("Input gain:----------------------------")
    k=askForNumber()
    filter_impulse_response, filter_freq_resp, frequencies, zeros, poles=IIR_filter_zpk(z, p, k)

elif(input_type==3):
    print("Input Numerator coefficients for transfer function (in increasing order of negative power):----------------------------")
    num=askForArray()
    print("Input Denominator coefficients for transfer function in (increasing order of negative power):----------------------------")
    den=askForArray()
    filter_impulse_response, filter_freq_resp, frequencies, zeros, poles=IIR_filter_rational(num, den)

elif(input_type==4):
    
    #print("ChooseType of window:\n1)Hamming\n2)Hann\n3)Blackman\n4)Cosine")
    print("----------------------------ChooseType of window:----------------------------")
    win=let_user_pick(fir_windows)
    if(win==1):
        w="hamming"
    elif(win==2):
        w="hann"
    elif(win==3):
        w="blackman"
    elif(win==4):
        w="cosine"

        #print("Choose Type of Filter:\n 1) lowpass\n 2) highpass\n 3) bandpass\n 4) bandstop")
    print("----------------------------Choose Type of Filter:----------------------------")
    type_filter=let_user_pick(filter_types)
    if(type_filter==1):
        print("Input cutoff frequency (in Hz):----------------------------")
        fc=askForNumber()
        print("Input the number of taps:----------------------------")
        num=askForNumber()
        filter_impulse_response, filter_freq_resp, frequencies, zeros, poles=FIR_filter_cutoff(fc, num, window_type=w, filter_type="lowpass")
    elif(type_filter==2):
        print("Input cutoff frequency (in Hz):----------------------------")
        fc=askForNumber()
        print("Input the number of taps(should be odd):----------------------------")
        num=askForNumber()
        filter_impulse_response, filter_freq_resp, frequencies, zeros, poles=FIR_filter_cutoff(fc, num, window_type=w, filter_type="highpass")
    elif(type_filter==3):
        print("Input lower cutoff frequency (in Hz):----------------------------")
        fc1=askForNumber()
        print("Input upper cutoff frequency (in Hz):----------------------------")
        fc2=askForNumber()
        print("Input the number of taps:----------------------------")
        num=askForNumber()
        filter_impulse_response, filter_freq_resp, frequencies, zeros, poles=FIR_filter_cutoff(np.array([fc1,fc2]), num, window_type=w, filter_type="bandpass")
    elif(type_filter==4):
        print("Input lower cutoff frequency (in Hz):----------------------------")
        fc1=askForNumber()
        print("Input upper cutoff frequency (in Hz):----------------------------")
        fc2=askForNumber()
        print("Input the number of taps(should be odd):----------------------------")
        num=askForNumber()
        filter_impulse_response, filter_freq_resp, frequencies, zeros, poles=FIR_filter_cutoff(np.array([fc1,fc2]), num, window_type=w, filter_type="bandstop")
elif(input_type==5):
    print("Input the taps:")
    t=askForArray()
    filter_impulse_response, filter_freq_resp, frequencies, zeros, poles=FIR_filter_coefficients(t)
    
#f=filter(u, filter_impulse_response)

plot_filter(filter_impulse_response, filter_freq_resp, frequencies, zeros, poles)
print('FILTER DESIGNED')
plt.show()



#############################################################################################################################################33


# create matplotlib figure and axes
'''fig3, (ax1, ax2) = plt.subplots(2, figsize=(15, 7))

# pyaudio class instance
p = pyaudio.PyAudio()

player = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    output=True,
    frames_per_buffer=CHUNK
)
# stream object to get data from microphone
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

# variable for plotting
x = np.arange(0, 2 * CHUNK, 2)       # samples (waveform)
xf = np.linspace(0, RATE, CHUNK)     # frequencies (spectrum)

# create a line object with random data
line, = ax1.plot(x, np.random.rand(CHUNK),label='Input', animated=True)


# create semilogx line for spectrum
line_fft, = ax2.semilogx(xf, np.random.rand(CHUNK),label='Input', animated=True)

line_filtered, = ax1.plot(x, np.random.rand(CHUNK),label='Output', animated=True)

line_fft_filtered, = ax2.semilogx(xf, np.random.rand(CHUNK),label='Output', animated=True)


# format waveform axes
ax1.set_title('AUDIO WAVEFORM')
ax1.set_xlabel('samples')
ax1.set_ylabel('volume')
ax1.set_ylim(0, 255)
ax1.set_xlim(0, 2 * CHUNK)
ax1.legend()
plt.setp(ax1, xticks=[0, CHUNK, 2 * CHUNK], yticks=[0, 128, 255])



# format spectrum axes
ax2.set_xlim(20, RATE / 2)
ax2.legend()

plt.show(block=False)

plt.pause(0.1)

bg = fig3.canvas.copy_from_bbox(fig3.bbox)

ax1.draw_artist(line)
ax1.draw_artist(line_filtered)
ax2.draw_artist(line_fft)
ax2.draw_artist(line_fft_filtered)

fig3.canvas.blit(fig3.bbox)

print('stream started')

# for measuring frame rate
frame_count = 0
start_time = time.time()

while True:
    
    fig3.canvas.restore_region(bg)
    # binary data
    data = stream.read(CHUNK)
    player.write(np.frombuffer(data ,dtype=np.int16),CHUNK)
    
    # convert data to integers, make np array, then offset it by 127
    data_int = struct.unpack(str(2 * CHUNK) + 'B', data)
    
    # create np array and offset by 128
    data_np = np.array(data_int, dtype='b')[::2] + 128
    
    line.set_ydata(data_np)
    
    filtered_np=filter(data_np, filter_impulse_response)
    filtered_int=filter(data_int, filter_impulse_response)
    #filtered_np = np.array(filtered_int, dtype='b')[::2] + 128
    line_filtered.set_ydata(filtered_np)
    #num=[100]
    #den=[1,10]
    #Hz=signal.TransferFunction(num,den)
    # compute FFT and update line
    #tout, resp, x = signal.lsim(Hz, data_int, x)
    yf=fft(data_int)
    line_fft.set_ydata(np.abs(yf[0:CHUNK])  / (128 * CHUNK))
    
    fft_filtered=fft(filtered_int)
    line_fft_filtered.set_ydata(np.abs(fft_filtered[0:CHUNK])  / (128 * CHUNK))
    #player.write(filtered_np.astype(np.float32).tobytes())

    ax1.draw_artist(line)
    ax1.draw_artist(line_filtered)
    ax2.draw_artist(line_fft)
    ax2.draw_artist(line_fft_filtered)
    #player.write(np.frombuffer(struct.pack('hhl',resp),CHUNK)
    # update figure canvas
    if len(plt.get_fignums())!=0:        
        #fig3.canvas.draw()
        fig3.canvas.blit(fig3.bbox)
        fig3.canvas.flush_events()
        frame_count += 1
        
    else:
        
        # calculate average frame rate
        frame_rate = frame_count / (time.time() - start_time)
        
        print('stream stopped')
        print('average frame rate = {:.0f} FPS'.format(frame_rate))
        break

p.terminate()'''

fig3, (ax1, ax2) = plt.subplots(2, figsize=(15, 7))

# pyaudio class instance
p = pyaudio.PyAudio()

player = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        output=True,
        frames_per_buffer=CHUNK
    )
    # stream object to get data from microphone
stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    # variable for plotting
x = np.arange(0, 2 * CHUNK, 2)       # samples (waveform)
xf = np.linspace(0, RATE, CHUNK)     # frequencies (spectrum)

    # create a line object with random data
line, = ax1.plot(x, np.random.rand(CHUNK),label='Input', animated=True)
line_filtered, = ax1.plot(x, np.random.rand(CHUNK),label='Output', animated=True)

    # create semilogx line for spectrum
line_fft, = ax2.semilogx(xf, np.random.rand(CHUNK),label='Input', animated=True)
line_fft_filtered, = ax2.semilogx(xf, np.random.rand(CHUNK),label='Output', animated=True)


    # format waveform axes
ax1.set_title('AUDIO WAVEFORM')
ax1.set_xlabel('samples')
ax1.set_ylabel('volume')
    #ax1.set_ylim(-255, 256)
ax1.set_xlim(0, 2 * CHUNK)
ax1.legend()
plt.setp(ax1, xticks=[0, CHUNK, 2 * CHUNK], yticks=[-255, 0, 256])



    # format spectrum axes
ax2.set_xlim(20, RATE / 2)
ax2.legend()

plt.show(block=False)

plt.pause(0.05)

bg = fig3.canvas.copy_from_bbox(fig3.bbox)

ax1.draw_artist(line)
ax1.draw_artist(line_filtered)
ax2.draw_artist(line_fft)
ax2.draw_artist(line_fft_filtered)

fig3.canvas.blit(fig3.bbox)

print('stream started')

    # for measuring frame rate
frame_count = 0
start_time = time.time()

while True:
        
    fig3.canvas.restore_region(bg)
        # binary data
    data = stream.read(CHUNK)

    data_int=np.frombuffer(data,dtype=np.int16)
        
    m=np.iinfo(np.int16)
    m1=m.max
    
    data_np=512*((data_int)/float(m1))

    line.set_ydata(data_np)
        
    filtered_int=filter(data_int, filter_impulse_response)
    filtered_np = (512*(filtered_int))/float(m1)

    line_filtered.set_ydata(filtered_np)
        
    yf=fft(data_int)
    line_fft.set_ydata(np.abs(yf[0:CHUNK])  / (m1/50 * CHUNK))
        
    fft_filtered=fft(filtered_int)
    line_fft_filtered.set_ydata(np.abs(fft_filtered[0:CHUNK])  / (m1/50 * CHUNK))
        
    filtered=np.array(filtered_int, dtype=np.int16)
    player.write(filtered,CHUNK)

    ax1.draw_artist(line)
    ax1.draw_artist(line_filtered)
    ax2.draw_artist(line_fft)
    ax2.draw_artist(line_fft_filtered)
        
        # update figure canvas
    if len(plt.get_fignums())!=0:        
            #fig3.canvas.draw()
        fig3.canvas.blit(fig3.bbox)
        fig3.canvas.flush_events()
        frame_count += 1
            
    else:
            
            # calculate average frame rate
        frame_rate = frame_count / (time.time() - start_time)
            
        print('stream stopped')
        print('average frame rate = {:.0f} FPS'.format(frame_rate))
        break

p.terminate()

print("---THE END---")