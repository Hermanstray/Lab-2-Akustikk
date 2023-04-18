
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import estimateAngle


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def raspi_import(path, channels=5): 
    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype='uint16').astype('float64')
        # The "dangling" `.astype('float64')` casts data to double precision
        # Stops noisy autocorrelation due to overflow
        print(len(data))
        data = data.reshape((-1, channels))
    return sample_period, data


# length = 0.065 #meters
Fs = 31500 #Hz

def maxCorrelationLag(signal1, signal2):
    correlatedSignal = abs(np.correlate(signal1, signal2, "full"))
    return correlatedSignal

def maxLag(signal1, signal2):
    correlatedSignal = abs(np.correlate(signal1, signal2, "full"))
    maxLag = np.argmax(correlatedSignal) - ((len(correlatedSignal)-1)/2)
    print("maxlag:", maxLag)
    return maxLag

fname = 'lab2v20deg45v2.bin'
_, dataOut = raspi_import(fname,5)

try:
    sample_period, data = raspi_import(fname)
except FileNotFoundError as err:
    print(f"File {fname} not found. Check the path and try again.")
    exit(1)

data = signal.detrend(data, axis=0)
sample_period *= 1e-6
num_of_samples = data.shape[0]  # returns shape of matrix
t = np.linspace(0, num_of_samples*sample_period, num_of_samples,
        endpoint=False)

signalMic1 = dataOut[:,4]
signalMic2 = dataOut[:,3]
signalMic3 = dataOut[:,2]

filteredSignalMic1 = bandpass_filter(signalMic1, 100, 15000, Fs)            #mainmic1
filteredSignalMic2= bandpass_filter(signalMic2, 100, 15000, Fs)             #mainmic2
filteredSignalMic3 = bandpass_filter(signalMic3, 100, 15000, Fs)            #cali

maxCorrLag21 = maxCorrelationLag(filteredSignalMic2, filteredSignalMic1)   
maxCorrLag31 = maxCorrelationLag(filteredSignalMic3, filteredSignalMic1)  
maxCorrLag32 = maxCorrelationLag(filteredSignalMic3, filteredSignalMic2)  

maxLag21 = maxLag(filteredSignalMic2, filteredSignalMic1)   
maxLag31 = maxLag(filteredSignalMic3, filteredSignalMic1)  
maxLag32 = maxLag(filteredSignalMic3, filteredSignalMic2) 

angle = estimateAngle.estimateAngleByLMS(maxLag21, maxLag31, maxLag32)
print("Angle:", angle)

freq = np.fft.fftfreq(n=num_of_samples, d=sample_period)
freq = np.fft.fftshift(freq)
spectrum = np.fft.fftshift(np.fft.fft(data, axis=0), axes=0)

time = np.arange(len(filteredSignalMic1)-1) / Fs

timeFig, timeAxs = plt.subplots(3,1,figsize =(10,8))

#timeAxs[0].subplot(3,1,1)
timeAxs[0].plot(time, filteredSignalMic1[1:], color = 'green')
timeAxs[0].set_title('Tidssignal Mikrofon 1')
timeAxs[0].set_xlabel('Tid [s]')
timeAxs[0].set_ylabel('Spenning [mV]')

#timeAxs[1].subplot(3,1,2)
timeAxs[1].plot(time, filteredSignalMic2[1:], color = 'red')
timeAxs[1].set_title('Tidssignal Mikrofon 2')
timeAxs[1].set_xlabel('Tid [s]')
timeAxs[1].set_ylabel('Spenning [mV]')

#timeAxs[2].subplot(3,1,3)
timeAxs[2].plot(time, filteredSignalMic3[1:], color = 'blue')
timeAxs[2].set_title('Tidssignal Mikrofon 3')
timeAxs[2].set_xlabel('Tid [s]')
timeAxs[2].set_ylabel('Spenning [mV]')

plt.tight_layout()
plt.show()

# print("maxcorrlag21", maxCorrLag21[1], " length", len(maxCorrLag21[1]))

corrLength = np.arange(99)

fig, axs = plt.subplots(3, 1, figsize=(10, 8))

axs[0].plot(corrLength, maxCorrLag21[((len(maxCorrLag21)//2)+1):(len(maxCorrLag21)//2)+100], color='green')
axs[0].set_title('Autokorrelasjon Mikrofon 21')
axs[0].set_xlabel('Tid [s]')
axs[0].set_ylabel('Signal amplitude [-]')

axs[1].plot(corrLength, maxCorrLag31[((len(maxCorrLag31)//2)+1):(len(maxCorrLag31)//2)+100], color='red')
axs[1].set_title('Autokorrelasjon Mikrofon 31')
axs[1].set_xlabel('Tid [s]')
axs[1].set_ylabel('Signal amplitude [-]')

axs[2].plot(corrLength, maxCorrLag32[((len(maxCorrLag32)//2)+1):(len(maxCorrLag32)//2)+100], color='blue')
axs[2].set_title('Autokorrelasjon Mikrofon 32')
axs[2].set_xlabel('Tid [s]')
axs[2].set_ylabel('Signal amplitude [-]')

plt.tight_layout()
plt.show()

# plt.plot(freq[len(freq)//2:], 20*np.log10(np.abs(spectrum[len(freq)//2:])))
# plt.legend()
# plt.show()

# plt.subplot(2,1,1)
# plt.plot(range(-(len(maxCorrLag21[2])//2)-1, (len(maxCorrLag31[2])//2)), maxCorrLag21[2])
# plt.plot(range(-round(len(maxCorrLag12[2])/2), round(len(maxCorrLag12[2])/2)-1), maxCorrLag12[2])
# plt.plot(range(-round(len(maxCorrLag12[2])/2), round(len(maxCorrLag12[2])/2)-1), maxCorrLag12[2])

# plt.plot(range(1,len(filteredSignalMic2)), filteredSignalMic2[1:])
# plt.plot(range(1,len(filteredSignalMic3)), filteredSignalMic3[1:])


