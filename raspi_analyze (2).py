
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
    correlatedSignal = abs(signal.correlate(signal1, signal2, "full", "auto"))
    return correlatedSignal

def maxLag(signal1, signal2):
    correlatedSignal = abs(signal.correlate(signal1, signal2, "full", "auto"))
    maxLag = np.argmax(correlatedSignal) - ((len(correlatedSignal)-1)/2)
    print("maxlag:", maxLag)
    return maxLag

fname = 'lab2v20deg45v4.bin'
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

autokorr11 = maxCorrelationLag(filteredSignalMic1, filteredSignalMic1)

maxLag21 = maxLag(filteredSignalMic2, filteredSignalMic1)   
maxLag31 = maxLag(filteredSignalMic3, filteredSignalMic1)  
maxLag32 = maxLag(filteredSignalMic3, filteredSignalMic2) 

angle = estimateAngle.estimateAngleByLMS(maxLag21, maxLag31, maxLag32)
print("Angle:", angle)

# freq = np.fft.fftfreq(n=num_of_samples, d=sample_period)
# freq = np.fft.fftshift(freq)
# spectrum = np.fft.fftshift(np.fft.fft(data, axis=0), axes=0)

time = np.arange(0, len(filteredSignalMic1)) / Fs
#time = [-len(time)//2, len(time)//2]

timeFig, timeAxs = plt.subplots(3,1,figsize =(10,8))

#timeAxs[0].subplot(3,1,1)
timeAxs[0].plot(time, signalMic1, color = 'green')
timeAxs[0].set_title('Tidssignal Mikrofon 1')
timeAxs[0].set_xlabel('Tid [s]')
timeAxs[0].set_ylabel('Spenning [mV]')

#timeAxs[1].subplot(3,1,2)
timeAxs[1].plot(time, signalMic2, color = 'red')
timeAxs[1].set_title('Tidssignal Mikrofon 2')
timeAxs[1].set_xlabel('Tid [s]')
timeAxs[1].set_ylabel('Spenning [mV]')

#timeAxs[2].subplot(3,1,3)
timeAxs[2].plot(time, signalMic3, color = 'blue')
timeAxs[2].set_title('Tidssignal Mikrofon 3')
timeAxs[2].set_xlabel('Tid [s]')
timeAxs[2].set_ylabel('Spenning [mV]')

plt.tight_layout()
plt.show()

#Zoomet inn tidssignal

# l = len(filteredSignalMic1)

# filteredSignalMic1l = filteredSignalMic1[19467: 19592]
# filteredSignalMic2l = filteredSignalMic2[19467: 19592]
# filteredSignalMic3l = filteredSignalMic3[19467: 19592]

# time = np.arange((l*0.618), l*0.622) / Fs
# #time = [-len(time)//2, len(time)//2]

# timeFig, timeAxs = plt.subplots(3,1,figsize =(10,8))

# timeAxs[0].plot(time, filteredSignalMic1l, color = 'green')
# timeAxs[0].set_title('Tidssignal Mikrofon 1')
# timeAxs[0].set_xlabel('Tid [s]')
# timeAxs[0].set_ylabel('Spenning [mV]')

# #timeAxs[1].subplot(3,1,2)
# timeAxs[1].plot(time, filteredSignalMic2l, color = 'red')
# timeAxs[1].set_title('Tidssignal Mikrofon 2')
# timeAxs[1].set_xlabel('Tid [s]')
# timeAxs[1].set_ylabel('Spenning [mV]')

# #timeAxs[2].subplot(3,1,3)
# timeAxs[2].plot(time, filteredSignalMic3l, color = 'blue')
# timeAxs[2].set_title('Tidssignal Mikrofon 3')
# timeAxs[2].set_xlabel('Tid [s]')
# timeAxs[2].set_ylabel('Spenning [mV]')

# plt.tight_layout()
# plt.show()


# print("maxcorrlag21", maxCorrLag21[1], " length", len(maxCorrLag21[1]))
#[((len(maxCorrLag32)//2)+1):(len(maxCorrLag32)//2)+100]
corrLength = np.arange((-len(maxCorrLag21)//2)+1, (len(maxCorrLag21))//2)

fig, axs = plt.subplots(figsize=(10, 8))
# fig.subplots_adjust(hspace=0.5)

# axs.plot(corrLength, maxCorrLag21[:len(maxCorrLag21)-1], color='green', label = 'Krysskorrelasjon Mikrofon 21')
# axs.plot(corrLength, maxCorrLag31[:len(maxCorrLag21)-1], color='red', label = 'Krysskorrelasjon Mikrofon 31')
# axs.plot(corrLength, maxCorrLag32[:len(maxCorrLag21)-1], color='blue', label = 'Krysskorrelasjon Mikrofon 32')
# axs.set_title('Krysskorrelasjon')
# axs.set_xlabel('Samples')
# axs.set_ylabel('Signal amplitude [-]')

# plt.xlim(-20,20)
# plt.legend()
# plt.show()



# Upsample the signals by a factor of 8 using resample_poly
upsampling_factor = 8
downsampling_factor = 1
upsampled_signal21 = signal.resample_poly(maxCorrLag21,  len(maxCorrLag21) * upsampling_factor, len(maxCorrLag21) * downsampling_factor)
upsampled_signal31 = signal.resample_poly(maxCorrLag31,  len(maxCorrLag21) * upsampling_factor, len(maxCorrLag21) * downsampling_factor)
upsampled_signal32 = signal.resample_poly(maxCorrLag32,  len(maxCorrLag21) * upsampling_factor, len(maxCorrLag21) * downsampling_factor)


maxLag21i = np.argmax(upsampled_signal21)- ((len(upsampled_signal21))/2)
maxLag31i = np.argmax(upsampled_signal31)- ((len(upsampled_signal21))/2) 
maxLag32i = np.argmax(upsampled_signal32)- ((len(upsampled_signal21))/2)

angle = estimateAngle.estimateAngleByLMS(maxLag21i, maxLag31i, maxLag32i)
print("Angle:", angle)

print(np.argmax(upsampled_signal21)- ((len(upsampled_signal21))/2))
print(np.argmax(upsampled_signal31)- ((len(upsampled_signal21))/2))
print(np.argmax(upsampled_signal32)- ((len(upsampled_signal21))/2))
xaks = np.arange(-len(upsampled_signal21)//2, len(upsampled_signal21)//2)

# fig, axs = plt.subplots()
# axs.plot(xaks, upsampled_signal21[:len(upsampled_signal21)], color='green', label = 'Krysskorrelasjon Mikrofon 21')
# axs.plot(xaks, upsampled_signal31[:len(upsampled_signal31)], color='red', label = 'Krysskorrelasjon Mikrofon 31')
# axs.plot(xaks, upsampled_signal32[:len(upsampled_signal32)], color='blue', label = 'Krysskorrelasjon Mikrofon 32')
# axs.set_title('Krysskorrelasjon med oppsamplingsfaktor = 8')
# axs.set_xlabel('Samples')
# axs.set_ylabel('Signal amplitude [-]')

# plt.xlim(-20,70)
# plt.legend()
# plt.show()



# plt.plot(corrLength, autokorr11[:len(maxCorrLag21)-1])
# plt.gca().set_title('Autokorrelasjon Mikrofon 1')
# plt.gca().set_xlabel('Samples')
# plt.gca().set_ylabel('Signal amplitude [-]')
# plt.gca().set_xlim(-20,20)

# plt.show()

# plt.plot(freq[len(freq)//2:], 20*np.log10(np.abs(spectrum[len(freq)//2:])))
# plt.legend()
# plt.show()

# plt.subplot(2,1,1)
# plt.plot(range(-(len(maxCorrLag21[2])//2)-1, (len(maxCorrLag31[2])//2)), maxCorrLag21[2])
# plt.plot(range(-round(len(maxCorrLag12[2])/2), round(len(maxCorrLag12[2])/2)-1), maxCorrLag12[2])
# plt.plot(range(-round(len(maxCorrLag12[2])/2), round(len(maxCorrLag12[2])/2)-1), maxCorrLag12[2])

# plt.plot(range(1,len(filteredSignalMic2)), filteredSignalMic2[1:])
# plt.plot(range(1,len(filteredSignalMic3)), filteredSignalMic3[1:])


