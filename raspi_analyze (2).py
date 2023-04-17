
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
    
    #print(len(correlatedSignal))
    maxLag = np.argmax(correlatedSignal) - ((len(correlatedSignal)-1)/2)
    #print(maxLag)
    
    #-(len(correlatedSignal)-1)/2
    
    maxLagVal = correlatedSignal[abs(round(maxLag))]
    #print(maxLag, '\n', maxLagVal)
    return maxLag, maxLagVal, correlatedSignal






fname = 'lab2v20deg45v4.bin'
_, dataOut = raspi_import(fname,5)

print(dataOut)

# radar1 = dataOut[1].T[3]
# radar2 = dataOut[1].T[2]
signalMic1 = dataOut[:,4]
signalMic2 = dataOut[:,3]
signalMic3 = dataOut[:,2]

print(signalMic1)


filteredSignalMic1 = bandpass_filter(signalMic1, 100, 10000, Fs)            #mainmic1
filteredSignalMic2= bandpass_filter(signalMic2, 100, 10000, Fs)             #mainmic2
filteredSignalMic3 = bandpass_filter(signalMic3, 100, 10000, Fs)            #cali

# filteredSignalMic1 = signalMic1
# filteredSignalMic2 = signalMic2
# filteredSignalMic3 = signalMic3


# print(maxCorrelationLag(filteredSignalMic2, filteredSignalMic1)[0])

maxCorrLag21 = maxCorrelationLag(filteredSignalMic2, filteredSignalMic1)   
maxCorrLag31 = maxCorrelationLag(filteredSignalMic3, filteredSignalMic1)  
maxCorrLag32 = maxCorrelationLag(filteredSignalMic3, filteredSignalMic2)  


angle = estimateAngle.estimateAngleByLMS(maxCorrLag21[0], maxCorrLag31[0], maxCorrLag32[0])
print(angle)

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


freq = np.fft.fftfreq(n=num_of_samples, d=sample_period)
freq = np.fft.fftshift(freq)
spectrum = np.fft.fftshift(np.fft.fft(data, axis=0), axes=0)



plt.subplot(2,1,1)
#plt.plot(range((len(data)-1)), data[1:])
plt.plot(range((len(filteredSignalMic1)-1)), filteredSignalMic1[1:])
plt.plot(range((len(filteredSignalMic2)-1)), filteredSignalMic2[1:])
plt.plot(range((len(filteredSignalMic3)-1)), filteredSignalMic3[1:])
plt.legend()
plt.show()

# plt.plot(freq[len(freq)//2:], 20*np.log10(np.abs(spectrum[len(freq)//2:])))

plt.subplot(2,1,1)
plt.plot(range((len(maxCorrLag21[2]))), maxCorrLag21[2])
plt.plot(range((len(maxCorrLag31[2]))), maxCorrLag31[2])
plt.plot(range((len(maxCorrLag32[2]))), maxCorrLag32[2])
plt.plot(freq[len(freq)//2:], 20*np.log10(np.abs(spectrum[len(freq)//2:])))
plt.legend()
plt.show()


plt.plot(range(-(len(maxCorrLag21[2])//2)-1, (len(maxCorrLag31[2])//2)), maxCorrLag21[2])
# plt.plot(range(-round(len(maxCorrLag12[2])/2), round(len(maxCorrLag12[2])/2)-1), maxCorrLag12[2])
# plt.plot(range(-round(len(maxCorrLag12[2])/2), round(len(maxCorrLag12[2])/2)-1), maxCorrLag12[2])

# plt.plot(range(1,len(filteredSignalMic2)), filteredSignalMic2[1:])
# plt.plot(range(1,len(filteredSignalMic3)), filteredSignalMic3[1:])
plt.legend()
plt.show()

