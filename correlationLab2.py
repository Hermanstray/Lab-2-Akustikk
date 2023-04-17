import numpy as np

# signal1 = [0,1,0,0, 0]
# signal2 = [0,0,0,0, 1]



length = 0.065 #meters
Fs = 4000 #Hz



def maxCorrelationLag(signal1, signal2):
    correlatedSignal = abs(np.correlate(signal1, signal2, "full"))
    print(correlatedSignal)
    maxLag = np.argmax(correlatedSignal)
    maxLagVal = correlatedSignal[maxLag]
    print(maxLag, '\n', maxLagVal)
    return maxLag, maxLagVal

maxCorrLag = maxCorrelationLag(signal1, signal2)

deltaT = maxCorrLag[0]/Fs
print(deltaT)