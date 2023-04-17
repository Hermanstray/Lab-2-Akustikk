
import numpy as np
import math


def estimateAngleByLMS(maxLag21, maxLag31, maxLag32):
    # d = 0.065
    # Fs = 31500
    # c = 340

    # tauMic1Mic2 = maxLag12/Fs
    # tauMic1Mic3 = maxLag13/Fs
    # tauMic2Mic3 = maxLag23/Fs
    print(maxLag21, maxLag31, maxLag32)

    theta = np.arctan2(np.sqrt(3)*(maxLag21+maxLag31), (maxLag21-maxLag31-2*maxLag32))
    print(theta)
    thetaDeg = np.degrees(theta)
    return thetaDeg
    # sensor1 = [0,1]*a
    # sensor2 = [-np.sqrt(3)/2, -3/2]*a
    # sensor3 = [np.sqrt(3)/2, -0.5]*a

    # x21 = sensor2-sensor1
    # x31 = sensor3-sensor1
    # x32 = sensor3-sensor2




    # eq12 = d*np.sin(alpha) + d*np.cos(alpha)+ c *tauMic1Mic2
    # eq13 = d*np.sin(alpha) + d*np.cos(alpha)+ c *tauMic1Mic3
    # eq23 = d*np.sin(alpha) + d*np.cos(alpha)+ c *tauMic2Mic3


