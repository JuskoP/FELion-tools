#!/usr/bin/python3
import numpy as np
import pylab as P
import sys
from os import path
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


################################################################################
class PowerCalibrator(object):
    """
    Reads the power file and provides the power and n_shot
    for any given wavenumber
    """
    def __init__(self, fname):
        """
        interpolation can be either 'cubic' or 'linear'
        """
        self.n_shots = 1
        self.interpol = 'linear'
        in_um = False
        xw, yw = [],[]
        f = open('DATA/' + fname + '.pow')
        for line in f:
            if line[0] == '#':
                if line.find('SHOTS')==1:
                    self.n_shots = float(line.split('=')[-1])
                if line.find('IN_UM')==1:
                    in_um = True
                if line.find('INTERP')==1:
                    self.interpol = line.split('=')[-1].strip('\n')
                continue
            else:
                x, y, = line.split()
                xw.append(float(x))
                yw.append(float(y))
    
        if in_um:
            self.xw = 10000/np.array(xw)
        else:
            self.xw = np.array(xw)

        self.yw = np.array(yw)
        f.close()
        self.f = interp1d(self.xw, self.yw, kind=self.interpol,fill_value='extrapolate')

    def power(self, x):
        return self.f(x) 

    def shots(self, x):
        if type(x)==float:
            return self.n_shots 
        else:
            return np.zeros(len(x)) + self.n_shots

    def GetCalibData(self):
        return self.xw, self.yw, self.n_shots

    def plot(self, ax, bx):
        ax.plot(self.xw, self.yw, ls='', marker='o', ms=5, markeredgecolor='r', c='r')
        ax.plot(self.xw, self.power(self.xw), ls='-', c='m')
        ax.set_ylabel("power (mJ)")
        ax.set_ylim((0, self.yw.max()*1.1))

        bx.plot(self.xw, self.shots(self.xw), ls='-', marker='o', ms=3, markeredgecolor='y', c='y')
        bx.set_ylabel("shots")


#----------------------------------------
#ENTRY POINT:
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("fname", help="Filename to process")
    args = parser.parse_args()

    if(args.fname.find('DATA')>=0):
        fname = args.fname.split('/')[-1]
    else:
        fname = args.fname

    if(fname.find('felix')>=0):
        fname = fname.split('.')[0]
        
    #x, y, n_shots = ReadPower(fname)
    powerWN = PowerCalibrator(fname)
    xc, yc, n_shots = powerWN.GetCalibData()
    X = np.arange(xc.min(),xc.max(), 1)

    fig, ax = plt.subplots()
    bx = ax.twinx()
    ax.plot(xc, yc, ls='', marker='o', ms=5, markeredgecolor='r', c='r')
    bx.plot(xc, powerWN.shots(xc), ls='-', marker='o', ms=3, markeredgecolor='b', c='b')

    #plot the power calibration line:
    ax.plot(X, powerWN.power(X), ls='-', c='m')

    ax.set_title('Power and Number of shots in the .pow file')
    ax.set_xlim((xc.min()-70, xc.max()+70))
    ax.set_ylim((0, yc.max()*1.1))
    ax.set_xlabel("wn (cm-1)")
    ax.set_ylabel("power (mJ)")
    bx.set_ylabel("n shots")
    plt.show()

    #input('press enter to quit...')
