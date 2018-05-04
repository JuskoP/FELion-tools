#!/usr/bin/python3
import numpy as np
import pylab as P
import sys
import copy 
from os import path
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy.interpolate import interp1d
from f_baseline import felix_read_file


################################################################################

class SpectrumAnalyserCalibrator(object):
    def __init__(self, fname, fit='linear'):
        """
        Spectrum analyser calibration initialisation
        fit can be either linear, or cubic
        """
        data = felix_read_file(fname)
    
        #Spectrum analyser calibration
        #Added 6.10.16:
        #Spectrum analyser is calibrated only above 540 cm-1, because for lower values SA gives bulshit values!
        #In case, someone does not follow the SA, value of SA < 100 will get also excluded from the fit!!!!
        sa_x = np.copy(data[0][np.logical_and(data[0]>400, data[2]>100)])
        sa_y = np.copy(data[2][np.logical_and(data[0]>400, data[2]>100)])


        if fit == 'linear':
            p0 = [1.0, 5.0]
            fitfunc = lambda p, x: p[0]*x + p[1]
        elif fit == 'cubic':
            p0 = [.01, .01, 1.0, 0.0]
            fitfunc = lambda p, x: p[0]*x**3 + p[1]*x**2 + p[2]*x + p[3]
        else:
            print("SA fitting function not defined!!!")


        errfunc = lambda p, x, y: (fitfunc(p, x)-y)

        p, cov_p, info, mesg, ier \
            = leastsq(errfunc, p0, args=(sa_x, sa_y), full_output=1)
            
        if ier not in [1, 2, 3, 4] or cov_p is None:
            msg = "SA calibration optimal parameters not found: " + mesg
            raise RuntimeError(msg)
        if any(np.diag(cov_p) < 0):
            raise RuntimeError("Optimal parameters not found: negative variance")
            
        chisq = np.dot(info["fvec"], info["fvec"])
        dof = len(info["fvec"]) - len(p)
        sigma = np.array([np.sqrt(cov_p[i,i])*np.sqrt(chisq/dof) for i in range(len(p))])

        self.p = p
        self.sigma = sigma
        self.f = fitfunc
        self.data = (data[0][data[2]>100], data[2][data[2]>100])
        info = self.getInfo()
        print(info)

    def sa_cm(self, x):
        return self.f(self.p, x)

    def getData(self):
        return self.data
    
    def getInfo(self):
        info = "#Fit SA vs UNDULATOR (polynome p[0]*x**n + p[1]*x**(n-1) + ... + p[n]):\n"
        info += "#p[]     = " + "".join("{:.4e} ".format(i) for i in self.p) + \
            "\n#sigma[] = " +    "".join("{:.4e} ".format(i) for i in self.sigma) + "\n"
        info += "#----------------------------------------\n"
        return info 

    def plot(self, ax):
        wn, sa = self.data
        X = np.arange(wn.min(),wn.max(), 1)
        ax.plot(wn, sa, ls='', marker='s', ms=3, markeredgecolor='r', c='r')
        ax.plot(X, self.sa_cm(X), ls='-', marker='', c='g')


#----------------------------------------
#ENTRY POINT:
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("fname", help="Filename to process")
    args = parser.parse_args()

    #a,b = line_felix(args.fname)
 

    if(args.fname.find('DATA')>=0):
        fname = args.fname.split('/')[-1]
    else:
        fname = args.fname

    if(fname.find('felix')>=0):
        fname = fname.split('.')[0]
        
    saCalibrator = SpectrumAnalyserCalibrator(fname, fit='cubic')
    
    fig, ax = plt.subplots()
    #plot the spectrum analyser calibration
    saCalibrator.plot(ax)

    ax.set_title('Spectrum analyser calibration from .felix file')
    #ax.set_xlim((wn.min()-70, wn.max()+70))
    #ax.set_ylim((0, 30))
    ax.set_xlabel("wn set (cm-1)")
    ax.set_ylabel("wn SA (cm-1)")
    plt.show()

