#!/usr/bin/python3
import numpy as np
import pylab as P
import sys
import copy 
from os import path
from scipy.optimize import leastsq
from f_norm_line import norm_line_felix
from f_norm_line import felix_binning
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, NullFormatter, NullLocator

DELTA=2.0

# 1cm-1 = 30GHz  ,  0.001cm-1=30MHz
################################################################################
def export_file(fname, wn, inten):
    f = open(fname + '.dat','w')
    f.write("#DATA points as shown in figure: " + fname + ".pdf file!\n")
    f.write("#wn (cm-1)       intensity\n")
    for i in range(len(wn)):
        f.write("{:8.3f}\t{:8.2f}\n".format(wn[i], inten[i]))
    f.close()



#----------------------------------------
#ENTRY POINT:
if __name__ == "__main__":

    fig = P.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)

    #line list to be plotted, filename without suffix
    l_list=[['25_04_18-7', 'r', 0.8],
            ['25_04_18-9', 'b', 1.0]]

    xs = np.array([],dtype='double')
    ys = np.array([],dtype='double')
    for l in l_list:
        a,b = norm_line_felix(l[0])
        ax.plot(a, b*l[2], ls='', marker='o', ms=1, c=l[1], markeredgecolor=l[1], markerfacecolor=l[1])
        xs = np.append(xs,a)
        ys = np.append(ys,b*l[2])

    binns, inten = felix_binning(xs, ys, delta=DELTA)
    ax.plot(binns, inten, ls='-', marker='', c='k')

    F = 'OUT/avg-spectrum.pdf'
    export_file(F, binns, inten)

    #ax.set_xlim([480,1020])
    ax.set_xlim([480,1020])
    ax.set_xlabel(r"calib. lambda (cm-1)")
    ax.set_ylabel(r"inten. (norm)")

    ax.grid(True)
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.xaxis.set_major_locator(MultipleLocator(50))

    fig.savefig(F)
    P.close(fig)
 
    #input('press enter to quit...')
