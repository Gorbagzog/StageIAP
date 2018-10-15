#!/usr/bin/env python3
# -*-coding:Utf-8 -*

"""Script to load and plot the the SMHM from Behroozi et al. 2013 updated to Planck 15 cosmology,
and compute the MhaloPeak from it."""

import numpy as np
import matplotlib.pyplot as pl
from scipy.interpolate import InterpolatedUnivariateSpline

redshift = np.array([0.1, 1, 2, 3, 4, 5, 6, 7, 8])

smhm = {}
for z in redshift:
    smhm[z]= np.loadtxt('B13_smhm_comp/behroozi13_z{:.2f}_planck_ns.dat'.format(z))

for z in redshift:
    plt.errorbar(smhm[z][:,0], smhm[z][:,1], yerr=[smhm[z][:,2], smhm[z][:,3]], label=str(z))
    plt.legend()
    plt.show()

mhpeak = {}
for z in redshift:
    """Find the peak of the smhm with an iterpolation of the data points I have"""
    print(smhm[z][np.argmax(smhm[z][:,1]),0])
    x_axis = smhm[z][:,0]
    y_axis = smhm[z][:,1]
    f = InterpolatedUnivariateSpline(x_axis, y_axis, k=4)
    x = np.linspace(x_axis[0], x_axis[-1], num=1000)
    plt.plot(x, f(x))
    cr_pts = f.derivative().roots()
    cr_pts = np.append(cr_pts, (x_axis[0], x_axis[-1]))  # also check the endpoints of the interval
    cr_vals = f(cr_pts)
    max_index = np.argmax(cr_vals)
    print("Maximum value {} at {}".format(cr_vals[max_index], cr_pts[max_index]))
    mhpeak[z] = cr_pts[max_index]

plt.scatter(redshift, list(mhpeak.values()))

np.savetxt('MhaloPeakB13_Planck.txt', np.array([redshift[:-1], list(mhpeak.values())[:-1]]))