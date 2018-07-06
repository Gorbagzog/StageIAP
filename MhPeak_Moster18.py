# -*-coding:Utf-8 -*

"""Some code to compute the MhaloPeak with the Moster 2018 relation"""

import numpy as np
import matplotlib.pyplot as plt
import time

"""Parameters"""

z = np.array([0.1, 0.5, 1.0, 2.0, 4.0, 8.0])

M1 = np.array([11.80, 11.85, 11.95, 12.00, 12.05, 12.10])

beta = np.array([1.75, 1.70, 1.60, 1.55, 1.50, 1.30])

gamma = np.array([0.57, 0.58, 0.60, 0.62, 0.64, 0.64])

"""Compute MhaloPeak"""



MhaloPeak = M1 + 1/(beta + gamma) * np.log10(beta / gamma)

temp = [z[:-1], MhaloPeak[:-1]] # the redshift z=8 does not see a peak on the Figure 12.

np.savetxt('MhaloPeakM18.txt', temp)

"""Plot"""

# plt.figure()
# plt.plot(redshift, MhaloPeak)
# plt.show()