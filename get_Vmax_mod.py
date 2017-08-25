#!/usr/bin/env python

# Jean coupon - 2014
# Modified Louis Legrand 2017
# script to compute 1/Vmax (weight) from zmax
# and comoving volume

import numpy
import scipy.integrate as si
import sys
import getopt

# ----------------------------------------------------- #
# global variables
# ----------------------------------------------------- #

PI = 3.14159265358979323846
EPS = 1.0e-5
GET_VOL = 0

# ----------------------------------------------------- #
# help message
# ----------------------------------------------------- #


def usage():
    print('\n                 get_Vmax.py\n\n\
USAGE: get_Vmax.py z_low z_high [OPTIONS] \n\
       input (stdin)   : X...X zmax\n\
       output (stdout) : X...X 1/Vmax\n\n\
OPTIONS:\n\
       -h,--help    displays this message\n\
       -v,--vol     only returns the volume in Mpc^3 between z_low and z_lhigh\n\
       -c,--cosmo   H0,Omega_m,Omega_L cosmology (default: 72.0,0.258,0.742)\n\n\
ATTENTION: flat Universe approximation Vc = 4PI/3 DM^3!! \n\
(avoids double integration)\n')

# ----------------------------------------------------- #
# main
# ----------------------------------------------------- #


cosmo = [72.0, 0.258, 0.742]


def main(z_low, z_high, cosmo=cosmo):

    # ----------------------------------------------------- #
    # initialisation
    # ----------------------------------------------------- #

    d_low, error = si.quad(drdz, 0.0, z_low, args=(cosmo), epsrel=EPS)
    d_high, error = si.quad(drdz, 0.0, z_high, args=(cosmo), epsrel=EPS)

    # ----------------------------------------------------- #
    # comoving volume
    # ----------------------------------------------------- #

    return(4.0 * PI / 3.0 * (d_high**3 - d_low**3))
    sys.exit(0)

    # ----------------------------------------------------- #
    # 1/vmax
    # ----------------------------------------------------- #

    # norm = d_high**3 - d_low**3

    # for line in sys.stdin:

    #     col = line.split()
    #     z_max = float(col[len(col) - 1])
    #     weight = 0.0
    #     if(z_low + EPS < z_max and z_max < z_high - EPS):
    #         d_low,  error = si.quad(drdz, 0.0, z_low, args=(cosmo), epsrel=EPS)
    #         d_high, error = si.quad(drdz, 0.0, z_max, args=(cosmo), epsrel=EPS)
    #         V_max = (d_high**3 - d_low**3) / norm
    #         weight = 1.0 / V_max
    #     else:
    #         if(z_max > z_high):
    #             weight = 1.0

    #     for value in col[:len(col) - 1]:
    #         print(value),
    #     print(weight)


# ----------------------------------------------------- #
# functions
# ----------------------------------------------------- #

def drdz(z, params):
    "Computes drdz"
    H0 = params[0]
    Omega_M = params[1]
    Omega_L = params[2]

    c = 299792.458
    return c / (H0 * numpy.sqrt(Omega_M * (1.0 + z)**3.0 + Omega_L))


# ----------------------------------------------------- #
# run main
# ----------------------------------------------------- #
if __name__ == "__main__":
    main(sys.argv[1:])
