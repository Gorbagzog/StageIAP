# -*-coding:Utf-8 -*

"""
Fit a HMF function to the data outputs of Bolshoi-Planck15 given by Behroozi et al. 18 in the Universe machine code.
This is in scope of using th epeak halo mass of the halo as the HMF inour SHAM code.

Started on January 18th 2019 by Louis Legrand at IAP and IAS.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from colossus.lss import mass_function
from colossus.halo import mass_so
from colossus.lss import peaks
from colossus.cosmology import cosmology
cosmology.setCosmology('planck15')


def load_um_hmf():
    """Load the hmf of universe machine code, which is using the peak halo mass and not the actual mass of the halo.
    The simulation is in the Planck 2015 cosmology.

    Returns:
    hmf : a dictionary with the redshift as index and giving the HMFs in numpy array.
        Peak halo masses in Msun
        Number densities in comoving Mpc^-3 dex^-1.
        Columns of the HMFS[a] array = Log10(HM) Number_Density Err+ Err- Satellite_fraction Err+ Err- HM_Left_Edge HM_Right_Edge
    redshifts : numpy array of the redshifts where the HMFs are evaluated
    """
    path = '../Data/umachine-edr/data/hmfs/'
    list_files = os.listdir(path)
    hmf = {}
    for file in list_files:
        a = float(file[5:-4])
        z = 1/a - 1
        hmf[z] = np.loadtxt(path + file).T
    redshifts = np.array(sorted(hmf.keys()))

    return hmf, redshifts


def plot(hmf, z):
    """Plot the hmf of bolshoi planck at a given redshift"""
    # plt.figure()
    plt.errorbar(hmf[z][0], hmf[z][1], yerr=[hmf[z][2], hmf[z][3]])


def plot_all(hmf, redshifts):
    plt.figure()
    for z in redshifts[redshifts < 5][::10]:
        plt.errorbar(hmf[z][0], hmf[z][1], yerr=[
                     hmf[z][2], hmf[z][3]], label=str(z), fmt='o')
    plt.yscale('log')
    plt.xlabel('Peak halo masses in Msun')
    plt.ylabel('Number densities in comoving Mpc^-3 dex^-1')
    plt.legend()


def despali(hmf, z):
    """Get the Despali HMF on the HMF from the Bolshoi PLanck simulation.

    Inputs:
        hmf : HMFs dictionnary from load_um_hmf
        z : redshift corresponding to one of the keys of hmf
    """
    mfunc = mass_function.massFunction(
        hmf[z][0], z, mdef='200m', model='despali16')


def modelDespali16_fit(theta, M, z, mdef, deltac_args={'corrections': True}):
	"""
	The mass function model of Despali et al 2016.
    This function is updated from the Colossus function to allow to fit paramters on the BP15 HMF.

    The parameters were fit for a number of different mass definitions, redshifts, and cosmologies.
    Here, we use the most general parameter set using the rescaling formula for redshift and mass
    definitions given in Equation 12.

    Furthermore, the user can choose between results based on
    conventional SO halo finding and an ellipsoidal halo finder.

	Parameters
	-----------------------------------------------------------------------------------------------
	theta: array_like
        Contains the 7 free parameters of Depsali HMF to be fitted
    M: array_like
		Mass in M_{\odot}/h; can be a number or a numpy array.
	z: float
		Redshift
	mdef: str
		The mass definition to which M corresponds. See :doc:`halo_mass` for details.

	Returns
	-----------------------------------------------------------------------------------------------
	dndlnM: array_like
		The halo mass function :math:`dndlnM(M)` in (Mpc/h)^-3.
	"""
    cosmo = cosmology.getCurrent()

    R = peaks.lagrangianR(M)
    sigma = cosmo.sigma(R, z)


	Delta = mass_so.densityThreshold(z, mdef)
	Delta_vir = mass_so.densityThreshold(z, 'vir')
	x = np.log10(Delta / Delta_vir)

    # A = -0.1362 * x + 0.3292
    # a = 0.4332 * x**2 + 0.2263 * x + 0.7665
    # p = -0.1151 * x**2 + 0.2554 * x + 0.2488
    # theta = [-0.1262, 0.3292, 0.4332, 0.2263, 0.7665, -0.1551, 0.2554, 0.2488]
    A1, A0, a2, a1, a0, p2, p1, p0 = theta
    A = A1 * x + A0
    a = a2* x**2 + a1 * x + a0
    p = p2 * x**2 + p1 * x + p0

	delta_c = peaks.collapseOverdensity(z=z, **deltac_args)

	nu_p = a * delta_c**2 / sigma**2
	f = 2.0 * A * np.sqrt(nu_p / 2.0 / np.pi) * \
            np.exp(-0.5 * nu_p) * (1.0 + nu_p ** -p)

    mfunc = mass_function.convertMassFunction(f, M, z, 'f', q_out)
	return f
