from MCMC_SHMR_main_clean import *
import matplotlib.gridspec as gridspec



cosmology.setCosmology('planck15')


# z = [0.0, 1.0, 2.0, 4.0]
# M = 10**np.arange(11.0, 15.5, 0.1)

# plt.figure()
# plt.xlabel('M200m')
# plt.ylabel('dn/dln(M)')
# plt.loglog()
# plt.xlim(1E11, 4E15)
# plt.ylim(1E-7, 1E-1)
# for i in range(len(z)):
#     mfunc = mass_function.massFunction(
#         M, z[i], mdef='200m', model='tinker08', q_out='dndlnM')
#     plt.plot(M, mfunc, '-', label='z = %.1f' % (z[i]))
# plt.legend()


directory = '../MCMC_2019-1-23T11-58-38'
params = load_params(directory + '/MCMC_param.ini')

names = ['despali16_Bolshoifit', 'despali16', 'tinker08',
         'watson13', 'bocquet16', 'bhattacharya11']

colors = {'despali16_Bolshoifit': '#BE00BF',
    'despali16':'black',
    'tinker08':'#BE00BF',
    'watson13':'#0C8A8A',
    'bocquet16':'#BE00BF',
    'bhattacharya11':'#0C8A8A'}

fmt = {'despali16_Bolshoifit': '-',
       'despali16': '-',
       'tinker08': '--',
       'watson13': '-.',
       'bocquet16': '--',
       'bhattacharya11': ':'}

label = {'despali16_Bolshoifit': 'BP15 fit',
         'despali16': None,
         'tinker08': 'Tinker+08',
         'watson13': 'Watson+13',
         'bocquet16': 'Bocquet+16',
         'bhattacharya11': 'Bhattacharya+11'}

hmf = {}

smf = load_smf(params)

for hmf_name in names:
    hmf[hmf_name] = []
    if hmf_name == 'despali16' or hmf_name == 'tinker08' or hmf_name == 'watson13' or hmf_name == 'bocquet16' or hmf_name == 'bhattacharya11':
        """Use the Colossus module for the HMF"""
        print('Use '+hmf_name+' HMF in Planck15 cosmo from Colossus module')
        if hmf_name == 'watson13' or hmf_name == 'bhattacharya11':
            # print(hmf_name)
            mdef = 'fof'
        elif hmf_name == 'bocquet16':
            mdef = '200m'
        elif hmf_name == 'tinker08':
            # Cannot use Mvir for tinker becasue the threshold cannont be lower than 200*rho_m
            mdef = '200m'
        elif hmf_name == 'despali16':
            mdef = 'vir'
        print('Use ' + mdef + ' for the SO defintion in ' + hmf_name)
        cosmo = cosmology.setCosmology('planck15')
        redshift_haloes = params['redshiftsbin']
        M = 10**np.arange(8.0, 20, 0.01)  # Mass in Msun / h
        for i in range(params['numzbin']):
            hmf[hmf_name].append(
                np.transpose(
                    np.array(
                        [np.log10(M / cosmo.h),
                         np.log10(mass_function.massFunction(M, redshift_haloes[i], mdef=mdef, model=hmf_name, q_out='dndlnM') * np.log(10) * cosmo.h**3
                                  ## Mass functions are in h^3 Mpc^-3, and need to multiply by ln(10) to have dndlog10m
                                  )]
                    )
                )
            )

    if hmf_name == 'despali16_Bolshoifit':
        """Use my fit on the Bolshoi Planck simulation of the Despali HMF.
        If they did like in Behroozi 2013 then the BP simulations outputs in the Universe machine define the halo masses as the virial mass overdensity criterion
        of Bryan and Norman 97, and the peak progenitor mass for the mass of the satellites.
        The HMF gives the total number of halos, (central and satellites)."""
        print('Use ' + hmf_name + ' HMF in Planck15 cosmo from Colossus module function fited on the Bolshoi-Planck15 simulations of Mhalo at peak mass')
        mdef = 'mvir'
        # theta_best_fit = np.array(
        #     [0.333, 0.794, 0.247])
        # theta_best_fit = np.array(
        #     [0.4752772505046221, 0.7532675208782209, 0.3520302943602699])
        theta_best_fit = np.array(
            [0.33090366262139637, 0.8311237426711129, 0.3512604117780071])  # Best fit supposing UM gives masses in Msun -> transformed them by Msun * h
        cosmo = cosmology.setCosmology('planck15')
        redshift_haloes = params['redshiftsbin']
        M = 10**np.arange(8.0, 20, 0.01) # Mass in Msun / h
        for i in range(params['numzbin']):
            hmf[hmf_name].append(
                np.transpose(
                    np.array(
                        [np.log10(M / cosmo.h),
                         np.log10(fitdespali16.modelDespali16_fit(
                             theta_best_fit, M, redshift_haloes[i], deltac_args={'corrections': True})*np.log(10)*cosmo.h** 3
                             # Mass functions are in h^3 Mpc^-3, and need to multiply by ln(10) to have dndlog10m
                            )]
                        )
                    )
                )

plt.close('all')
names = ['despali16_Bolshoifit', 'despali16', 'tinker08',
         'watson13', 'bocquet16', 'bhattacharya11']
ref = 'despali16'
# names.remove(ref)

# gs = gridspec.GridSpec(3,1)
fig, axes = plt.subplots(3, sharex=True, sharey=True)
for idxax, idxz in enumerate([0, 4, 9]):
    ax = axes[idxax]
    ax.set_xlim(10 ** 9, 10 ** 15)
    ax.set_ylim(-0.7, 0.7)
    ax.set_xscale('log')
    ax.tick_params(labelsize=13)
    if idxax < 2:
        ax.tick_params(axis='both', direction='in', which='both')
    for name in names:
        line1, = ax.plot(10**hmf[ref][idxz][:, 0],
                         hmf[name][idxz][:, 1] - hmf[ref][idxz][:, 1],
                         color=colors[name], linestyle=fmt[name], linewidth=1, label=label[name])
        if name == 'tinker08':
            line1.set_dashes([8, 8])
    if idxax < 1:
        ax.legend(frameon=False, loc=9, ncol=3)
    ax.text(2*10**9, -0.5, str(params['redshifts'][idxz]) +
            '<z<' + str(params['redshifts'][idxz + 1]))
plt.xlabel('halo mass [$M_\odot]$', size=17)
plt.ylabel(
    'log($\Phi/\Phi_{Despali+16}$)', size=17, **{'y':+1.5})
# fig.text(0.04, 0.5, 'common Y', va='center', rotation='vertical')
fig.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
