

"""
Small script to make interpolation of the SMF of galaxies and halos.
"""

from scipy.interpolate import interp1d

Mstar = interp1d(Ngal, np.linspace(mmingal, mmaxgal, num=n))

Mhalo = interp1d(Nhalo, np.linspace(mminhalo, mmaxhalo, num=n))

x = np.linspace(Ngal[0], Ngal[-1], 10000)

# plt.plot(Mhalo(x), Mstar(x))
# plt.xlabel('Halo Mass, $log(M_{\odot})$')
# plt.ylabel('S tellar Mass, $log(M_{\odot})$')
# plt.title('Jean Coupon vs Behroozi, z='+str(zmin)+'-'+str(zmax))
# plt.show()

plt.loglog(10**Mhalo(x), (10**Mstar(x))/(10**Mhalo(x)))
plt.ylabel('$M_{*}/M_{h}$')
plt.xlabel('$M_{h}$  [$M_{\odot}]$')
plt.title('Jean Coupon vs Behroozi, z='+str(zmin)+'-'+str(zmax))
plt.show()
