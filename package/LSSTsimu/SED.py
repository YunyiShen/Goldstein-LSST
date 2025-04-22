import h5py
import numpy as np
import os
from cosmolopy import magnitudes, cc
import math
import astropy.units as u
import pysynphot as S
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import sys
from astropy.cosmology import FlatLambdaCDM


h = 6.626e-27  # Planck's constant (erg·s)
c = 2.998e10*u.cm/u.s   # Speed of light (cm/s)
k_B = 1.381e-16  # Boltzmann's constant (erg/K)
pi = np.pi
c_AAs = (c).to(u.AA/u.s).value

## simple blackbody function

def blackbody_flux_density(wavelength, temperature):
    """
    Calculate blackbody flux density B_lambda(T) in erg/s/cm^2/Å.

    Parameters:
        wavelength (ndarray): Wavelengths in Angstroms.
        temperature (float): Temperature in Kelvin.

    Returns:
        ndarray: Flux density in erg/s/cm^2/Å.
    """
    wavelength_cm = wavelength * 1e-8  # Convert Å to cm
    exponent = h * c.value / (wavelength_cm * k_B * temperature)
    bb_flux = (2 * h * c.value**2 / wavelength_cm**5) / (np.exp(exponent) - 1)
    return bb_flux * 1e-8  # Convert from per cm to per Å


## simple sed
def compute_sed(time, radius, temperature, wavelengths, redshift=0):
    """
    Compute the SED surface for a time-evolving blackbody.

    Parameters:
        time (ndarray): Time array in seconds.
        radius (ndarray): Radius array in cm corresponding to time.
        temperature (ndarray): Temperature array in K corresponding to time.
        wavelengths (ndarray): Wavelengths in Angstroms.
        redshift (float): Redshift to the transient (default 0).

    Returns:
        ndarray: SED surface (time x wavelengths) in erg/s/cm^2/Angstrom.
    """

    sed_surface = np.zeros((len(time), len(wavelengths)))
    for i, (t, r, temp) in enumerate(zip(time, radius, temperature)):
        # Calculate the blackbody spectrum
        bb_flux = blackbody_flux_density(wavelengths, temp)
        # Scale by the projected area and distance
        sed_surface[i, :] = pi * (r) **2 * bb_flux / (1 + redshift) # projected surface area

    #time dilation and redshifting of wavelengths and flux values
    time *= (1+redshift)
    wavelengths *= (1+redshift)

    return time, wavelengths, sed_surface


def get_goldstein_luminosity(filename):
    '''
    read in the goldstein SED file, get its identity
    Returns:
    - tim: time in days
    - lam: wavelength in Angstroms
    - Flambda: flux in erg/s/cm^2/A
    - peak: index of the peak flux
    - params: parameters of the simulation
    '''
    with h5py.File(filename, 'r') as f:
        tim = np.array(f['time'])/86400 # in days
        lam = magnitudes.nu_lambda(np.array(f['nu'])) # frequency to wavelength
        Lnu = np.array(f['Lnu']) # luminosity in erg/s/angstroms
        Fnu = np.array(f['Lnu']) / (4. * math.pi * (10*cc.pc_cm)**2) # to absolute flux, at 10 pc
        total_brightness = np.sum(Fnu, axis=1)
        peak = tim[np.argmax(total_brightness)]
        sed_surface = Fnu * c_AAs/lam[None,:]**2 # Flambda
        # get simulation parameters from filename
        params = re.findall(r'[-+]?\d*\.\d+e[-+]?\d+', filename)
        params = [float(i) for i in params]
        params = np.array(params)
    
    return tim, lam, Fnu, Lnu, peak, params


def sed_frame_change(Lnu, z, time, wavelengths, cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)):
    """
    get SED Flambda to the observer frame.

    Parameters:
        Lnu (ndarray): luminosity in erg/s/angstroms at source.
        z (float): Redshift to the transient.
        time (ndarray): Time array in seconds.
        wavelengths (ndarray): Wavelengths in Angstroms.
        cosmo (FlatLambdaCDM): Cosmology object (default is FlatLambdaCDM with H0=70, Om0=0.3, Tcmb0=2.725).

    Returns:
        tuple: Time in observer frame, wavelengths in observer frame, and SED surface in observer frame.
    """
    time = time * (1 + z)
    wavelengths = wavelengths * (1 + z)
    Llum = cosmo.luminosity_distance(z) # Mpc
    Llum = Llum.to(u.cm).value # luminosity distance in cm
    Fnu = Lnu / (4 * pi * Llum**2 * (1 + z)) # to flux on earth
    Flambda = Fnu * c_AAs / wavelengths[None, :]**2 # Flambda

    return time, wavelengths, Flambda