from .SED import sed_frame_change
from .lightcurve import simulate_lsstLCraw
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import numpy as np
import pandas as pd
from importlib import resources
from .cuts import make_point_snr_cut, make_event_snr_length_cut



def single_simulated_lsst_data(Lnu,time, # time in days
                     wavelength, 
                     z, 
                     parameters,
                     point_cut = make_point_snr_cut(min_snr = 3.),
                     event_cut = make_event_snr_length_cut(maxsnr = 5, minband = 2, min_measures = 10), 
                     len_per_filter = 20,
                     phase_range = None,
                     cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725),
                     wavelength_range = [2000, 10000],
                     LSSTschedule = pd.read_pickle(resources.path("LSSTsimu.data","baseline_v3.3_10yrs.pkl"))
                     ):
    """
    Simulate a single LSST light curve together with some spectra measures at redshift z, and some parameters about the event other than redshift.

    Arguments:
        Lnu: luminosity, in Lnu, at the source
        time: the time of the SED, in days
        wavelength: the wavelength of the SED, at the source
        z: the redshift
        parameters: the parameters of simulation
        point_cut: a function taking photoband, photomag, phototime, photoerror, photosnr, photomask, return a filtered version of them
        event_cut: a function taking photoband, photomag, phototime, photoerror, photosnr, photomask, return if the event is accepted
        len_per_filter: the number of points per filter, default is 20 be padded
        phase_range: the range of phases to choose from for spectra
        cosmo: cosmology object to use for distance calculations
        wavelength_range: range of wavelengths to consider for spectra

    return:
      a dictionary with the following keys
        LC_band: the band of the light curve
        LC_mjd: the time of the light curve
        LC_mag: the magnitude of the light curve
        LC_error: the error of the light curve
        LC_snr: the SNR of the light curve
        LC_mask: the mask of the light curve
        redshift: the redshift of the light curve
        goldstein_params: the parameters of the light curve
        spectrum_mjd: the time of the spectrum on earth
        spectrum: the spectrum, in Fnu, on earth
        spectrum_wavelength: the wavelength of the spectrum, on earth
        (no more) Flambda_ours: the SED, in Flambda, on earth
        sed_time_ours: the time of the SED, in mjd
        (no more) sed_wavelength_ours: the wavelength of the SED, in Angstrom
        Lnu_emission: the SED, in Lnu, at the source
        sed_time_emission: the time of the SED, in days from ``detection''
        sed_wavelength_emission: the wavelength of the SED, at the source

        or none if the SNR is too low
    """
    
    # Fnu = Lnu / (4. * math.pi * (10*cc.pc_cm)**2) # to absolute flux, at 10 pc
    c = 2.998e10*u.cm/u.s
    # frame change to get SED
    ourFlambda, ourtime, ourwavelength, Fnu = sed_frame_change(Lnu, time, wavelength, z, cosmo=cosmo)
    #total_brightness_ours = np.sum(ourFlambda, axis=1)
    # get the light curve
    photoband, photomag, phototime, photoerror, photosnr, photomask, mjd = simulate_lsstLCraw(ourFlambda, ourtime, # time in days
                    ourwavelength, len_per_filter = len_per_filter, LSSTschedule = LSSTschedule)

    photoband, photomag, phototime, photoerror, photosnr, photomask = point_cut(photoband, photomag, phototime, photoerror, photosnr, photomask)
    pass_snr = event_cut(photoband, photomag, phototime, photoerror, photosnr, photomask)
    if not pass_snr:
        return None

    res = {}
    ### light curve ###
    res['LC_band'] = photoband
    res['LC_mjd'] = phototime
    res['LC_mag'] = photomag
    res['LC_error'] = photoerror
    res['LC_snr'] = photosnr
    res['LC_mask'] = photomask
    
    ### redshift and parameters ###
    res['redshift'] = np.array(z)
    res['goldstein_params'] = parameters

    ### spectra ###
    if phase_range is None:
        phase_range = [0, 50]
    rand_int = np.random.randint(phase_range[0], phase_range[1])
    
    res['spectrum_mjd'] = mjd[rand_int]
    spectrum = ourFlambda * (ourwavelength[None,:]**2 / (c)) # convert to Fnu
    spectrum = spectrum[:, np.logical_and(ourwavelength >= wavelength_range[0], 
                        ourwavelength <= wavelength_range[1])]
    res['spectrum'] = spectrum[rand_int].value
    res['spectrum_wavelength'] = ourwavelength[np.logical_and(ourwavelength >= wavelength_range[0], 
                        ourwavelength <= wavelength_range[1])]
    
    ### whole sed ###
    #res['Flambda_ours'] = ourFlambda
    res['sed_time_ours'] = mjd
    #res['sed_wavelength_ours'] = ourwavelength

    res['Lnu_emission'] = Lnu
    res['sed_time_emission'] = time
    res['sed_wavelength_emission'] = wavelength
    
    return res