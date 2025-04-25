from .SED import get_goldstein_luminosity, sed_frame_change
from .lightcurve import simulate_lsstLCraw, snr_cut
from .volumetric import volumetric_redshift
from datasets import Dataset
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import json
from tqdm import tqdm
import numpy as np
from datasets import Dataset, DatasetDict
import pandas as pd
from importlib import resources



def single_goldstein_lsst_data(Lnu,time, # time in days
                     wavelength, 
                     z, 
                     parameters,
                     maxsnr = 15, minband = 2, min_measures = 10,
                     len_per_filter = 20,
                     phase_range = None,
                     cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725),
                     wavelength_range = [2000, 10000],
                     LSSTschedule = pd.read_pickle(resources.path("LSSTsimu.data","baseline_v3.3_10yrs.pkl"))
                     ):
    """
    Simulate a single LSST light curve together with some spectra measures at redshift z.

    Arguments:
        Lnu: the SED, in Lnu, at the source
        time: the time of the SED, in days
        wavelength: the wavelength of the SED, at the source
        z: the redshift
        parameters: the parameters of simulation
        maxsnr: maximum SNR for light curve
        minband: minimum number of bands for light curve
        min_measures: minimum measurements taken
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
    pass_snr = snr_cut(photoband, photosnr, photomag, maxsnr, minband)
    if not pass_snr or np.sum(photomask) < min_measures:
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


def simulate_goldstein_lsst_data(list_of_events, 
                     num_samples = None, 
                     n_year = None,
                     file_name = "goldstein",
                     max_try = None, 
                     r_v=2.42e-5, z_min=0.1, z_max=1.0, num_bins=100,
                     save_every = None, 
                     maxsnr = 15, minband = 2,
                     min_measures = 10,
                     len_per_filter = 20,
                     phase_range = None,
                     cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725),
                     wavelength_range = [2000, 10000], 
                     LSSTschedule = pd.read_pickle(resources.path("LSSTsimu.data","baseline_v3.3_10yrs.pkl")),
                     save = True
                     ):
    """
    Simulate a list of LSST light curves together with some spectra measures at redshift z.
    Arguments:
        list_of_events: a list of events that gives goldstein results
        num_samples: number of samples to generate
        n_year: years of survey, if given sample size will be determined by volumetric rate
        max_try: maximum number of tries to get a valid sample
        r_v: volumetric rate of supernovae (SNe/yr/Mpc^3)
        z_min: minimum redshift
        z_max: maximum redshift
        num_bins: number of redshift bins
        save_every: save every n tries
        maxsnr: maximum SNR for light curve
        minband: minimum number of bands for light curve
        min_measures: minimum measurements taken
        len_per_filter: the number of points per filter, default is 20 be padded
        phase_range: the range of phases to choose from for spectra
        cosmo: cosmology object to use for distance calculations
        wavelength_range: range of wavelengths to consider for spectra
        LSSTschedule: a pd dataframe for LSST schedule
        save: whether to save data onto disk
    return:
        a list of dictionaries with the same keys as above
    """
    assert num_samples is not None or n_year is not None, "one of num_samples or n_year has to be given"

    if max_try is None and n_year is None:
        max_try = num_samples * 200
    
    _, nominate_z = volumetric_redshift(max_try, n_year = n_year,
                        cosmo = cosmo,
                        r_v=r_v, z_min=z_min, z_max=z_max, num_bins=num_bins)
    if save_every is None:
        save_every = len(nominate_z) // 20
    events = []
    how_many_we_got = 0
    zs = []
    batch = 0
    total_hit = 0
    print("start simulating...")
    pbar = tqdm(nominate_z.tolist())
    for i, z in enumerate(pbar):
        # randomly choose a goldstein event
        event = np.random.choice(list_of_events)
        Lnu, tim, lam, params = get_goldstein_luminosity(event)
        #breakpoint()
        data_we_got = single_goldstein_lsst_data(Lnu,tim, # time in days
                     lam, 
                     z, 
                     params,
                     maxsnr, minband,min_measures,
                     len_per_filter,
                     phase_range,
                     cosmo,
                     wavelength_range,
                     LSSTschedule
                     )
        if data_we_got is not None:
            events.append(data_we_got)
            how_many_we_got += 1
            total_hit += 1
            zs.append(z)
            #breakpoint()
        
        if how_many_we_got == save_every and save:
            
            filename_dump = f"{file_name}/batch{batch}"
            #breakpoint()
            huggin_dataset = Dataset.from_list(events)
            huggin_dataset.save_to_disk(filename_dump)
            batch += 1
            how_many_we_got = 0
            events = []
        if how_many_we_got == num_samples and n_year is None: # we targeting fix number of samples
            break
        pbar.set_postfix(total_detection=f"{total_hit}")
    
    if len(events) > 0 and save: # if just want to get zs, for debugging
        filename_dump = f"{file_name}_batch{batch}"
        huggin_dataset = Dataset.from_list(events)
        huggin_dataset.save_to_disk(filename_dump)
    
    return np.array(zs)

