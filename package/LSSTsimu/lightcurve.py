import numpy as np
import pandas as pd
import astropy.units as u
#import rubin_sim.maf.db as db
from scipy.integrate import simpson as simps
from scipy.interpolate import interp1d
import numpy as np
import pysynphot as S
#import cosmolopy
pd.options.mode.chained_assignment = None 
from importlib import resources



def simulate_lsstLCraw(Flambda, time, # time in days
                    wavelength, 
                    filters_encoding = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5},
                    filters_loc = None,
                    LSSTschedule = pd.read_pickle(resources.path("LSSTsimu.data","baseline_v3.3_10yrs.pkl")),
                    len_per_filter = 20
                    ):
    
    """

    Simulate LSST light curves for a given SED surface and time.
        The SED should be in observer frame (i.e., has redshift), and the time should be in days.
    Parameters:
    - Flambda: 2D array of SED surface fluxes, specifically Fnu (time x wavelength)
    - time: 1D array of time values (in days)
    - wavelength: 1D array of wavelength values (in Angstrom)
    - filters_encoding: Dictionary mapping filter names to indices
    - filters_loc: Path to the LSST filter files "./data/filters/LSST"
    - LSSTschedule: Path to the LSST schedule file "./data/baseline_v3.3_10yrs.pkl"
    - len_per_filter: Number of observations per filter, things will be padded
    Returns:
    - photoband: 1D array of filter indices
    - photomag: 1D array of flux values
    - photoerror: 1D array of error values
    - phototime: 1D array of time values (in days)
    - photosnr: 1D array of SNR values
    - photomask: 1D array of mask values (1 for valid, 0 for invalid)
    - date_Flambda: 1D array of time values aligned with Flambda (in mjd days)
    """


    #LSSTschedule = "baseline_v3.3_10yrs.pkl",
    all_filters = filters_encoding.keys()

    ######## Physical constants #######
    h = 6.626e-27  # Planck's constant (erg·s)
    c = 2.998e10*u.cm/u.s   # Speed of light (cm/s)
    c_AAs = (c).to(u.AA/u.s).value
    
    LSST_rad = 321.15  # Radius in cm
    LSST_area = np.pi * LSST_rad**2  # Collecting area in cm²
    exp_time = 30.0  # Exposure time in seconds
    pix_scale = 0.2  # Pixel scale in arcsec/pixel
    readout_noise = 12.7  # Internal noise in electrons/pixel
    gain = 2.2  # Gain in electrons/photon
    fov = 3.5 #degrees on each side

    tmax_d = 100 # max phase of the transient in days
    tmax_s = tmax_d

    #cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725) # flat Lambda CDM cosmology
    #dist_cm = 10*cc.pc_cm #cosmo.luminosity_distance(z=0).to(u.cm).value
    
    ###### generate events ######
    # Generate num_points random coordinates
    num_events = 1
    ori_Flambda = Flambda
    ori_spec_time = time
    ori_spec_wavelengths = wavelength
    #if randomly generating locations, uncomment below
    # reset the original values

    # generate random locations
    eventRA = np.random.uniform(0, 2 * np.pi, num_events) * u.rad
    eventRA = eventRA.to(u.deg).value
    eventDec = np.arcsin(np.random.uniform(-1, 1, num_events)) * u.rad
    eventDec = eventDec.to(u.deg).value


    dRA = np.abs(LSSTschedule['fieldRA'].values - eventRA)
    dRA = np.minimum(dRA, 360 - dRA)
    dDec = np.abs(LSSTschedule['fieldDec'] - eventDec)

    df_obs = LSSTschedule[(dRA < fov/2) &
        (dDec < fov/2)]

    df_obs.dropna(inplace=True)

    tmin_obs = np.min(df_obs['observationStartMJD'])
    tmax_obs = np.max(df_obs['observationStartMJD'])
    # Pick a random start time for the transient
    try:
        start_mjd = np.random.uniform(tmin_obs, tmax_obs)
    except:
        start_mjd = 0

    # Exclude points beyond max_phase
    df_obs = df_obs[(df_obs['observationStartMJD'] >= start_mjd) & (df_obs['observationStartMJD'] <= (start_mjd+tmax_d))]
    df_obs.sort_values(by=['observationStartMJD'], inplace=True)

    time = time + start_mjd
    
    
    new_s = (df_obs['observationStartMJD'].values)
    df_obs = df_obs[(new_s >= np.min(time)) & (new_s <= np.max(time))]
    new_s = new_s[(new_s >= np.min(time)) & (new_s <= np.max(time))]
    #breakpoint()
    if len(df_obs) == 0: # no observations
        tmp = np.zeros(len_per_filter * 6)
        return tmp + np.nan, tmp + np.nan, tmp + np.nan, tmp + np.nan, tmp + np.nan, tmp, ori_spec_time+start_mjd 
    #print("Observation captured")
    Flambda = interp1d(time, Flambda, axis=0, bounds_error=False, fill_value=0)(new_s)
    time = new_s
            

    sky_brightness_mags = df_obs['skyBrightness'].values  # Sky brightness, in mag/arcsec^2
    fwhm_eff = df_obs['seeingFwhmEff'].values  # Seeing FWHM in arcsec
    filters = df_obs['filter'].values  # Filters for each observation

    # Effective number of pixels for a point source (based on the nightly seeing and pixel scale)
    n_eff = 2.266 * (fwhm_eff / pix_scale)**2

    bpass_dict = {}
    for flt in filters:
        if filters_loc is None:
            with resources.path("LSSTsimu.data","filters").joinpath(f"LSST_LSST.{flt}.dat") as path:
                band = pd.read_csv(path, header=None, sep='\s+', names=['w','t'])
        else:
            band = pd.read_csv(f"{filters_loc}/LSST_LSST.{flt}.dat", header=None, sep='\s+', names=['w','t'])    
        #band = pd.read_csv(f"{filters_loc}/LSST_LSST.{flt}.dat", header=None, sep='\s+', names=['w','t'])
        bpass_dict[flt] = S.ArrayBandpass(band['w'].values, band['t'].values)
        # Compute Signal, Background, and Magnitudes for each observation
    filts = df_obs['filter'].values
    bandpasses = np.array([bpass_dict[filt] for filt in filts])

    source_mag = []
        # Loop through each observation
    for i, bandpass in enumerate(bandpasses):
                # Get bandpass properties
        wavelengths = bandpass.wave  # Bandpass wavelengths in Angstrom
        throughput = bandpass.throughput  # Bandpass throughput

        spec_flux_at_time = Flambda[i, :]

                # Interpolate the spectrum onto the bandpass wavelength grid
        spec_flux_interp = interp1d(wavelength, spec_flux_at_time, bounds_error=False, fill_value=0)(wavelengths)

        # Perform the bandpass integrations
        I1 = simps(y=spec_flux_interp * throughput * wavelengths, x=wavelengths)  # Weighted integral
        I2 = simps(y=throughput / wavelengths, x=wavelengths)  # Normalization integral
        fnu = I1 / I2 / c_AAs #/ (4 * np.pi * dist_cm**2)  # Flux density in erg/s/Hz
        #breakpoint()
        # Calculate the apparent magnitude
        with np.errstate(divide='ignore'):
            mAB = -2.5 * np.log10(fnu) - 48.6  # AB magnitude
            # Append the magnitude for this observation
        source_mag.append(mAB)

        #convert from mag to Flam
    sky_fluxes = [10**(-0.4 * (mag + 48.6)) for mag in sky_brightness_mags]  # Flux per arcsec^2
    source_fluxes = [10**(-0.4 * (mag + 48.6)) for mag in source_mag] # Flux

    sky_counts = []
    source_counts = []

    for i, (sky_flux, source_flux, bandpass) in enumerate(zip(sky_fluxes, source_fluxes, bandpasses)):

        # Get bandpass properties
        wavelengths = bandpass.wave  # Bandpass wavelengths in Angstrom
        throughput = bandpass.throughput  # Bandpass throughput

        # Compute sky counts
        bpass_integral = np.trapz(throughput / (wavelengths * 1e-8), wavelengths * 1e-8)  # Throughput normalization
        sky_count = (exp_time * LSST_area / gain / h * sky_flux * bpass_integral * (pix_scale**2))  # Counts for sky per pixel
        sky_counts.append(sky_count)

        # Compute source counts
        source_count = (exp_time * LSST_area / gain / h * source_flux * bpass_integral) # Total counts
        source_counts.append(source_count)


        # Convert counts to numpy arrays
    sky_counts = np.array(sky_counts)
    source_counts = np.array(source_counts)
    if np.sum(source_counts) == 0:
        # no signal at all
        tmp = np.zeros(len_per_filter * 6)
        return tmp + np.nan, tmp + np.nan, tmp + np.nan, tmp + np.nan, tmp + np.nan, tmp, ori_spec_time+start_mjd 
            
    # Compute Signal-to-Noise Ratio (SNR)
    snr = source_counts / np.sqrt(source_counts / gain + (sky_counts / gain + readout_noise**2) * n_eff)
    #no_signal = np.where(snr == 0)
    no_signal = np.where(snr <= 0.1)
    snr[no_signal] = 1e-8
    # Compute magnitude errors and add noise
    mag_err = 1.09 / snr
    noisy_mags = source_mag + np.random.normal(0, mag_err)
    noisy_mags[no_signal] = np.nan
    photoband = []
    photomag = []
    photomask = []
    phototime = []
    photoerror = []
    photosnr = []
    encod = 0
    #breakpoint()
            
            
    for flt in all_filters:
                
        flt_idx = np.where(filts == flt)[0]
        len_this_band = len(flt_idx)
                #breakpoint()
        if len_this_band > len_per_filter:
            # randomly select len_per_filter observations
            #this_spec_time = time[flt_idx]
            #taking = np.where(np.logical_and(this_spec_time >= phase_lim_for_lc_too_long[0] * 86400, 
            #                                         this_spec_time <= phase_lim_for_lc_too_long[1]* 86400))
            #flt_idx = flt_idx[taking]
            #if len(taking[0]) > len_per_filter:
            flt_idx = np.random.choice(flt_idx, len_per_filter, replace=False)
            flt_idx = np.sort(flt_idx)
                #len_this_band = len_per_filter
            #len_this_band = len(flt_idx)
        photoband_tmp = np.zeros(len_per_filter)
        photoflux_tmp = np.zeros(len_per_filter)
        phototime_tmp = np.zeros(len_per_filter)
        photomask_tmp = np.zeros(len_per_filter)
        photoerror_tmp = np.zeros(len_per_filter)
        photosnr_tmp = np.zeros(len_per_filter)
        photoband_tmp[:len_this_band] = encod
        photoflux_tmp[:len_this_band] = noisy_mags[flt_idx]
        phototime_tmp[:len_this_band] = time[flt_idx]
        photomask_tmp[:len_this_band] = 1
        photoerror_tmp[:len_this_band] = mag_err[flt_idx]
        photosnr_tmp[:len_this_band] = snr[flt_idx]

        #plt.scatter(phototime_tmp, photoflux_tmp, label=flt)

        anyna = np.where(np.logical_not(np.isfinite(photoflux_tmp)))
        photoflux_tmp[anyna] = 0
        photomask_tmp[anyna] = 0

        photoband.append(photoband_tmp)
        photomag.append(photoflux_tmp)
        photomask.append(photomask_tmp)
        phototime.append(phototime_tmp)
        photoerror.append(photoerror_tmp)
        photosnr.append(photosnr_tmp)
        encod += 1
    
    # flat
    photoband = np.array(photoband).flatten()
    photomag = np.array(photomag).flatten()
    phototime = np.array(phototime).flatten()
    photoerror = np.array(photoerror).flatten()
    photomask = np.array(photomask).flatten()
    photosnr = np.array(photosnr).flatten()


    '''
    plt.ylim((-19.5,-16.5))
    plt.gca().invert_yaxis()
    plt.legend()
    plt.xlabel('Time (days)')
    plt.ylabel('Magnitude')
    plt.title('Light curves')
    plt.show()
    plt.savefig('light_curves.png')
    #breakpoint()
    '''
    #breakpoint()  
    return photoband, photomag, phototime, photoerror, photosnr, photomask, ori_spec_time+start_mjd


import matplotlib.pyplot as plt
def plot_lsst_lc(photoband, photomag, phototime, photoerror, photomask, ax = None):
    lsst_bands = ["u", "g", "r", "i", "z", "y"]
    colors = ["purple", "blue", "darkgreen", "lime", "orange", "red"]
    photoband = photoband[photomask == 1]
    photomag = photomag[photomask == 1]
    phototime = phototime[photomask == 1]
    photoerror = photoerror[photomask == 1]
    if ax is None:
        fig, ax = plt.subplots()
    for bnd in range(6):
        idx = np.where(photoband == bnd)[0]
        if len(idx) > 0:
            ax.scatter(phototime[idx], photomag[idx], label=lsst_bands[bnd], s=5, color=colors[bnd])
            ax.plot(phototime[idx], photomag[idx], color=colors[bnd], alpha=0.5)
            ax.errorbar(phototime[idx], photomag[idx], 
                        yerr=photoerror[idx], fmt='o', 
                        markersize=5, capsize=3, color=colors[bnd])
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Magnitude')
    ax.set_title('LSST Light Curve')
    ax.invert_yaxis()
    ax.legend()
    if ax is None:
        return fig
    #return ax


