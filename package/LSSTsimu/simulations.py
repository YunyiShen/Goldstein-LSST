from .SED import get_goldstein_luminosity, sed_frame_change
from .lightcurve import simulate_lsstLC
from .volumetric import volumetric_redshift
from datasets import Dataset
from astropy.cosmology import FlatLambdaCDM

def single_lsst_data(Lnu,time, # time in days
                     wavelength, 
                     z, 
                     phase_range = None,
                     n_phase = 5,
                     other = None
                     cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725),
                     LCconfig = None,
                     ):
    """
    Simulate a single LSST light curve together with some spectra measures at redshift z.
    """
    if phase_range is None:
        phase_range = [-10, 30]
    Fnu = Lnu / (4. * math.pi * (10*cc.pc_cm)**2) # to absolute flux, at 10 pc
    total_brightness = np.sum(Fnu, axis=1)
    peak = tim[np.argmax(total_brightness)]

    # frame change to get SED
    ourFlambda, ourtime, ourwavelength = sed_frame_change(Lnu, time, wavelength, z, cosmo=cosmo)
    # get the light curve
    photoband, photoflux, phototime, photomask = simulate_lsstLC(ourFlambda, ourtime, # time in days
                    ourwavelength, *LCconfig)
    # get the spectra
    reasonable_phase = ourtime[logical_and(ourtime > phase_range[0], ourtime < phase_range[1])]
    phase_see_spectra = np.choice(reasonable_phase, n_phase, replace=False)
    


    

