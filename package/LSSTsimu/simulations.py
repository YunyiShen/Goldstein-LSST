from .SED import get_goldstein_sed
from .lightcurve import simulate_lsstLC
from .volumetric import volumetric_redshift


def single_lsst_spectra(sed, z, spec_time, # time in days
                     spec_wavelengths, phase_range = None):
    """
    Simulate a single LSST spectrum together with  at redshift z.
    """
    if phase_range is None:
        phase_range = [-10, 30]

    spec_time_our_frame = spec_time * (1+z)# convert to observer frame
    spec_wavelengths_our_frame = spec_wavelengths * (1+z) # convert to observer frame
    reasonable_phase = spec_time[logical_and(spec_time > phase_range[0], spec_time < phase_range[1])]

