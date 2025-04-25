import numpy as np
import cosmolopy
from astropy.cosmology import FlatLambdaCDM


def volumetric_redshift(n_events, n_year = None,
                        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725),
                        r_v=2.42e-5, z_min=0.1, z_max=1.0, num_bins=100):
    """
    generate redshift distribution of supernovae
    Simulate the redshift distribution of supernovae using a volumetric rate.
    Parameters:
    - n_events: Number of events to simulate
    - n_year: Number of years, if given, n_event will be determined by n_year
    - r_v: Volumetric rate of supernovae (SNe/yr/Mpc^3)
    - z_min: Minimum redshift
    - z_max: Maximum redshift
    - num_bins: Number of redshift bins
    Returns:
    - supernovae_counts: List of supernovae counts in each redshift bin
    """

    r_v = 2.42e-5  # volumetric rate is SNe/yr/Mpc^3
    z_min, z_max = 0.1, 1.0  # redshift range
    num_bins = 100
    z_bins = np.linspace(z_min, z_max, num_bins)
    weight = np.zeros(num_bins - 1)

    # Calculate number of supernovae in each redshift bin
    supernovae_counts = []

    for i in range(len(z_bins) - 1):
        z_low, z_high = z_bins[i], z_bins[i+1]
        #d_L = cosmo.luminosity_distance(z)  # Mpc
        z_avg = (z_high + z_low)/2
        dVdz_avg = cosmo.differential_comoving_volume(z_avg).value  # Mpc^3/(1+z) dz
        N_z = r_v * dVdz_avg * (z_high - z_low)  # SNe in the redshift bin
        weight[i] = N_z
    if n_year is not None:
        n_events = n_year * np.sum(weight)  # Scale to the number of years
    weight = weight / np.sum(weight)  # Normalize weights
    supernovae_counts = np.random.multinomial(n_events, weight)  # Sample from multinomial distribution
    # Generate redshifts for each supernova
    redshifts = []
    for i in range(len(z_bins) - 1):
        z_low, z_high = z_bins[i], z_bins[i+1]
        N_z = supernovae_counts[i]
        if N_z > 0:
            sampled_redshifts = np.random.uniform(z_low, z_high, int(N_z)) # assume that we uniformly sample redshifts within that bin, obviously this is less of an effect as the number of bins increases
            redshifts += [sampled_redshifts]
    redshifts = np.concatenate(redshifts)
    np.random.shuffle(redshifts)
    
    # Sort redshifts
    #redshifts.sort()
    #return redshifts
    return supernovae_counts, redshifts


