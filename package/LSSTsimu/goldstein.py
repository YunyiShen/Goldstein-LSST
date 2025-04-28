from .simulations import single_simulated_lsst_data
from .SED import get_goldstein_luminosity, sed_frame_change
from .lightcurve import simulate_lsstLCraw
from .volumetric import volumetric_redshift
from .cuts import make_point_snr_cut, make_event_snr_length_cut

import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from datasets import Dataset, DatasetDict
import pandas as pd
from importlib import resources
import gc
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import re

def get_goldstein_params(filename):
    params = re.findall(r'[-+]?\d*\.\d+e[-+]?\d+', filename)
    params = [float(i) for i in params]
    return np.array(params)
    


def split_goldstein(path, test = 0.2, random_state = 42):
    '''
    split goldstein files into train and test
    '''
    # Step 1: List all files of a certain type
    all_files = [f for f in os.listdir(path) if f.endswith(".h5")]
    all_files_np = np.array(all_files)

    # Step 2: Randomly split (e.g., 80% train, 20% test)
    train_files, test_files = train_test_split(all_files_np, test_size=test, random_state=42)

    # Step 3: Save to CSV
    np.savetxt(f'{path}/goldstein_train_{random_state}.csv', train_files, fmt='%s', delimiter=',')
    np.savetxt(f'{path}/goldstein_test_{random_state}.csv', test_files, fmt='%s', delimiter=',')



def simulate_goldstein_lsst_data(list_of_events, 
                     num_samples = None, 
                     n_year = None,
                     param_weighting = lambda x: 1, 
                     r_v=2.42e-5, z_min=0.1, z_max=1.0, num_bins=100,
                     point_cut = make_point_snr_cut(min_snr = 3.),
                     event_cut = make_event_snr_length_cut(maxsnr = 5, minband = 2, min_measures = 10), 
                     file_name = "goldstein",
                     max_try = None, 
                     save_every = None, 
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
        param_weighting: a function return weight given goldstein parameters
        r_v: volumetric rate of supernovae (SNe/yr/Mpc^3)
        z_min: minimum redshift
        z_max: maximum redshift
        num_bins: number of redshift bins
        save_every: save every n tries
        point_cut: a function taking photoband, photomag, phototime, photoerror, photosnr, photomask, return a filtered version of them
        event_cut: a function taking photoband, photomag, phototime, photoerror, photosnr, photomask, return if the event is accepted
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
        save_every = 100
    events = []
    how_many_we_got = 0
    zs = []
    batch = 0
    total_hit = 0
    if n_year is not None:
        num_samples = len(nominate_z)
    print(f"start simulating... targeting samplesize {num_samples}")
    pbar = tqdm(nominate_z.tolist())
    weight_each_event = np.array([param_weighting(get_goldstein_params(i)) for i in list_of_events])
    weight_each_event = weight_each_event/np.sum(weight_each_event)
    for i, z in enumerate(pbar):
        # randomly choose a goldstein event
        event = np.random.choice(list_of_events, p = weight_each_event)
        Lnu, tim, lam, params = get_goldstein_luminosity(event)
        #breakpoint()
        data_we_got = single_simulated_lsst_data(Lnu,tim, # time in days
                     lam, 
                     z, 
                     params,
                     point_cut, event_cut,
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
            huggin_dataset.set_format("arrow")
            huggin_dataset.save_to_disk(filename_dump)
            batch += 1
            how_many_we_got = 0
            events = []
            gc.collect()
        if total_hit == num_samples and n_year is None: # we targeting fix number of samples
            break
        pbar.set_postfix(total_detection=f"{total_hit}")
    
    if len(events) > 0 and save: # if just want to get zs, for debugging
        filename_dump = f"{file_name}_batch{batch}"
        huggin_dataset = Dataset.from_list(events)
        huggin_dataset.set_format("arrow")
        huggin_dataset.save_to_disk(filename_dump)
    
    return np.array(zs)

