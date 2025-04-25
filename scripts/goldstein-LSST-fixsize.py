from LSSTsimu.simulations import simulate_goldstein_lsst_data
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import argparse
import warnings



def main():
    parser = argparse.ArgumentParser(description="Simulation setups")

    # Add arguments
    parser.add_argument('--maxsnr', type=float, default=5, help='Signal to noise ratio to declare detection')
    parser.add_argument('--minband', type=int, default=2, help='Minimum number of band to see that SNR')
    parser.add_argument('--minmeasures', type=int, default = 10, help='Minumum total number of measurements')
    parser.add_argument('--samplesize', type=int, default = 20000, help='Number of training samples targeted')
    parser.add_argument('--saveevery', type=int, default = 500, help='Number of training samples targeted')
    parser.add_argument('--jobid', type = int, default = 0, help = 'jobid for parallelism')
    parser.add_argument('--totaljobs', type = int, default = 1, help = 'total number of jobs for parallelism')
    parser.add_argument('--seed', type = int, default = 42, help = 'seed for job 0')
    # Parse the arguments
    args = parser.parse_args()
    
    

    maxsnr = args.maxsnr
    minband = args.minband
    min_measures = args.minmeasures
    samplesize = args.samplesize
    totaljobs = args.totaljobs
    jobid = args.jobid
    initseed = args.seed

    np.random.seed(jobid + initseed)

    print(f"maxsnr:{maxsnr}, minband:{minband}, min_measures:{min_measures}, samplesize:{samplesize}, totaljobs:{totaljobs}")

    if samplesize % totaljobs != 0:
        warnings.warn("Target sample size is not divisible to number of jobs, this may result undesired samplesize")

    subfolder = f"goldstein_{samplesize//1000}k_train_maxsnr{maxsnr}_minband{minband}_minmeasures{min_measures}"
    #breakpoint()
    goldstein_train_list = np.loadtxt('../goldstein/goldstein_train_42.csv', dtype=str, delimiter=',')
    train_list = np.array(["../goldstein/" + i for i in goldstein_train_list])
    zs_train = simulate_goldstein_lsst_data(train_list, samplesize//totaljobs, 
                                    maxsnr = maxsnr, minband = minband,
                                    min_measures = min_measures,
                                    len_per_filter = 20,
                                    save_every = args.saveevery,
                                file_name = f"../goldstein_lsstLC/{subfolder}/train/job_{jobid}")


    goldstein_test_list = np.loadtxt('../goldstein/goldstein_test_42.csv', dtype=str, delimiter=',')
    test_list = np.array(["../goldstein/" + i for i in goldstein_test_list])
    zs_test = simulate_goldstein_lsst_data(train_list, (samplesize//4)//totaljobs, 
                                    maxsnr = maxsnr, minband = minband,
                                    min_measures = min_measures,
                                    len_per_filter = 20,
                                    save_every = args.saveevery,
                                file_name = f"../goldstein_lsstLC/{subfolder}/test/job_{jobid}")
    
    np.savez(f"../goldstein_lsstLC/{subfolder}/zs_job_{jobid}.npz", train = zs_train, test = zs_test)



if __name__ == '__main__': 
    main()
