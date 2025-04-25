from LSSTsimu.simulations import simulate_goldstein_lsst_data
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import argparse



def main():
    parser = argparse.ArgumentParser(description="Simulation setups")

    # Add arguments
    parser.add_argument('--maxsnr', type=float, default=5, help='Signal to noise ratio to declare detection')
    parser.add_argument('--minband', type=int, default=2, help='Minimum number of band to see that SNR')
    parser.add_argument('--minmeasures', type=int, default = 10, help='Minumum total number of measurements')
    parser.add_argument('--samplesize', type=int, default = 20000, help='Number of training samples targeted')
    parser.add_argument('--saveevery', type=int, default = 500, help='Number of training samples targeted')

    # Parse the arguments
    args = parser.parse_args()

    maxsnr = args.maxsnr
    minband = args.minband
    min_measures = args.minmeasures
    samplesize = args.samplesize

    subfolder = f"goldstein_{samplesize//1000}k_train_maxsnr{maxsnr}_minband{minband}_minmeasures{min_measures}"

    goldstein_train_list = np.loadtxt('../goldstein/goldstein_train_42.csv', dtype=str, delimiter=',')
    train_list = np.array(["../goldstein/" + i for i in goldstein_train_list])
    zs_train = simulate_goldstein_lsst_data(train_list, samplesize, 
                                    maxsnr = maxsnr, minband = minband,
                                    min_measures = min_measures,
                                    len_per_filter = 20,
                                    save_every = args.saveevery,
                                file_name = f"../goldstein_lsstLC/{subfolder}/train")


    goldstein_test_list = np.loadtxt('../goldstein/goldstein_test_42.csv', dtype=str, delimiter=',')
    test_list = np.array(["../goldstein/" + i for i in goldstein_test_list])
    zs_test = simulate_goldstein_lsst_data(train_list, samplesize//4, 
                                    maxsnr = maxsnr, minband = minband,
                                    min_measures = min_measures,
                                    len_per_filter = 20,
                                    save_every = args.saveevery,
                                file_name = f"../goldstein_lsstLC/{subfolder}/test")
    
    np.savez(f"../goldstein_lsstLC/{subfolder}/zs.npz", train = zs_train, test = zs_test)



if __name__ == '__main__': 
    main()
