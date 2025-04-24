from LSSTsimu.simulations import simulate_goldstein_lsst_data
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM

def main():
    goldstein_train_list = np.loadtxt('../goldstein/goldstein_train_42.csv', dtype=str, delimiter=',')
    train_list = np.array(["../goldstein/" + i for i in goldstein_train_list])
    zs_train = simulate_goldstein_lsst_data(train_list, 20000, 
                                    maxsnr = 15, minband = 2,
                                    min_measures = 10,
                                    len_per_filter = 20,
                                file_name = "../goldstein_lsstLC/goldstein_20k_train_maxsnr15_minband2_minmeasures10.jsonl")


    goldstein_test_list = np.loadtxt('../goldstein/goldstein_test_42.csv', dtype=str, delimiter=',')
    test_list = np.array(["../goldstein/" + i for i in goldstein_test_list])
    zs_test = simulate_goldstein_lsst_data(train_list, 5000, 
                                    maxsnr = 15, minband = 2,
                                    min_measures = 10,
                                    len_per_filter = 20,
                                file_name = "../goldstein_lsstLC/goldstein_20k_test_maxsnr15_minband2_minmeasures10.jsonl")
    
    np.savez("zs", train = zs_train, test = zs_test)



if __name__ == '__main__': 
    main()