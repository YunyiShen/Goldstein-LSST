import numpy as np
import os
import numpy as np
from sklearn.model_selection import train_test_split

def split_goldstein(path, test = 0.2, random_state = 42):
    # Step 1: List all files of a certain type
    all_files = [f for f in os.listdir(path) if f.endswith(".h5")]
    all_files_np = np.array(all_files)

    # Step 2: Randomly split (e.g., 80% train, 20% test)
    train_files, test_files = train_test_split(all_files_np, test_size=test, random_state=42)

    # Step 3: Save to CSV
    np.savetxt(f'{path}/goldstein_train_{random_state}.csv', train_files, fmt='%s', delimiter=',')
    np.savetxt(f'{path}/goldstein_test_{random_state}.csv', test_files, fmt='%s', delimiter=',')

