import h5py
import numpy as np
from yaml import safe_load
from sys import getsizeof
import random

def give_random_interval(dim_shape = None, AROUND = None, lower = None, upper = None):
    # RANDOM_POS = np.random.randint(AROUND+10, dim_shape - AROUND-10-80)
    if ((upper+AROUND) >= (dim_shape - AROUND-10-80)):
        RANDOM_POS = (np.random.randint(10 + AROUND, lower - AROUND - 80))
    elif((10 + AROUND) >= (lower - AROUND - 80)):
        RANDOM_POS = (np.random.randint(upper+AROUND, dim_shape - AROUND-10-80))
    else:
        RANDOM_POS = random.choice([np.random.randint(10 + AROUND, lower - AROUND - 80),
                            np.random.randint(upper+AROUND, dim_shape - AROUND-10-80)])
    return slice(RANDOM_POS - AROUND, RANDOM_POS +AROUND +80)


def main():
    # load configurations
    print("[INFO] Loading vars")
    with open("./config.yml", 'r') as file:
        config_var = safe_load(file)["preparation"]
    data_path = config_var['input_path']
    print("[INFO] Preparing data")
    h5f = h5py.File(data_path)
    images = h5f["data"] # (1460,16,768,1152) numpy array
    boxes = h5f["boxes"] # (1460,15,5) numpy array

    # size of the box around the anomaly to be taken
    AROUND = config_var['around']
    if config_var["cyclon"]:
        # Takes a long time to create
        print("[INFO] Get cyclon train data")
        data_with_cyclon = [
                images[i, :, (cyclon[0] - AROUND):(cyclon[0] + 80 + AROUND),
                (cyclon[1] - AROUND):(cyclon[1] + 80 + AROUND)]
                for i, box in zip(range(images.shape[0]), boxes)
                for cyclon in box[(box[:, -1] == 2) | (box[:, -1] == 1)]
        ]
        print("RAM usage for array: %.4fMB" % (float(getsizeof(data_with_cyclon)) * 1e-6))
        # just remove the ones with smaller shape
        data_with_cyclon = [el for el in data_with_cyclon if el.shape == data_with_cyclon[0].shape]

        data_with_cyclon = np.stack(data_with_cyclon, axis = 0)
        print("[INFO] Preparing output file")
        hf = h5py.File(config_var['output_path']+'train_pos_2.h5', 'w')
        hf.create_dataset('positive', data = data_with_cyclon, compression = 'gzip')
        print("[INFO] Saving output file")
        hf.close()
    else:
        # Takes a long time to create
        print("[INFO] Get non-cyclon train data")
        # using the second loop i can get the same amount of data as the first one
        data_without_cyclon = [images[i, :,
                             give_random_interval(images.shape[2], AROUND = AROUND, lower = cyclon[0], upper = cyclon[2]),
                             give_random_interval(images.shape[3], AROUND = AROUND, lower = cyclon[1], upper = cyclon[3])]
                       for i, box in zip(range(images.shape[0]), boxes)
                       for cyclon in box[(box[:, -1] == 2)
                                         | (box[:, -1] == 1)]]
        data_without_cyclon = np.stack(data_without_cyclon, axis = 0)
        print("[INFO] Preparing output file")

        hf = h5py.File(config_var['output_path']+'train_neg_2.h5', 'w')
        hf.create_dataset('negative', data = data_without_cyclon, compression = 'gzip')
        print("[INFO] Saving output file")
        hf.close()

if __name__ == "__main__":
    main()
