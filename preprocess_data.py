import h5py
from yaml import safe_load
import numpy as np
from datetime import datetime, timedelta

def main():
    # load configurations
    print("[INFO] Loading vars")
    with open("./config.yml", 'r') as file:
        config_var = safe_load(file)["preprocess"]
    data_path = config_var['input_path']
    print("[INFO] Loading data")
    input_hf = h5py.File(data_path)
    images = input_hf["images"][:,config_var['vars_to_use'], :, :] # (1456,16,768,1152) numpy array
    boxes = input_hf["boxes"][()] # (1456,15,5) numpy array
    input_hf.close()
    print("[INFO] Preparing latitudes, longitudes and times")
    LAT = np.append(np.arange(-90, 90, 180/(images.shape[-2]-1)), 90)
    LON = np.append(np.arange(0,360, 360/(images.shape[-1]-1)), 360)
    print("[INFO] Saving time object with format '<i8'. Might want to change to '<M8[us]'")
    TIME = np.arange(datetime(2005,1,1), datetime(2005,12,31), timedelta(hours = 6)).astype('<i8')
    print("[INFO] Saving data. May take a few minutes")
    output_hf = h5py.File(config_var['output'], 'w')
    output_hf.create_dataset('data', data=images, compression='gzip')
    output_hf.create_dataset('boxes', data=boxes, compression='gzip')
    output_hf.create_dataset('latitude', data=LAT, compression='gzip')
    output_hf.create_dataset('longitude', data=LON, compression='gzip')
    output_hf.create_dataset('time', data=TIME, compression='gzip')
    output_hf.close()
    print('[INFO] Data successfully save. Hurray!')
if __name__ == "__main__":
    main()
