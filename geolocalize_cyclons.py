import h5py
import numpy as np
from yaml import safe_load
from sys import getsizeof
import random
from warnings import warn
from progressbar import progressbar

SKIPPED = 0
#
#
#
#
def get_box_lat(around=None,
                diff_lats=0.23468057366362416,
                lat_cyclon_min=None,
                lat_cyclon_max=None):
    if diff_lats < 0:
        warn("Negative step given. Changing sign to have positive value.")
        diff_lats = abs(diff_lats)

    ymin_out = (lat_cyclon_min - around * diff_lats)
    ymax_out = (lat_cyclon_max + around * diff_lats)
    #     check if box goes around the edges
    if ymin_out < -90:
        ymin_out = -180 - (ymin_out)
    if ymax_out > 90:
        ymax_out = 180 - ymax_out

    return (ymin_out, ymax_out)


#
#
#
#
#
#
def get_box_lon(around=None,
                diff_lons=0.31277150304083406,
                lon_cyclon_min=None,
                lon_cyclon_max=None):
    if diff_lons < 0:
        warn("Negative step given. Changing sign to have positive value.")
        diff_lons = abs(diff_lons)

    ymin_out = (lon_cyclon_min - around * diff_lons)
    ymax_out = (lon_cyclon_max + around * diff_lons)
    #     check if box goes around the edges
    if ymin_out < 0:
        ymin_out = 360 + ymin_out
    if ymax_out > 360:
        ymax_out = ymax_out - 360

    return (ymin_out, ymax_out)


#
#
#
#
#
def select_box_cyclon(image=None,
                      time=None,
                      lat_cyclon_min=None,
                      lat_cyclon_max=None,
                      lon_cyclon_min=None,
                      lon_cyclon_max=None,
                      lats=None,
                      lons=None,
                      around=10):
    assert around > 0, "Box_size = "+str(around)+". Please give the box size as positive. You may need to increae the box size for the whole data."
    lat_cyclon_min, lat_cyclon_max = get_box_lat(lat_cyclon_min=lat_cyclon_min,
                                                 lat_cyclon_max=lat_cyclon_max,
                                                 around=around)
    lon_cyclon_min, lon_cyclon_max = get_box_lon(lon_cyclon_min=lon_cyclon_min,
                                                 lon_cyclon_max=lon_cyclon_max,
                                                 around=around)
    if lon_cyclon_min > lon_cyclon_max:
        return {
            'lat_min':
            lat_cyclon_min,
            'lat_max':
            lat_cyclon_max,
            'lon_min':
            lon_cyclon_min,
            'lon_max':
            lon_cyclon_max,
            'time':
            time,
            'images':
            np.append(
                image[:,
                      slice(
                          np.
                          where(np.isclose(lats, lat_cyclon_min, 1e-6))[0][0],
                          np.
                          where(np.isclose(lats, lat_cyclon_max, 1e-6))[0][0]),
                      slice(
                          0,
                          np.where(
                              np.isclose(lons, lon_cyclon_max, 1e-6))[0][0])],
                image[:,
                      slice(
                          np.
                          where(np.isclose(lats, lat_cyclon_min, 1e-6))[0][0],
                          np.
                          where(np.isclose(lats, lat_cyclon_max, 1e-6))[0][0]),
                      slice(
                          np.
                          where(np.isclose(lons, lon_cyclon_min, 1e-6))[0][0]
                                , -1)], axis = 2)
        }
    else:
        return {
            'lat_min':
            lat_cyclon_min,
            'lat_max':
            lat_cyclon_max,
            'lon_min':
            lon_cyclon_min,
            'lon_max':
            lon_cyclon_max,
            'time':
            time,
            'images':
            image[:,
                  slice(
                      np.where(np.isclose(lats, lat_cyclon_min, 1e-6))[0][0],
                      np.where(np.isclose(lats, lat_cyclon_max, 1e-6))[0][0]),
                  slice(
                      np.where(np.isclose(lons, lon_cyclon_min, 1e-6))[0][0],
                      np.where(np.isclose(lons, lon_cyclon_max, 1e-6))[0][0])]
        }



def do_selection(image = None, lats = None, lons = None, cyclon = None, box_size = 120, i = None, times = None, verbose = 1):
    global SKIPPED
    try:
        return (select_box_cyclon(image=image[:, :, :],
                          time=times[i],
                          lat_cyclon_min=lats[cyclon[0]],
                          lat_cyclon_max=lats[cyclon[2]],
                          lon_cyclon_min=lons[cyclon[1]],
                          around=((box_size - abs(cyclon[0] - cyclon[2])) / 2),
                          lon_cyclon_max=lons[cyclon[3]],
                          lats=lats[()],
                          lons=lons[()]))
    except Exception as e:
        SKIPPED += 1
        if verbose == 1:
            print("\n", e)
            warn("For some reason the selection at step "+str(i)+" did not work. May be due to error in data (some latitudes have indexes above 1152)")
        return None
#
#
#
def main():
    global SKIPPED
    # load configurations
    print("[INFO] Loading vars")
    with open("./config.yml", 'r') as file:
        config_var = safe_load(file)["geo_localize"]
    data_path = config_var['input_path']
    print("[INFO] Preparing data")
    h5f = h5py.File(data_path)
    images = h5f["data"][()] # (1460,16,768,1152) numpy array
    boxes = h5f["boxes"] # (1460,15,5) numpy array
    lats = h5f['latitude']
    lons = h5f['longitude']
    times = h5f['time']
    # size of the box around the anomaly to be taken
    box_size = config_var['box_size']
    HOW_MANY = config_var['iterations']

    if HOW_MANY == "All":
        HOW_MANY = images.shape[0]

    assert type(HOW_MANY) is int, "Plase give the number of interations either as an integer or as 'All'."
    print('[INFO] Numbero of iterations %i' %HOW_MANY)

    print("[INFO] Selecting data with cyclons")
    data_with_cyclon = [
        do_selection(image=images[i],
                 lats=lats,
                 lons=lons,
                 cyclon=cyclon,
                 box_size=box_size,
                 times = times,
                 verbose = int(config_var['verbose_selection']),
                 i=i) for i in progressbar((range(HOW_MANY)))
                 for cyclon in boxes[i][(boxes[i, :, -1] == 2) | (boxes[i, :, -1] == 1)]
    ]
    print('[INFO] Skipped %i cyclons' %SKIPPED)
    # eliminate the Nones and keep only images with the correct size. This should take too long
    data_with_cyclon = [el for el in data_with_cyclon if ((el is not None) and (el['images'].shape == (images.shape[1],box_size, box_size)))]
    print("[INFO] Preparing output file")
    hf = h5py.File(config_var['output_path'], 'w')
    for key in data_with_cyclon[0].keys():
        hf.create_dataset(key,
                      data=np.stack([el[key] for el in data_with_cyclon],
                                    axis=0),
                      compression='gzip')
    print("[INFO] Saving output file")
    hf.close()


if __name__ == "__main__":
    main()
