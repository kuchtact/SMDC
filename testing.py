import h5py
import imageio
from matplotlib import pyplot as plt
import numpy as np
import os
import paths

def main():
    file = h5py.File(paths._datapath)
    f = file['entry1']['data']
    dat = f['signal']
    maximum = np.max(np.nan_to_num(dat))
    for i in range(0, 501, 1):
        show(dat, z_slice=i, maximum=maximum)
    make_gif()


def show(data, z_slice=0, maximum=None):
    box_size = 501
    start = [0, 0]
    end = [start[0] + box_size, start[1] + box_size]

    data = data[start[0]:end[0], start[1]:end[1], z_slice]
    data -= np.min(data)
    data = np.float_power(data, 0.3)
    data = np.nan_to_num(data)
    if maximum is not None:
        data[0][0] = maximum

    plt.imshow(data)
    plt.title('slice: ' + str(z_slice), loc='left')
    slice_str = "%03d" % (z_slice)
    path = os.path.join(paths._projectpath, 'images/', 'slice_' + slice_str)
    print(path)
    plt.savefig(path)

def make_gif():
    images = []

    path = os.path.join(paths._projectpath, 'images')
    filenames = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(os.path.join(paths._projectpath, 'images/movie.gif'), images)


if __name__ == '__main__':
    main()
