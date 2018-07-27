import h5py
import imageio
from matplotlib import pyplot as plt
import numpy as np
import os
import paths

def main():
    f = h5py.File(paths._datapath)
    f = f['entry1']['data']
    dat = f['signal']
    for i in range(0, 20, 2):
        show(dat, z_slice=i)


def show(data, z_slice=0):
    box_size = 501
    start = [0, 0]
    end = [start[0] + box_size, start[1] + box_size]

    data = data[start[0]:end[0], start[1]:end[1], z_slice]
    data -= np.min(data)
    data = np.float_power(data, 0.3)
    data = np.nan_to_num(data)

    plt.imshow(data)
    plt.savefig('/Users/cameronkuchta/PycharmProjects/SMDC/images/slice_' + str(z_slice))

def make_gif():
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('/path/to/movie.gif', images)


if __name__ == '__main__':
    main()
