from matplotlib import pyplot as plt
from scripts import load_data
import numpy as np

def create_hist(array):
    array = array.flatten()
    ax = plt.subplot(111)
    n, bins, patches = ax.hist(array, 100)
    ax.set_yscale("log")
    plt.show()

if __name__ == '__main__':
    arr = load_data.load_dataset()[()]
    arr = np.nan_to_num(arr)
    create_hist(arr)
