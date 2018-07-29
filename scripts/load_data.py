import h5py
import paths

def load_dataset():
    file = h5py.File(paths._datapath)
    f = file['entry1']['data']
    dat = f['signal']
    return dat

def load_array():
    return load_dataset()[()]
