import h5py
import paths

def load():
    file = h5py.File(paths._datapath)
    f = file['entry1']['data']
    dat = f['signal']
    return dat

if __name__ == '__main__':
    tom = load()