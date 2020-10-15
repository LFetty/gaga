import os.path
from torch import nn
import torch
import numpy as np
import torch.utils.data as data
import os
import gatetools.phsp as phsp

NUMPY_EXTENSION = ['.npy']

def is_DICOM_image_file(filename):
    if isinstance(filename, (tuple, list)):
        for f in filename:
            assert any(f.endswith(extension) for extension in NUMPY_EXTENSION), 'numpy arrays can not be mixed with other data types. Please make sure only numpy files are used.'
        PASS = True
    else:
        PASS = any(filename.endswith(extension) for extension in NUMPY_EXTENSION)
    return PASS

class make_dataset(data.Dataset):
    def __init__(self, root, batchsize):
        self.path = root
        self.files = os.listdir(root)
        self.batchsize = batchsize

        self.file_size = len(self.files)

        self.mask = np.array([[]])

    def __getitem__(self, index):

        phase, keys, _ = phsp.load(os.path.join(self.path, self.files[index]), nmax=self.batchsize, shuffle=True)


        return phase[:,[0,1,2,4,5,6]]

    def __len__(self):
        return self.file_size

    def name(self):
        return 'phasespaceDataset'
