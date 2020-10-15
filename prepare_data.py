import numpy as np
import glob
import os

path = 'data\\train_conditional'


for file in os.listdir(path):

    array = np.load(os.path.join(path, file))

    energy = float(file.split('_')[1][:-2])

    if energy == 60:
        energy = 0
    elif energy == 80:
        energy = 1
    elif energy == 100:
        energy = 2
    elif energy == 120:
        energy = 3

    keys = ['Ekine', 'X', 'Y', 'dX', 'dY', 'dZ', 'label']

    res = array.shape



    dtype = []
    for k in keys:
        dtype.append((k, 'f4'))

    r = np.zeros(len(array), dtype=dtype)
    i = 0
    for k in keys:

        if k == 'label':
            r[k] = np.ones(res)*energy
        else:
            r[k] = array[k]

        i+=1

        #r[i] = temp


    np.save(path+'\\'+file[:-4]+'_label', r)


    #np.save(f'data\\train\\{i}.npy', array[i:i+1])