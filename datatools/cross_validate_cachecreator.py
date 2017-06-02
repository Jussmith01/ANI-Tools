import numpy as np
import pyanitools as pyt
from pyNeuroChem import cachegenerator as cg

wkdir = '/home/jujuman/Research/DataReductionMethods/models/cv'

saef   = wkdir + "/sae_6-31gd.dat"

h5files = ['/home/jujuman/Research/DataReductionMethods/models/datasets/ani_red_cnl_c08f.h5',
           #'/home/jujuman/Research/DataReductionMethods/models/train_c08f/ani_red_c08f.h5',
           #'/home/jujuman/Research/ANI-DATASET/h5data/ani-begdb_h2o.h5',
           '/home/jujuman/Research/WaterData/ani-water_fix_1.h5',
           #wkdir + "/h5data/gdb9-2500-div_new.h5",
           #wkdir + "/h5data/ani-gdb-c08e.h5",
           ]

store_dir = wkdir + "/cache-c08e-"

#adl.split_load(10)
N = 10

train_idx = [[2, 3, 4, 5, 6, 7, 8, 9],
             [0, 1, 4, 5, 6, 7, 8, 9],
             [0, 1, 2, 3, 6, 7, 8, 9],
             [0, 1, 2, 3, 4, 5, 8, 9],
             [0, 1, 2, 3, 4, 5, 6, 7]
             ]

valid_idx = [[0,1]
            ,[2,3]
            ,[4,5]
            ,[6,7]
            ,[8,9]
             ]

cachet = [cg('_train', saef, store_dir + str(r) + '/',False) for r in range(5)]
cachev = [cg('_valid', saef, store_dir + str(r) + '/',False) for r in range(5)]

for fn in h5files:
    adl = pyt.anidataloader(fn)

    for c,data in enumerate(adl):
        # Print file
        print('Processing file: ', c)

        # Extract the data
        xyz = data['coordinates']
        erg = data['energies']
        spc = data['species']

        xyz = np.array_split(xyz, N)
        erg = np.array_split(erg, N)

        #spc = [a.decode('utf-8') for a in spc]
        #print(spc)

        for i,(t,v) in enumerate(zip(cachet, cachev)):
            xyz_t = np.array(np.concatenate([xyz[j] for j in train_idx[i]]), order='C', dtype=np.float32)
            erg_t = np.array(np.concatenate([erg[j] for j in train_idx[i]]), order='C', dtype=np.float64)

            xyz_v = np.array(np.concatenate([xyz[j] for j in valid_idx[i]]), order='C', dtype=np.float32)
            erg_v = np.array(np.concatenate([erg[j] for j in valid_idx[i]]), order='C', dtype=np.float64)

            print('shape: ', xyz_t.shape, ' V shape: ', xyz_v.shape)

            t.insertdata(xyz_t, erg_t, list(spc))
            v.insertdata(xyz_v, erg_v, list(spc))

    adl.cleanup()

for t,v in zip(cachet, cachev):
    t.makemetadata()
    v.makemetadata()

