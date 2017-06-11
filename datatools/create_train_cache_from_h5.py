import numpy as np
import os
# Store test split
import pyanitools as pyt
from pyNeuroChem import cachegenerator as cg

# Set the HDF5 file containing the data
hdf5files = ['/home/jujuman/Research/ANI-DATASET/ANI-1_release/ani_gdb_s01.h5',
             #'/home/jujuman/Research/ANI-DATASET/ANI-1_release/ani_gdb_s02.h5',
             #'/home/jujuman/Research/ANI-DATASET/ANI-1_release/ani_gdb_s03.h5',
             #'/home/jujuman/Research/ANI-DATASET/ANI-1_release/ani_gdb_s04.h5',
             #'/home/jujuman/Research/ANI-DATASET/ANI-1_release/ani_gdb_s05.h5',
             #'/home/jujuman/Research/ANI-DATASET/ANI-1_release/ani_gdb_s06.h5',
             #'/home/jujuman/Research/ANI-DATASET/h5data/ani-gdb-c01.h5',
             #'/home/jujuman/Research/ANI-DATASET/h5data/ani-gdb-c02.h5',
             #'/home/jujuman/Research/ANI-DATASET/h5data/ani-gdb-c03.h5',
             #'/home/jujuman/Research/ANI-DATASET/h5data/ani-gdb-c04.h5',
             #'/home/jujuman/Research/ANI-DATASET/h5data/ani-gdb-c05.h5',
             #'/home/jujuman/Research/ANI-DATASET/h5data/ani-gdb-c06.h5',
             #'/home/jujuman/Research/ANI-DATASET/h5data/ani-gdb-c08f.h5',
             #'/home/jujuman/Research/ANI-DATASET/h5data/ani-begdb_h2o.h5',
             #'/home/jujuman/Research/SingleNetworkTest/datah5/ani-homo_water.h5',
             ]

#hdf5file = '/home/jujuman/Research/ANI-DATASET/ani-1_data_c03.h5'
storecac = '/home/jujuman/Research/DataReductionMethods/model6/cache/'
saef   = "/home/jujuman/Research/DataReductionMethods/model6/train/sae_6-31gd.dat"
path = "/home/jujuman/Research/DataReductionMethods/model6/cache/testset/testset.h5"

if os.path.exists(path):
    os.remove(path)

# Declare data cache
cachet = cg('_train', saef, storecac, False)
cachev = cg('_valid', saef, storecac, False)

# Declare test cache
dpack = pyt.datapacker(path)

for f in hdf5files:
    # Construct the data loader class
    print(f)
    adl = pyt.anidataloader(f)

    print(adl.get_group_list())

    # Loop over data in set
    dc = 0
    for i,data in enumerate(adl):
        #if (i == 2):
        xyz = np.array_split(data['coordinates'], 10)
        eng = np.array_split(data['energies'], 10)
        spc = data['species']
        nme = data['parent']

        #print('Parent: ', nme, eng)
        dc = dc + np.concatenate(eng[0:8]).shape[0]

        # Prepare and store the training and validation data
        cachet.insertdata(np.concatenate(xyz[0:8]), np.array(np.concatenate(eng[0:8]), dtype=np.float64), list(spc))
        cachev.insertdata(xyz[8], np.array(eng[8], dtype=np.float64), list(spc))

        # Prepare and store the test data set
        if xyz[9].shape[0] != 0:
            #print(xyz[9].shape)
            t_xyz = xyz[9].reshape(xyz[9].shape[0],xyz[9].shape[1]*xyz[9].shape[2])
            dpack.store_data(nme + '/mol' + str(i), coordinates=t_xyz, energies=np.array(eng[9]), species=spc)
    print('Count: ',dc)

    adl.cleanup()

# Make meta data file for caches
cachet.makemetadata()
cachev.makemetadata()

# Cleanup the disk
dpack.cleanup()