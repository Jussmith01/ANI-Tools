import numpy as np
import os
# Store test split
import pyanitools as pyt
from pyNeuroChem import cachegenerator as cg
import hdnntools as hdn

import matplotlib.pyplot as plt

# Set the HDF5 file containing the data
hdf5files = ['/home/jujuman/Research/HIPNN-MD-Traj/ANI-Traj-test/cv1/data_benzene_md.h5',
             #'/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnnts_red/ani_correct.h5',
             #'/home/jujuman/Research/DataReductionMethods/model6/model0.05me/ani_red_c06.h5',
             #'/home/jujuman/Research/ANI-DATASET/ANI-1_release/ani_gdb_s01.h5',
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
storecac = '/home/jujuman/Research/HIPNN-MD-Traj/ANI-Traj-test/cv1/cache/'
saef   = "/home/jujuman/Research/HIPNN-MD-Traj/ANI-Traj-test/cv1/sae_6-31gd.dat"
path = "/home/jujuman/Research/HIPNN-MD-Traj/ANI-Traj-test/cv1/cache/testset/testset.h5"

Ts = 0.002
Vs = 0.002

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
        xyz = data['coordinates']
        frc = data['forces']
        eng = data['energies']
        spc = data['species']

        #eng = eng - eng.mean()

        ds_path = data['path']

        Ndat = eng.size
        idx = np.arange(0,Ndat)
        np.random.shuffle(idx)

        print('Training Size:   ',Ts*Ndat)
        print('Validation Size: ',Vs*Ndat)

        print(eng)
        # Prepare and store the training and validation data
        cachet.insertdata(xyz[idx[0:int(Ts*Ndat)]], frc[idx[0:int(Ts*Ndat)]], eng[idx[0:int(Ts*Ndat)]], list(spc))
        cachev.insertdata(xyz[idx[int(Ts*Ndat):int((Ts+Vs)*Ndat)]], frc[idx[int(Ts*Ndat):int((Ts+Vs)*Ndat)]], eng[idx[int(Ts*Ndat):int((Ts+Vs)*Ndat)]], list(spc))

        # Prepare and store the test data set
        #if xyz[9].shape[0] != 0:
            #print(xyz[9].shape)
        #t_xyz = xyz[9].reshape(xyz[9].shape[0],xyz[9].shape[1]*xyz[9].shape[2])
        dpack.store_data(ds_path, coordinates=xyz[int((Ts+Vs)*Ndat):], forces=frc[int((Ts+Vs)*Ndat):], energies=eng[int((Ts+Vs)*Ndat):], species=spc)
    print('Count: ',dc)

    adl.cleanup()

# Make meta data file for caches
cachet.makemetadata()
cachev.makemetadata()

# Cleanup the disk
dpack.cleanup()
