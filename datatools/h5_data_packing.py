import numpy as np
import hdnntools as gt
from os import listdir
import os
import pandas as pd
import itertools
import time as tm
import pyanitools as pyt

#path = "/home/jujuman/Research/ANI-DATASET/rxn_db_mig.h5"
#path = "/home/jujuman/Research/ANI-DATASET/ani_data_c01test.h5"
#path = "/home/jujuman/Research/WaterData/ani-water_fix_1.h5"
#path = '/home/jujuman/Research/ReactionGeneration/DataGen/ani-DA_rxn.h5'
path = '/home/jujuman/Research/ANI-DATASET/h5data/ani-peptide_org.h5'
#path = "/home/jujuman/Research/SingleNetworkTest/datah5/ani-homo_water.h5"

dtdirs = [#"/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnnts_rxns/scans_double_bond_migration/data/",
          "/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnnts_dipeptides/testdata2/",
          "/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnnts_dipeptides/testdata/",
          #"/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnnts_begdb/testdata/",
          #"/home/jujuman/Research/ReactionGeneration/DataGen/finished1/data/",
          #"/home/jujuman/Research/WaterData/cv_md_1/data/",
          "/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnnts_aminoacids/testdata/",
          #"/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnnts_fixdata/data/",
          #"/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnnts_dissociation/scans_cc_bonds_dft/double/data/",
          #"/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnnts_dissociation/scans_cc_bonds_dft/single/data/",
          #"/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnnts_rxns/benz_dbm_rxns/scans_dbm_benz_4/data/",
          #"/home/jujuman/Research/ANI-DATASET/h5data",
          #"/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnntsgdb11_01/testdata/",
          #"/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnntsgdb11_02/testdata/",
          #"/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnntsgdb11_03/testdata/",
          #"/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnntsgdb11_04/testdata/",
          #"/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnntsgdb11_05/testdata/",
          #"/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnntsgdb11_06/testdata/",
          #"/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnntsgdb11_07/testdata/",
          #"/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnntsgdb11_08/testdata/",
          #"/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnntsgdb11_10/testdata/",
          #"/home/jujuman/Research/SingleNetworkTest/testdata/",
         ]

#namelist = ["_train.dat", "_valid.dat", "_test.dat"]
namelist = ["_train.dat", ]
#namelist = [".dat",]

if os.path.exists(path):
    os.remove(path)

#open an HDF5 for compressed storage.
#Note that if the path exists, it will open whatever is there.
#store = pd.HDFStore(path,complib='blosc',complevel=8)
dpack = pyt.datapacker(path)

totaltime = 0.0
Ns = 0
for d in dtdirs:

    tmp = listdir(d)
    print(d)
    files = list({("_".join(f.split("_")[:-1])) for f in tmp})
    files = sorted(files, key=lambda x: int(x.split('-')[1].split('.')[0]))
    gn = files[0].split("-")[0]

    fcounter = 0
    for n,i in enumerate(files):
        allarr = []

        print(d+i)

        nc = 0
        for k in namelist:
            if os.path.exists(d+i+k):
                try:
                    _timeloop = tm.time()
                    readarrays = gt.readncdat(d+i+k)
                    _timeloop2 = (tm.time() - _timeloop)
                    totaltime += _timeloop2

                    shapesarr = [x.shape for x in readarrays]
                    typ = readarrays[1]
                except FileNotFoundError:
                    readarrays = [np.zeros((0,x[1:])) for x in shapesarr]

                ncsub, nat, ndim = readarrays[0].shape
                nc += ncsub
                readarrays[0] = readarrays[0].reshape(ncsub*nat, ndim)


                allarr.append(readarrays)

        xyz, typ, E = zip(*allarr)

        # Prepare coordinate arrays
        xyz = np.concatenate(xyz).reshape((nc,nat,3))
        xyz = np.array(xyz, dtype=np.float32)

        # Prepare energy arrays
        E = np.concatenate(E).reshape(nc)

        Ns += nc

        # Prepare and store the data
        spc = typ[0]
        #spc = [a.encode('utf8') for a in typ[0]]
        if xyz.shape[0] != 0:
            #print(gn)
            dpack.store_data(gn + "/mol" + str(n), coordinates=xyz, energies=E, species=list(spc))

        fcounter = fcounter + 1

    print('Total load function time: ' + "{:.4f}".format(totaltime) + 's Mols:',fcounter)

print("Total Structures: ", Ns)

dpack.cleanup()

