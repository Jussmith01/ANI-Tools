import numpy as np
import pyanitools as pyt
from pyNeuroChem import cachegenerator as cg
import sys

def interval(v,S):
    ps = 0.0
    ds = 1.0 / float(S)
    for s in range(S):
        if v > ps and v <= ps+ds:
            return s
        ps = ps + ds

wkdir = '/home/jujuman/Research/DataReductionMethods/models/cv'

saef   = wkdir + "/sae_6-31gd.dat"

h5files = ['/home/jujuman/Research/WaterData/ani-water_fix_1.h5',
           '/home/jujuman/Research/ReactionGeneration/DataGen/ani-DA_rxn.h5',
           '/home/jujuman/Research/DataReductionMethods/models/datasets/ani_red_cnl_c08f.h5',
           #'/home/jujuman/Research/DataReductionMethods/models/train_c08f/ani_red_c08f.h5',
           #'/home/jujuman/Research/ANI-DATASET/h5data/ani-begdb_h2o.h5',
           #wkdir + "/h5data/gdb9-2500-div_new.h5",
           #wkdir + "/h5data/ani-gdb-c08e.h5",
           ]

store_dir = wkdir + "/cache-data-"

N = 5

cachet = [cg('_train', saef, store_dir + str(r) + '/',False) for r in range(N)]
cachev = [cg('_valid', saef, store_dir + str(r) + '/',False) for r in range(N)]

Nd = np.zeros(N,dtype=np.int32)
for f,fn in enumerate(h5files):
    print('Processing file('+ str(f+1) +' of '+ str(len(h5files)) +'):', fn)
    adl = pyt.anidataloader(fn)

    To = adl.size()
    for c, data in enumerate(adl):
        # Progress indicator
        sys.stdout.write("\r%d%%" % int(100*c/float(To)))
        sys.stdout.flush()

        # Extract the data
        X = data['coordinates']
        E = data['energies']
        S = data['species']

        # Random mask
        R = np.random.uniform(0.0, 1.0, E.shape[0])
        idx = np.array([interval(r,N) for r in R])

        # Build random split lists
        split = []
        for j in range(N):
            split.append([i for i, s in enumerate(idx) if s == j])
            nd = len([i for i, s in enumerate(idx) if s == j])
            Nd[j] = Nd[j] + nd

        # Store data
        for i,t,v in zip(range(N), cachet, cachev):
            X_t = np.array(np.concatenate([X[s] for j, s in enumerate(split) if j != i]), order='C', dtype=np.float32)
            E_t = np.array(np.concatenate([E[s] for j, s in enumerate(split) if j != i]), order='C', dtype=np.float64)

            X_v = np.array(X[split[i]], order='C', dtype=np.float32)
            E_v = np.array(E[split[i]], order='C', dtype=np.float64)

            if E_t.shape[0] != 0:
                t.insertdata(X_t, E_t, list(S))

            if E_v.shape[0] != 0:
                v.insertdata(X_v, E_v, list(S))
    sys.stdout.write("\r%d%%" % int(100 * To / float(To)))
    sys.stdout.flush()
    print("")

print('Data split:',100.0*Nd/np.sum(Nd),'%')
# Save Meta File
for t,v in zip(cachet, cachev):
    t.makemetadata()
    v.makemetadata()