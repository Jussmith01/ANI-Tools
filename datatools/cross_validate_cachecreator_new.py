import numpy as np
import pyanitools as pyt
from pyNeuroChem import cachegenerator as cg
import sys
import os

def interval(v,S):
    ps = 0.0
    ds = 1.0 / float(S)
    for s in range(S):
        if v > ps and v <= ps+ds:
            return s
        ps = ps + ds

wkdir = '/home/jujuman/Research/DataReductionMethods/model6r/model-gdb_r06_comb08_3/cv3/'
#wkdir ='/home/jujuman/Research/HIPNN-MD-Traj/ANI-Traj-test/cv1/'
saef   = wkdir + "sae_6-31gd.dat"
#saef   = wkdir + "sae_ccsd_cbs.dat"

h5files = ['/home/jujuman/Research/GDB_Dimer/dimer_gen_1/dimers1.h5',
           '/home/jujuman/Research/GDB_Dimer/dimer_gen_2/dimers2.h5',
           '/home/jujuman/Research/GDB_Dimer/dimer_gen_3/dimers3.h5',
           '/home/jujuman/Research/ReactionGeneration/reactiondata/DA_rxn_1/DA_rxn_1.h5',
           '/home/jujuman/Research/ReactionGeneration/reactiondata/comb_rxn_1/comb_rxn_1.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_3/gdb_r06_comb08_03_3.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_3/gdb_r06_comb08_03_2.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_3/gdb_r06_comb08_03_1.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_2/gdb_r06_comb08_02_4.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_2/gdb_r06_comb08_02_3.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_2/gdb_r06_comb08_02_2.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_2/gdb_r06_comb08_02_1.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_1/gdb_r06_comb08_5.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_1/gdb_r06_comb08_4.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_1/gdb_r06_comb08_3.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_1/gdb_r06_comb08_2.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_1/gdb_r06_comb08_1.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_mdal_resample/mdal.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/h2o_cluster/h2o_nms_clusters.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-05_red03-05/confs_cv_gdb01-05_rs1.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-05_red03-05/confs_cv_gdb01-05_rs2.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-05_red03-05/confs_cv_gdb01-05_rs3.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-05_red03-05/confs_cv_gdb01-05_rs4.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-06/confs_cv_gdb01-06_rs1.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-06/confs_cv_gdb01-06_rs2.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-06/confs_cv_gdb01-06_rs3.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-06/confs_cv_gdb01-06_rs4.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-07/confs_cv_gdb01-07_rs1.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-07/confs_cv_gdb01-07_rs2.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-07/confs_cv_gdb01-07_rs3.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-07/confs_cv_gdb01-07_rs4.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-08/confs_cv_gdb01-08_rs1.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-08/confs_cv_gdb01-08_rs2.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-08/confs_cv_gdb01-08_rs3.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-08/confs_cv_gdb01-08_rs4.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_h5/gdb11_S01_06r.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_h5/gdb11_S02_06r.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_h5/gdb11_S03_06r.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_h5/gdb11_S04_06r.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_h5/gdb11_S05_06r.h5',
           '/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_h5/gdb11_S06_06r.h5',
           #'/home/jujuman/Research/DataReductionMethods/model6/model0.05me/ani_red_c06.h5',
           #'/home/jujuman/Research/ANI-DATASET/h5data/r10_ccsd.h5',
           ]

store_dir = wkdir + "cache-data-"

N = 5

for i in range(N):
    if not os.path.exists(store_dir + str(i)):
        os.mkdir(store_dir + str(i))

    if os.path.exists(store_dir + str(i) + '/testset/testset.h5'):
        os.remove(store_dir + str(i) + '/testset/testset.h5')

    if not os.path.exists(store_dir + str(i) + '/testset'):
        os.mkdir(store_dir + str(i) + '/testset')

cachet = [cg('_train', saef, store_dir + str(r) + '/',False) for r in range(N)]
cachev = [cg('_valid', saef, store_dir + str(r) + '/',False) for r in range(N)]
testh5 = [pyt.datapacker(store_dir + str(r) + '/testset/testset.h5') for r in range(N)]

Nd = np.zeros(N,dtype=np.int32)
for f,fn in enumerate(h5files):
    print('Processing file('+ str(f+1) +' of '+ str(len(h5files)) +'):', fn)
    adl = pyt.anidataloader(fn)

    To = adl.size()
    for c, data in enumerate(adl):
        # Get test store name
        Pn = fn.split('/')[-1].split('.')[0] + data['path']

        # Progress indicator
        sys.stdout.write("\r%d%% %s" % (int(100*c/float(To)), Pn))
        sys.stdout.flush()

        # Extract the data
        X = data['coordinates']
        F = data['forces']
        E = data['energies']
        S = data['species']

        if (set(S).issubset(['C', 'N', 'O', 'H'])):

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
            for i,t,v,te in zip(range(N), cachet, cachev, testh5):
                ## Store training data
                X_t = np.array(np.concatenate([X[s] for j, s in enumerate(split) if j != i]), order='C', dtype=np.float32)
                E_t = np.array(np.concatenate([E[s] for j, s in enumerate(split) if j != i]), order='C', dtype=np.float64)
                if E_t.shape[0] != 0:
                    t.insertdata(X_t, E_t, list(S))

                ## Split test/valid data and store\
                tv_split = np.array_split(split[i],2)

                ## Store Validation
                if tv_split[0].size > 0:
                    X_v = np.array(X[tv_split[0]], order='C', dtype=np.float32)
                    E_v = np.array(E[tv_split[0]], order='C', dtype=np.float64)
                    if E_v.shape[0] != 0:
                        v.insertdata(X_v, E_v, list(S))

                ## Store testset
                if tv_split[1].size > 0:
                    X_te = np.array(X[tv_split[1]], order='C', dtype=np.float32)
                    F_te = np.array(F[tv_split[1]], order='C', dtype=np.float32)
                    E_te = np.array(E[tv_split[1]], order='C', dtype=np.float64)
                    if E_te.shape[0] != 0:
                        te.store_data(Pn, coordinates=X_te, forces=F_te, energies=E_te, species=list(S))

    sys.stdout.write("\r%d%%" % int(100))
    sys.stdout.flush()
    print("")

# Print some stats
print('Data count:',Nd)
print('Data split:',100.0*Nd/np.sum(Nd),'%')

# Save train and valid meta file and cleanup testh5
for t,v,th in zip(cachet, cachev, testh5):
    t.makemetadata()
    v.makemetadata()
    th.cleanup()
