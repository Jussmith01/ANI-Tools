import numpy as np
import pyanitools as pyt
from pyNeuroChem import cachegenerator as cg
import sys
import os

import hdnntools as hdn

import matplotlib.pyplot as plt
import matplotlib as mpl


def interval(v,S):
    ps = 0.0
    ds = 1.0 / float(S)
    for s in range(S):
        if v > ps and v <= ps+ds:
            return s
        ps = ps + ds

#wkdir = '/home/jujuman/Scratch/Research/DataReductionMethods/model6r/model-gdb_r06_comb08_2/cv3/'
wkdir ='/home/jujuman/Research/ForceTrainTesting/train/'
saef   = wkdir + "sae_6-31gd.dat"
bddir = '/home/jujuman/Research/ForceTrainTesting/train/bad_mols/'
#saef   = wkdir + "sae_ccsd_cbs.dat"
P = 1.0

h5files = [#(P, '/home/jujuman/Research/GDB_Dimer/dimer_gen_1/dimers1.h5'),
           #(P, '/home/jujuman/Research/GDB_Dimer/dimer_gen_2/dimers2.h5'),
           #(0.4, '/home/jujuman/Research/ReactionGeneration/reactiondata/DA_rxn_1/DA_rxn_1.h5'),
           #(P, '/home/jujuman/Research/ReactionGeneration/reactiondata/comb_rxn_1/comb_rxn_1.h5'),
           #(P, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_2/gdb_r06_comb08_02_3.h5'),
           #(P, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_2/gdb_r06_comb08_02_2.h5'),
           #(P, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_2/gdb_r06_comb08_02_1.h5'),
           #(P, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_1/gdb_r06_comb08_5.h5'),
           #(P, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_1/gdb_r06_comb08_4.h5'),
           #(P, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_1/gdb_r06_comb08_3.h5'),
           #(P, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_1/gdb_r06_comb08_2.h5'),
           #(P, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_1/gdb_r06_comb08_1.h5'),
           #(P, '/home/jujuman/Scratch/Research/GDB-11-AL-wB97x631gd/dnnts_mdal_resample/mdal.h5'),
           #(P, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/h2o_cluster/h2o_nms_clusters.h5'),
           #(P, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-05_red03-05/confs_cv_gdb01-05_rs1.h5'),
           #(P, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-05_red03-05/confs_cv_gdb01-05_rs2.h5'),
           #(P, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-05_red03-05/confs_cv_gdb01-05_rs3.h5'),
           #(P, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-05_red03-05/confs_cv_gdb01-05_rs4.h5'),
           #(P, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-06/confs_cv_gdb01-06_rs1.h5'),
           #(P, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-06/confs_cv_gdb01-06_rs2.h5'),
           #(P, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-06/confs_cv_gdb01-06_rs3.h5'),
           #(P, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-06/confs_cv_gdb01-06_rs4.h5'),
           #(P, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-07/confs_cv_gdb01-07_rs1.h5'),
           #(P, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-07/confs_cv_gdb01-07_rs2.h5'),
           #(P, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-07/confs_cv_gdb01-07_rs3.h5'),
           #(P, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-07/confs_cv_gdb01-07_rs4.h5'),
           #(P, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-08/confs_cv_gdb01-08_rs1.h5'),
           #(P, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-08/confs_cv_gdb01-08_rs2.h5'),
           #(P, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-08/confs_cv_gdb01-08_rs3.h5'),
           #(P, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-08/confs_cv_gdb01-08_rs4.h5'),
           (P, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_h5/gdb11_S01_06r.h5'),
           #(P, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_h5/gdb11_S02_06r.h5'),
           #(P, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_h5/gdb11_S03_06r.h5'),
           #(P, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_h5/gdb11_S04_06r.h5'),
           #(P, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_h5/gdb11_S05_06r.h5'),
           #(P, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_h5/gdb11_S06_06r.h5'),
           ]


#h5files = [(P, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_h5/gdb11_S01_06r.h5'),
#           ]


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
Nbf = 0
for f,fn in enumerate(h5files):
    print('Processing file('+ str(f+1) +' of '+ str(len(h5files)) +'):', fn[1])
    adl = pyt.anidataloader(fn[1])

    To = adl.size()
    Ndc = 0
    Fmt = []
    Emt = []
    for c, data in enumerate(adl):
        #if c == 2 or c == 2 or c == 2:
        # Get test store name
        Pn = fn[1].split('/')[-1].split('.')[0] + data['path']

        # Progress indicator
        sys.stdout.write("\r%d%% %s" % (int(100*c/float(To)), Pn))
        sys.stdout.flush()

        # Extract the data
        X = data['coordinates']
        F = data['forces']
        E = data['energies']
        S = data['species']

        Fmt.append(np.max(np.linalg.norm(F,axis=2),axis=1))
        Emt.append(E)
        Mv = np.max(np.linalg.norm(F,axis=2),axis=1)
        #print(Mv.shape,X.shape)
        index = np.where(Mv > 1.0)[0]
        indexk = np.where(Mv <= 1.0)[0]
        #if index.size > 0:
            #print(Mv[index])
            #hdn.writexyzfile(bddir+'mols_'+str(c).zfill(3)+'_'+str(f).zfill(3)+'.xyz',X[index],S)
        Nbf += index.size

        # CLear forces
        X = X[indexk]
        F = F[indexk]
        E = E[indexk]

        #exit(0)
        #print(" MAX FORCE:", F.max(), S)
        '''
        print('meanforce:',F.flatten().mean())
        print("FORCE:",F)
        print(np.max(F.reshape(E.size,F.shape[1]*F.shape[2]),axis=1))
        print("MAX FORCE:", F.max(),S)

        if F.max() > 0.0:
            print(np.mean(F.reshape(E.size,F.shape[1]*F.shape[2]),axis=1).shape, E.size)
            plt.hist(np.max(np.abs(F).reshape(E.size,F.shape[1]*F.shape[2]),axis=1),bins=100)
            plt.show()
            plt.scatter(np.max(np.abs(F).reshape(E.size,F.shape[1]*F.shape[2]),axis=1), E)
            plt.show()
        '''
        Ru = np.random.uniform(0.0, 1.0, E.shape[0])
        nidx = np.where(Ru < fn[0])
        X = X[nidx]
        F = F[nidx]
        E = E[nidx]

        Ndc += E.size
        #for i in range(E.size):
        #    X[i] = X[0]
        #    F[i] = F[0]
        #    E[i] = E[0]

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
                F_t = np.array(np.concatenate([F[s] for j, s in enumerate(split) if j != i]), order='C', dtype=np.float32)
                E_t = np.array(np.concatenate([E[s] for j, s in enumerate(split) if j != i]), order='C', dtype=np.float64)

                if E_t.shape[0] != 0:
                    t.insertdata(X_t, F_t, E_t, list(S))

                ## Split test/valid data and store\
                tv_split = np.array_split(split[i],2)

                ## Store Validation
                if tv_split[0].size > 0:
                    X_v = np.array(X[tv_split[0]], order='C', dtype=np.float32)
                    F_v = np.array(F[tv_split[0]], order='C', dtype=np.float32)
                    E_v = np.array(E[tv_split[0]], order='C', dtype=np.float64)
                    if E_v.shape[0] != 0:
                        v.insertdata(X_v, F_v, E_v, list(S))

                ## Store testset
                if tv_split[1].size > 0:
                    X_te = np.array(X[tv_split[1]], order='C', dtype=np.float32)
                    F_te = np.array(F[tv_split[1]], order='C', dtype=np.float32)
                    E_te = np.array(E[tv_split[1]], order='C', dtype=np.float64)
                    if E_te.shape[0] != 0:
                        te.store_data(Pn, coordinates=X_te, forces=F_te, energies=E_te, species=list(S))

    #plt.hist(np.concatenate(Fmt), bins=150)
    #plt.show()

    #plt.hist(Emt, bins=150)
    #plt.show()

    sys.stdout.write("\r%d%%" % int(100))
    print(" Data Kept: ", Ndc, 'High Force: ', Nbf)
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
