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

#wkdir = '/scratch/Research/force_train_testing/'
#saef   = wkdir + "sae_6-31gd.dat"

wkdir = '/nh/nest/u/jsmith/Research/gutzwiller_research/train_all/gutz_model-5/'
saef   = wkdir + "sae.dat"

#wkdir = '/scratch/Research/datasets/iso17/train_test/'
#saef   = wkdir + "sae_6-31gd.dat"

#data_root = '/scratch/Research/GDB-11-AL-wB97x631gd/'
data_root = '/auto/nest/nest/u/jsmith/scratch/Research/gutzwiller_research/h5files/'
#data_root = '/scratch/Research/datasets/iso17/'

h5files = [#'/home/jujuman/Research/Cluster_AL/waterclusters1.h5',
           #data_root + 'gutzwiller1-U2-rs1.5.h5',
           #data_root + 'gutzwiller1-U4-rs1.5.h5',
           #data_root + 'gutzwiller1-U6-rs1.5.h5',
           #data_root + 'gutzwiller1-U8-rs1.5.h5',
           #data_root + 'gutzwiller1-U10-rs1.5.h5',
           data_root + 'gutzwiller1-U12-rs1.5.h5',
           ]

store_dir = wkdir + "cache-data-"

N = 5

for i in range(N):
    if not os.path.exists(store_dir + str(i)):
        os.mkdir(store_dir + str(i))

if os.path.exists(wkdir + 'testset.h5'):
    os.remove(wkdir + 'testset.h5')

cachet = [cg('_train', saef, store_dir + str(r) + '/',False) for r in range(N)]
cachev = [cg('_valid', saef, store_dir + str(r) + '/',False) for r in range(N)]
testh5 = pyt.datapacker(wkdir + 'testset.h5')

Nd = np.zeros(N,dtype=np.int32)
Nbf = 0
for f,fn in enumerate(h5files):
    print('Processing file('+ str(f+1) +' of '+ str(len(h5files)) +'):', fn[1])
    adl = pyt.anidataloader(fn)

    To = adl.size()
    Ndc = 0
    Fmt = []
    Emt = []
    for c, data in enumerate(adl):
        #if c == 2 or c == 2 or c == 2:
        # Get test store name
        #Pn = fn.split('/')[-1].split('.')[0] + data['path']
        Pn = data['path']+'_'+str(f).zfill(6)+'_'+str(c).zfill(6)
        #print(Pn)

        # Progress indicator
        sys.stdout.write("\r%d%% %s" % (int(100*c/float(To)), Pn))
        sys.stdout.flush()

        #print(data.keys())

        # Extract the data
        X = data['coordinates']
        E = data['energies']
        F = -data['forces']
        S = data['species']

        Fmt.append(np.max(np.linalg.norm(F,axis=2),axis=1))
        Emt.append(E)
        Mv = np.max(np.linalg.norm(F,axis=2),axis=1)
        #print(Mv.shape,X.shape)
        index = np.where(Mv > 10000000.5)[0]
        indexk = np.where(Mv <= 10000000.5)[0]
        #if index.size > 0:
            #print(Mv[index])
            #hdn.writexyzfile(bddir+'mols_'+str(c).zfill(3)+'_'+str(f).zfill(3)+'.xyz',X[index],S)
        Nbf += index.size

        #if data['path'] == '/dimer7/grp_0':
        #    print(data['path'])
        #    print(E)
        #    print(F)

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
        #Ru = np.random.uniform(0.0, 1.0, E.shape[0])
        #nidx = np.where(Ru < fn[0])
        #X = X[nidx]
        #F = F[nidx]
        #E = E[nidx]

        Ndc += E.size
        #for i in range(E.size):
        #    X[i] = X[0]
        #    F[i] = F[0]
        #    E[i] = E[0]

        if (set(S).issubset(['C', 'N', 'O', 'H', 'F', 'S', 'Cl'])):

            Si = int(E.shape[0]*0.9)

            X_te = X[Si:]
            E_te = E[Si:]
            F_te = F[Si:]
            testh5.store_data(Pn, coordinates=X_te, forces=F_te, energies=E_te, species=list(S))

            X = X[0:Si]
            E = E[0:Si]
            F = F[0:Si]

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
                ## Store training data
                X_t = np.array(np.concatenate([X[s] for j, s in enumerate(split) if j != i]), order='C', dtype=np.float32)
                F_t = np.array(np.concatenate([F[s] for j, s in enumerate(split) if j != i]), order='C', dtype=np.float32)
                E_t = np.array(np.concatenate([E[s] for j, s in enumerate(split) if j != i]), order='C', dtype=np.float64)

                if E_t.shape[0] != 0:
                    t.insertdata(X_t, F_t, E_t, list(S))

                ## Store Validation
                if len(split[i]) > 0:
                    X_v = np.array(X[split[i]], order='C', dtype=np.float32)
                    F_v = np.array(F[split[i]], order='C', dtype=np.float32)
                    E_v = np.array(E[split[i]], order='C', dtype=np.float64)
                    if E_v.shape[0] != 0:
                        v.insertdata(X_v, F_v, E_v, list(S))

    sys.stdout.write("\r%d%%" % int(100))
    print(" Data Kept: ", Ndc, 'High Force: ', Nbf)
    sys.stdout.flush()
    print("")

# Print some stats
print('Data count:',Nd)
print('Data split:',100.0*Nd/np.sum(Nd),'%')

# Save train and valid meta file and cleanup testh5
for t,v in zip(cachet, cachev):
    t.makemetadata()
    v.makemetadata()
testh5.cleanup()
