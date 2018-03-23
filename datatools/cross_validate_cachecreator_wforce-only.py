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

#wkdir = '/home/jujuman/Research/DataReductionMethods/model6r/model-gdb_r06_comb09_1/cv5_6/'
#saef   = wkdir + "sae_6-31gd.dat"

wkdir = '/home/jsmith48/scratch/ccsd_extrapolation/ccsd_train_new/'
saef   = wkdir + "sae_linfit.dat"

#wkdir = '/home/jujuman/Research/DataReductionMethods/modelCNOSFCl/ANI-AL-0605/ANI-AL-0605.0001/cv1/'
#saef   = wkdir + "sae_wb97x-631gd.dat"

data_root = '/home/jsmith48/scratch/ccsd_extrapolation/h5files_combined/'

h5files = [data_root+f for f in os.listdir(data_root) if '.h5' in f]

store_dir = wkdir + "cache-data-"

N = 5

for i in range(N):
    if not os.path.exists(store_dir + str(i)):
        os.mkdir(store_dir + str(i))

    if os.path.exists(store_dir + str(i) + '/../testset/testset'+str(i)+'.h5'):
        os.remove(store_dir + str(i) + '/../testset/testset'+str(i)+'.h5')

    if not os.path.exists(store_dir + str(i) + '/../testset'):
        os.mkdir(store_dir + str(i) + '/../testset')

cachet = [cg('_train', saef, store_dir + str(r) + '/',False) for r in range(N)]
cachev = [cg('_valid', saef, store_dir + str(r) + '/',False) for r in range(N)]
testh5 = [pyt.datapacker(store_dir + str(r) + '/../testset/testset'+str(r)+'.h5') for r in range(N)]

Nd = np.zeros(N,dtype=np.int32)
Nbf = 0
for f,fn in enumerate(h5files):
    print('Processing file('+ str(f+1) +' of '+ str(len(h5files)) +'):', fn)
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
	
        if len(X.shape) == 2:
            X = X.reshape((X.shape[0],-1,3))

        E = data['energies']
        #E = data['energies']
        F = 0.0*X
        #F = data['forces']
        S = data['species']

        #print(X.shape)
        Fmt.append(np.max(np.linalg.norm(F,axis=2),axis=1))
        Emt.append(E)
        Mv = np.max(np.linalg.norm(F,axis=2),axis=1)
        #print(Mv.shape,X.shape)
        index = np.where(Mv > 10.5)[0]
        indexk = np.where(Mv <= 10.5)[0]
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
        Ru = np.random.uniform(0.0, 1.0, E.shape[0])
        nidx = np.where(Ru < 1.0)
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
                    X_te = np.array(X[split[i]], order='C', dtype=np.float32)
                    F_te = np.array(F[split[i]], order='C', dtype=np.float32)
                    E_te = np.array(E[split[i]], order='C', dtype=np.float64)
                    if E_te.shape[0] != 0:
                        te.store_data(Pn, coordinates=X_te, forces=F_te, energies=E_te, species=list(S))

    #plt.hist(np.concatenate(Fmt), bins=150)
    #plt.show()

    #plt.hist(Emt, bins=150)
    #plt.show()

    #sys.stdout.write("\r%d%%" % int(100))
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
