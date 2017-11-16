import os
import hdnntools as hdt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def convert_eformula(S):
    Z = set(S)
    rtn = str()
    for z in Z:
        N = list(S).count(z)
        rtn += z+str(N)
    return rtn

#sdir = "/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb09_1/confs_5/confs/"
#ndir = "/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb09_1/confs_5/confs_iso/"
#sdir = "/home/jujuman/Research/GDB_Dimer/dimer_gen_7/confs/"
#ndir = "/home/jujuman/Research/GDB_Dimer/dimer_gen_7/confs_iso/"

#sdir = "/home/jujuman/Research/Cluster_AL/water/confs/"
#ndir = "/home/jujuman/Research/Cluster_AL/water/confs_iso/"

sdir = "/home/jujuman/Research/GDB-11-AL-wB97x631gd/elements_SFCl/ANI-AL-SFCl/ANI-AL-0605/ANI-AL-0605.0001/confs_1/confs/"
ndir = "/home/jujuman/Research/GDB-11-AL-wB97x631gd/elements_SFCl/ANI-AL-SFCl/ANI-AL-0605/ANI-AL-0605.0001/confs_1/confs_iso/"

#prefix = 'dimers'
#prefix = 'watercluster'
prefix = 'aniSFCl'

files = os.listdir(sdir)
files = [f for f in files if f.rsplit('.',maxsplit=1)[-1] == 'xyz']
print (len(files))

ds = dict()
of = open(ndir+'info_confstoiso_map.dat', 'w')
for i,f in enumerate(files):
    print(sdir+f)
    X, S, N, C = hdt.readxyz2(sdir+f)
    S = np.array(S)

    idx = sorted(range(len(S)), key=lambda k: S[k])
    S = S[np.array(idx)]

    for j,x in enumerate(X):
        X[j] = x[idx]

    id = "".join(S)

    if id in ds:
        sid = len(ds[id])
        of.write(f+' '+convert_eformula(S)+' '+str(sid)+' '+str(X.shape[0])+'\n')
        ds[id].append((X,S))
    else:
        of.write(f+' '+convert_eformula(S)+' '+str(0)+' '+str(X.shape[0])+'\n')
        ds.update({id: [(X,S)]})
of.close()
    #print(i,len(ds))

Nt = 0
for i in ds.keys():
    X = []
    S = []
    for j in ds[i]:
        X.append(j[0])
        S.append(j[1])

    X = np.vstack(X)
    S = list(S[0])
    N = X.shape[0]

    Nt += N

    print(type(S),S)
    fn = prefix + '_' + convert_eformula(S) + '-' + str(N).zfill(5) + '.xyz'
    print('Writing: ',fn)
    hdt.writexyzfile(ndir+fn, X,S)
print('Total data:',Nt)

