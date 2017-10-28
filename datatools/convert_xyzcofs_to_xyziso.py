import os
import hdnntools as hdt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def convert_eformula(sstr):
    Z = set(list(sstr))
    rtn = str()
    for z in Z:
        N = sstr.count(z)
        rtn += z+str(N)
    return rtn

#sdir = "/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb09_1/confs_5/confs/"
#ndir = "/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb09_1/confs_5/confs_iso/"
#sdir = "/home/jujuman/Research/GDB_Dimer/dimer_gen_7/confs/"
#ndir = "/home/jujuman/Research/GDB_Dimer/dimer_gen_7/confs_iso/"

sdir = "/home/jujuman/Research/Cluster_AL/water/confs/"
ndir = "/home/jujuman/Research/Cluster_AL/water/confs_iso/"


#prefix = 'dimers'
prefix = 'watercluster'

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
        sid = np.vstack(ds[id]).shape[0]
        of.write(f+' '+convert_eformula(id)+' '+str(sid)+' '+str(X.shape[0])+'\n')
        ds[id].append(X)
    else:
        of.write(f+' '+convert_eformula(id)+' '+str(0)+' '+str(X.shape[0])+'\n')
        ds.update({id: [X]})
of.close()
    #print(i,len(ds))

Nt = 0
for i in ds.keys():
    ds[i] = np.vstack(ds[i])
    X = ds[i]
    S = list(i)
    N = X.shape[0]

    Nt += N

    fn = prefix + '_' + convert_eformula(i) + '-' + str(N).zfill(5) + '.xyz'
    print('Writing: ',fn)
    hdt.writexyzfile(ndir+fn, X, S)
print('Total data:',Nt)

