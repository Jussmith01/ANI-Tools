import hdnntools as hdn
import pyanitools as pyt
import numpy as np
import re
import os

Kb = 1.38064852e-23
AJ = 4.35974417e-18

def boltzmann_prob_ratio(ei,E,T):
    c = 1.0/(Kb*T)
    return np.exp(AJ * (E-ei)*c),E-ei

def get_smiles(file):
    f = open(file,'r').read()
    r = re.compile('#Smiles:(.+?)\n')
    s = r.search(f)
    return s.group(1).strip()

dtdir = '/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnntsgdb11_08/data/'
indir = '/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnntsgdb11_08/inputs/'
h5dir = '/home/jujuman/Research/ANI-DATASET/ANI-1_release/ani_gdb_s08.h5'

if os.path.exists(h5dir):
    os.remove(h5dir)

T = 5000.0
R = 1.0E12

files = list(set([f.rsplit('_',1)[0] for f in os.listdir(dtdir)]))
files.sort()
ends = ['_train',
        '_valid',
        '_test',]

dp = pyt.datapacker(h5dir)
for f in files:
    X = []
    E = []
    for e in ends:
        data = hdn.readncdat(dtdir+f+e+'.dat')
        X.append(np.array(data[0],dtype=np.float32))
        E.append(np.array(data[2],dtype=np.float64))
        S = data[1]

    X = np.concatenate(X)
    E = np.concatenate(E)

    pr,dl = boltzmann_prob_ratio(E.min(),E,T)
    bid = list(np.where( pr >  R ))
    gid = list(np.where( pr <= R ))

    smiles = get_smiles(indir+f+'.ipt')
    #ipd = hdn.read_rcdb_coordsandnm(indir+f+'.ipt')

    dp.store_data(f.split("-")[0]+'/' + f,coordinates=X[gid],
                                          energies=E[gid],
                                          coordinatesHE=X[bid],
                                          energiesHE=E[bid],
                                          species=list(S),
                                          smiles=list(smiles),
                                          )

    print(f.split("-")[0]+'/' + f, smiles, "{:.3f}".format(100.0*len(bid[0])/float(E.shape[0])) )

dp.cleanup()