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

datlist = ['1',
           '2',
           '3',
           '4',
           '5',
           '6',
           '7',
           '8',]

for d in datlist:
    print('FILE:',d)
    dtdir = '/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnntsgdb11_0'+d+'/data/'
    indir = '/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnntsgdb11_0'+d+'/inputs/'
    h5dir = '/home/jujuman/Research/ANI-DATASET/Compressed_ANI1_Public/ANI-1_release_fixed_2/ani_gdb_s0'+d+'.h5'

    if os.path.exists(h5dir):
        os.remove(h5dir)

    T = 5000.0
    R = 1.0E12

    mtdat = 39370

    files = list(set([f.rsplit('_',1)[0] for f in os.listdir(dtdir)]))
    files.sort()
    ends = ['_train',
            #'_valid',
            #'_test',
            ]

    #tdat = 0
    #tmol = 0
    #ds = []
    #for f in files:
    #    E = []
    #    for e in ends:
    #        data = hdn.readncdat(dtdir+f+e+'.dat')
    #        E.append(np.array(data[2],dtype=np.float64))
    #    E = np.concatenate(E)
    #    tidx = np.where(E - E.min() <= 300.0)
    #    tdat += np.array(tidx).size
    #    ds.append(np.array(tidx).size)
    #    tmol += 1

    #print('total data:',tdat)

    #ddif = tdat - mtdat

    #ds = np.array(ds)
    #cnt = np.ones(ddif,dtype=np.int32)
    #rmtot = np.zeros(tmol,dtype=np.int32)
    #for i in range(tmol):
    #    rmtot[i] = cnt[]

    #print(rmtot)

    dp = pyt.datapacker(h5dir)
    #print('Diff data:',ddif,'tmol:',tmol)
    tmol = 0
    gcount = 0
    bcount = 0
    for i,f in enumerate(files):
        X = []
        E = []
        tmol += 1
        for e in ends:
            data = hdn.readncdat(dtdir+f+e+'.dat')
            X.append(np.array(data[0],dtype=np.float32))
            E.append(np.array(data[2],dtype=np.float64))
            S = data[1]

        X = np.concatenate(X)
        E = np.concatenate(E)

        #pr,dl = boltzmann_prob_ratio(E.min(),E,T)
        gid = np.where(E - E.min() <= 300.0/hdn.hatokcal)
        bid = np.where(E - E.min() > 300.0/hdn.hatokcal)

        #gid = gid[0][0:E[gid].size-rmtot[i]]
        #print(gid.size)

        #bid = list(np.where( pr >  R ))
        #gid = list(np.where( pr <= R ))

        smiles = get_smiles(indir+f+'.ipt')
        #ipd = hdn.read_rcdb_coordsandnm(indir+f+'.ipt')
        gcount += E[gid].size
        bcount += E[bid].size

        dp.store_data(f.split("-")[0]+'/' + f,coordinates=X[gid],
                                              energies=E[gid],
                                              coordinatesHE=X[bid],
                                              energiesHE=E[bid],
                                              species=list(S),
                                              smiles=list(smiles),
                                              )

        #if i % 10 == 0:
            #print(f.split("-")[0]+'/' + f, "{:.3f}".format(100.0*len(bid[0])/float(E.shape[0])) )
    print('good:',gcount)
    print('bad: ',bcount)
    print('mols:',tmol)
    dp.cleanup()