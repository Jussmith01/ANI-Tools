from pyNeuroChem import pyAniTrainer as pyani
from pyNeuroChem import cachegenerator as cg
import pyNeuroChem as pync
import pyanitools as pyt

import hdnntools as hdn
import random as rn
import numpy as np
import time as tm
import os

class anitrainer(object):
    def __init__(self, build_dict, layers):
        # Declare the trainer class
        self.ani = pyani(build_dict)

        # Add layers
        for l in layers:
            self.ani.add_layer(l)

        # Build network
        self.ani.build_network()

        self.tr_lc = []
        self.va_lc = []

    def write_learning_curve(self,file):
        o = open(file,'w')
        for i,(t,v) in enumerate(zip(self.tr_lc,self.va_lc)):
            o.write(str(i) + ' ' + "{:.5f}".format(t) + ' ' + "{:.5f}".format(v) + '\n')
        o.close()

    def train_network(self,LR,LA,CV,ST,PS=1):
        # Initialize training variables
        best_valid = 100.0
        Epoch = 1
        Toler = ST

        # Loop until LR is converged
        Avg_t_err = 0
        Avg_v_err = 0
        while LR > CV:
            # Loop until tol is 0
            while Toler is not 0:
                _timeloop = tm.time()

                # Train and validate network
                self.ani.train(learning_rate=LR)
                self.ani.valid()

                # Get cost information
                train_cost = self.ani.gettcost()
                valid_cost = self.ani.getvcost()

                Avg_t_err = Avg_t_err + np.sqrt(train_cost)
                Avg_v_err = Avg_v_err + np.sqrt(valid_cost)

                # Check for better validation
                if (valid_cost < best_valid):
                    best_valid = valid_cost

                    Toler = ST

                    # saves the current model
                    self.ani.store_model()

                else:
                    # Decrement tol
                    Toler = Toler - 1

                # Print some informations
                if Epoch%PS == 0:
                    self.tr_lc.append(hdn.hatokcal * Avg_t_err/float(PS))
                    self.va_lc.append(hdn.hatokcal * Avg_t_err/float(PS))

                    print('Epoch(', str(Epoch).zfill(4), ') Errors:',
                          "{:7.3f}".format(hdn.hatokcal * Avg_t_err/float(PS)), ':',
                          "{:7.3f}".format(hdn.hatokcal * Avg_v_err/float(PS)), ':',
                          "{:7.3f}".format(hdn.hatokcal * np.sqrt(best_valid)), 'Time:',
                          "{:.3f}".format(tm.time() - _timeloop) + 's',
                          'LR:', "{:.3e}".format(LR), 'Tol:', str(Toler).zfill(3))

                    Avg_t_err = 0
                    Avg_v_err = 0

                Epoch = Epoch + 1

            # Anneal LR
            LR = LR * LA

            # Reset Tol
            Toler = ST

        print('Final - Epoch(', str(Epoch).zfill(4), ') Errors:',
              "{:7.3f}".format(hdn.hatokcal * np.sqrt(best_valid)))

class anitester (object):
    def __init__(self, cnstfile, saefile, nnfdir, gpuid, sinet):
        # Construct pyNeuroChem classes
        self.nc = pync.conformers(cnstfile, saefile, nnfdir, gpuid, sinet)

    def compute_test (self, h5file):
        mNa = 100

        # Declare loader
        adl = pyt.anidataloader(h5file)

        # Declare containers
        Eact = []
        Ecmp = []
        Nmt = 0

        for data in adl:
            # Extract the data
            xyz = data['coordinates']
            Eqm = data['energies']
            spc = data['species']

            xyz = xyz.reshape(Eqm.shape[0], len(spc), 3)

            if xyz.shape[0] > 0:
                Nm = xyz.shape[0]
                Na = xyz.shape[1]

                if Na < mNa:
                    mNa = Na

                Nat = Na * Nm

                Nit = int(np.ceil(Nat / 65000.0))
                Nmo = int(65000 / Na)
                Nmx = Nm

                for j in range(0, Nit):
                    # Setup idicies
                    i1 = j * Nmo
                    i2 = min(j * Nmo + Nmo, Nm)

                    # copy array subset
                    Eact_t = Eqm[i1:i2]

                    # Set the conformers in NeuroChem
                    self.nc.setConformers(confs=xyz[i1:i2], types=list(spc))

                    Ecmp_t = self.nc.energy()

                    Ecmp.append(np.sum(np.power(hdn.hatokcal * Ecmp_t - hdn.hatokcal * Eact_t,2)))
                    Nmt = Nmt + Ecmp_t.size
                    #Eact.append(Eact_t)

                    #print(hdn.hatokcal * np.sum(np.abs(Ecmp_t-Eact_t))/float(Ecmp_t.size))

        Ecmp = np.array(Ecmp, dtype=np.float64)

        return np.sqrt(np.sum(Ecmp) / float(Nmt))

    def test_for_bad (self, xyz, Eact_W, spc, index, Emax):
        mNa = 100
        if index.size > 0:

            Nm = index.size
            Na = len(spc)

            bad_l = []

            if Na < mNa:
                mNa = Na

            Nat = Na * Nm

            Nit = int(np.ceil(Nat / 65000.0))
            Nmo = int(65000 / Na)
            Nmx = Nm

            for j in range(0, Nit):

                # Setup idicies
                i1 = j * Nmo
                i2 = min(j * Nmo + Nmo, Nm)

                # copy array subset
                Eact_t = Eact_W[index[i1:i2]]

                self.nc.setConformers(confs=xyz[index[i1:i2]], types=list(spc))
                Ecmp_t = self.nc.energy()

                deltas = hdn.hatokcal * np.abs(Ecmp_t - np.array(Eact_t, dtype=float))
                deltas = deltas/float(Na)
                bad_l = [ n for n,i in zip(index,deltas) if i>Emax ]

            return np.array(bad_l),Nm, deltas
        return np.array([]), 0

class ActiveANI (object):

    '''' -----------Constructor------------ '''
    def __init__(self, hdf5files, saef, storecac, storetest):
        self.xyz = []
        self.Eqm = []
        self.spc = []
        self.idx = []
        self.prt = []

        self.kid = [] # list to track data kept

        self.nt = [] # total conformers
        self.nc = [] # total kept

        self.tf = 0

        for f in hdf5files:
            # Construct the data loader class
            adl = pyt.anidataloader(f)

            # Declare test cache
            if os.path.exists(storetest):
                os.remove(storetest)

            dpack = pyt.datapacker(storetest)

            for i, data in enumerate(adl):
                xyz = np.array_split(data['coordinates'], 10)
                eng = np.array_split(data['energies'], 10)
                spc = data['species']
                nme = data['parent']

                self.prt.append(nme)

                self.xyz.append( np.concatenate(xyz[0:9]) )
                self.Eqm.append( np.concatenate(eng[0:9]) )
                self.spc.append(spc)

                Nd = np.concatenate(eng[0:9]).shape[0]

                self.idx.append( np.arange(Nd) )
                self.kid.append( np.array([], dtype=np.int) )

                self.tf = self.tf + Nd

                self.nt.append(Nd)
                self.nc.append(0)

                # Prepare and store the test data set
                if xyz[9].size != 0:
                    t_xyz = xyz[9].reshape(xyz[9].shape[0], xyz[9].shape[1] * xyz[9].shape[2])
                    dpack.store_data(nme + '/mol' + str(i), coordinates=t_xyz, energies=np.array(eng[9]), species=spc)

            # Clean up
            adl.cleanup()

            # Clean up
            dpack.cleanup()

        self.nt = np.array(self.nt)
        self.nc = np.array(self.nc)

        self.ts = 0
        self.vs = 0

        self.Nbad = self.tf

        self.saef = saef
        self.storecac = storecac

    def get_Nbad(self):
        return self.Nbad

    def get_percent_bad(self):
        return 100.0 * (self.Nbad/float(self.tf))

    def init_dataset(self, P, T=0.8, V=0.2):

        # Declare data cache
        cachet = cg('_train', self.saef, self.storecac, False)
        cachev = cg('_valid', self.saef, self.storecac, False)

        for i,(X,E,S) in enumerate(zip(self.xyz,self.Eqm,self.spc)):
            N = E.shape[0]

            Tp = int(float(T)*float(P)*float(N))
            Vp = int(float(V)*float(P)*float(N))

            # Randomize index
            np.random.shuffle(self.idx[i])

            # get indicies
            idxt = self.idx[i][0:Tp].copy()
            idxv = self.idx[i][Tp+1:Tp+Vp].copy()

            self.kid[i] = np.concatenate([self.kid[i], idxt])

            self.nc[i] = self.nc[i] + idxt.shape[0] + idxv.shape[0]

            self.ts = self.ts + idxt.shape[0]
            self.vs = self.vs + idxv.shape[0]

            # Update index list
            self.idx[i] = self.idx[i][Tp+Vp+1:]

            # Add data to the cache
            if idxt.shape[0] != 0:
                cachet.insertdata(X[idxt], E[idxt], list(S))

            if idxv.shape[0] != 0:
                cachev.insertdata(X[idxv], E[idxv], list(S))

        print('Full: ', self.tf)
        print('Used: ', self.ts,':',self.vs, ':', self.ts+self.vs)

        # Make meta data file for caches
        cachet.makemetadata()
        cachev.makemetadata()

    def store_random(self, cache, X, E, S, index, P, T):
        #print(index, index.shape, index.size)
        if index.size != 0:
            # Array of random floats from 0 to 1
            selection = np.random.uniform(low=0.0, high=1.0, size=index.shape[0])

            # Obtain the sample
            new_index = np.array([n for n, i in enumerate(selection) if i > P * T])
            cur_index = np.array([n for n, i in enumerate(selection) if i <= P * T])

            if cur_index.shape[0] != 0:
                # Get the new index
                cur_index = index[cur_index]

                # Add data to the cache
                cache.insertdata(X[cur_index], E[cur_index], list(S))

            if new_index.size != 0:
                return np.array(index[new_index]), np.array(cur_index), cur_index.size
            else:
                return np.array([]), np.array([]), cur_index.size
        return np.array([]), np.array([]), 0

    def add_bad_data (self, cnstfile, saefile, nnfdir, gpuid, sinet, P, T=0.8, V=0.2, M=0.06):
        atest = anitester(cnstfile, saefile, nnfdir, gpuid, sinet)

        # Declare data cache
        cachet = cg('_train', self.saef, self.storecac, True)
        cachev = cg('_valid', self.saef, self.storecac, True)

        Nidx = 0
        Nbad = 0
        Nadd = 0

        for i, (X, E, S) in enumerate(zip(self.xyz, self.Eqm, self.spc)):

            if self.idx[i].size != 0:
                Nidx = Nidx + self.idx[i].size
                self.idx[i],m,diff = atest.test_for_bad(X,E,S,self.idx[i],M)

                Nbad = Nbad + self.idx[i].shape[0]

                self.idx[i], kat, Nt = self.store_random(cachet, X, E, S, self.idx[i], P, T)
                self.idx[i], kav, Nv = self.store_random(cachev, X, E, S, self.idx[i], P, V)

                self.kid[i] = np.array(np.concatenate([self.kid[i],kat]),dtype=np.int)

                self.ts = self.ts + Nt
                self.vs = self.vs + Nv

                self.nc[i] = self.nc[i] + Nt + Nv

                Nadd = Nadd + Nt + Nv

        self.Nbad = Nbad

        print('\n--------Data health intformation---------')
        print('   -Full: ', self.tf, 'Percent of full used:',"{:.2f}".format(100.0*(self.ts+self.vs)/float(self.tf))+'%')
        print('   -Used: ', self.ts,':',self.vs, ':', self.ts+self.vs)
        print('   -Added:', Nadd,' bad:',Nbad,'of',Nidx,'('+"{:.1f}".format(self.get_percent_bad())+'%)')
        print('-----------------------------------------\n')

        # Make meta data file for caches
        cachet.makemetadata()
        cachev.makemetadata()

    def store_train_h5(self,path):
        if os.path.exists(path):
            os.remove(path)

        dpack = pyt.datapacker(path)

        for j, (i,X,E,S,P) in enumerate(zip(self.kid,self.xyz,self.Eqm,self.spc,self.prt)):
            xyz = X[i]
            eng = E[i]
            spc = S
            nme = P

            # Prepare and store the test data set
            if xyz.size != 0:
                dpack.store_data(nme + '/mol' + str(j), coordinates=xyz, energies=eng, species=spc)

        # Cleanup the disk
        dpack.cleanup()

    def get_keep_info(self):
        for i,(c,t) in enumerate(zip(self.nc,self.nt)):
            yield (self.spc[i], 100.0 * (c/float(t)))

    def get_diff_kept(self, cnstfile, saefile, nnfdir, gpuid, sinet, M):
        atest = anitester(cnstfile, saefile, nnfdir, gpuid, sinet)
        for k,X,E,S in zip(self.kid,self.xyz,self.Eqm,self.spc):
            index, m, diff = atest.test_for_bad(X, E, S, k, M)
            yield diff



# Network 1 Files
wkdir = '/home/jujuman/Research/DataReductionMethods/models/train_c08f/'
cnstf = 'rHCNO-4.6A_16-3.1A_a4-8.params'
saenf = 'sae_6-31gd.dat'
nfdir = 'networks/'

# Data Dir
datadir = '/home/jujuman/Research/DataReductionMethods/models/cache/'
testdata = datadir + 'testset/testset.h5'
trainh5 = wkdir + 'ani_red_c08f.h5'

# Test data
test_files = ["/home/jujuman/Research/ANI-DATASET/h5data/ani-gdb-c08f.h5",
              #"/home/jujuman/Research/ANI-DATASET/h5data/ani-gdb-c02.h5",
              #"/home/jujuman/Research/ANI-DATASET/h5data/ani-gdb-c03.h5",
              #"/home/jujuman/Research/ANI-DATASET/h5data/ani-gdb-c04.h5",
              #"/home/jujuman/Research/ANI-DATASET/h5data/ani-gdb-c05.h5",
              #"/home/jujuman/Research/ANI-DATASET/h5data/ani-gdb-c06.h5",
              ]

#---- Parameters ----
GPU = 0
LR  = 0.001
LA  = 0.25
CV  = 1.0e-6
ST  = 100
M   = 0.05 # Max error per atom in kcal/mol
P   = 0.025
ps  = 25
#--------------------

# Training varibles
d = dict({'wkdir'         : wkdir,
          'sflparamsfile' : cnstf,
          'ntwkStoreDir'  : wkdir+'networks/',
          'atomEnergyFile': saenf,
          'datadir'       : datadir,
          'tbtchsz'       : '512',
          'vbtchsz'       : '1024',
          'gpuid'         : str(GPU),
          'ntwshr'        : '1',
          'nkde'          : '2',
          'runtype'       : 'ANNP_CREATE_HDNN_AND_TRAIN',
          'adptlrn'       : 'OFF',
          'moment'        : 'ADAM',})

l1 = dict({'nodes'      : '256',
           'activation' : '5',
           'maxnorm'    : '1',
           'norm'       : '3.0',
           'btchnorm'   : '0',})

l2 = dict({'nodes'      : '128',
           'activation' : '5',
           'maxnorm'    : '1',
           'norm'       : '3.0',
           'btchnorm'   : '0',})

l3 = dict({'nodes'      : '64',
           'activation' : '5',
           'maxnorm'    : '1',
           'norm'       : '3.0',
           'btchnorm'   : '0',})

l4 = dict({'nodes'      : '1',
           'activation' : '6',})

layers = [l1, l2, l3, l4]

aani = ActiveANI(test_files, wkdir+saenf, datadir, testdata)
aani.init_dataset(P)

inc = 0
while aani.get_percent_bad() > 1.0:
    # Remove existing network
    network_files = os.listdir(wkdir + 'networks/')
    for f in network_files:
        os.remove(wkdir + 'networks/' + f)

    # Setup trainer
    tr = anitrainer(d,layers)

    # Train network
    tr.train_network(LR, LA, CV, ST, ps)

    # Write the learning curve
    tr.write_learning_curve(wkdir+'learning_curve_'+str(inc)+'.dat')

    # Test network
    ant = anitester(wkdir+cnstf, wkdir+saenf, wkdir+nfdir, GPU, True)
    test_rmse = ant.compute_test(testdata)
    print('Test RMSE:',"{:.3f}".format(test_rmse),'kcal/mol')

    # Check for and add bad data
    aani.add_bad_data(wkdir+cnstf, wkdir+saenf, wkdir+nfdir, GPU, True, P=0.05 + inc * 0.025, M=M)

    inc = inc + 1

aani.add_bad_data(wkdir + cnstf, wkdir + saenf, wkdir + nfdir, GPU, True, P=1.0, M=M)
aani.store_train_h5(trainh5)

# Remove existing network
network_files = os.listdir(wkdir + 'networks/')
for f in network_files:
    os.remove(wkdir + 'networks/' + f)

# Setup trainer
tr = anitrainer(d, layers)

# Train network
tr.train_network(LR, LA, CV, ST, ps)

o = open(wkdir + 'keep_info.dat', 'w')
for k in aani.get_keep_info():
    o.write(str(int(k[1])) + ' : ' + str(k[0]) + '\n')

f = open(wkdir + 'diffs.dat', 'w')
for K in aani.get_diff_kept (wkdir + cnstf, wkdir + saenf, wkdir + nfdir, GPU, True, M=M):
    string = ""
    for k in K:
        string = string + "{:.7f}".format(k) + ','
    f.write(string[:-1] + '\n')
