# Python libs
import random as rn
import numpy as np
import time as tm
import os

# NeuroChem Libs
from pyNeuroChem import pyAniTrainer as pyani
from pyNeuroChem import cachegenerator as cg
import pyNeuroChem as pync
import pyanitools as pyt
import hdnntools as hdn

# Scipy
import scipy.spatial as scispc

from rdkit.SimDivFilters import rdSimDivPickers

class anitrainer(object):
    #-----------------------------
    #         Constructor
    # Declare ANI and build layers
    #-----------------------------
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

    def train_network(self,LR,LA,CV,ST,OP,PS=1):
        f = open(OP,'w')

        # Initialize training variables
        best_valid_E = 1000.0
        best_valid_dE = 1000.0
        best_valid_F = 1000.0
        Epoch = 1
        Toler = ST

        # Loop until LR is converged
        Avg_t_err_E = 0
        Avg_v_err_E = 0
        Avg_t_err_dE = 0
        Avg_v_err_dE = 0
        Avg_t_err_F = 0
        Avg_v_err_F = 0
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

                Avg_t_err_E = Avg_t_err_E + np.sqrt(train_cost['Ecost'])
                Avg_v_err_E = Avg_v_err_E + np.sqrt(valid_cost['Ecost'])

                Avg_t_err_dE = Avg_t_err_dE + np.sqrt(train_cost['dEcost'])
                Avg_v_err_dE = Avg_v_err_dE + np.sqrt(valid_cost['dEcost'])

                Avg_t_err_F = Avg_t_err_F + np.sqrt(train_cost['Fcost'])
                Avg_v_err_F = Avg_v_err_F + np.sqrt(valid_cost['Fcost'])

                # Check for better validation
                if Epoch > 1 and (valid_cost['Ecost'] + valid_cost['dEcost']) < (best_valid_E + best_valid_dE):
                    best_valid_E = valid_cost['Ecost']
                    best_valid_dE = valid_cost['dEcost']
                    best_valid_F = valid_cost['Fcost']

                    Toler = ST

                    # saves the current model
                    self.ani.store_model()

                else:
                    # Decrement tol
                    Toler = Toler - 1

                # Print some informations
                if Epoch%PS == 0:
                    self.tr_lc.append(hdn.hatokcal * Avg_t_err_E/float(PS))
                    self.va_lc.append(hdn.hatokcal * Avg_t_err_E/float(PS))

                    output = ('Epoch(' + str(Epoch).zfill(4) + ') Errors: ' +
                             "{:7.3f}".format(hdn.hatokcal * Avg_t_err_E/float(PS)) + ' : ' +
                             "{:7.3f}".format(hdn.hatokcal * Avg_v_err_E/float(PS)) + ' : ' +
                             #"{:7.3f}".format(hdn.hatokcal * Avg_v_err_F/float(PS)), ':',
                             "{:7.3f}".format(hdn.hatokcal * np.sqrt(best_valid_E)) + ' : ' +
                             "{:7.3f}".format(hdn.hatokcal * np.sqrt(best_valid_dE)) +  ' : ' +
                             #"{:7.3f}".format(hdn.hatokcal * np.sqrt(best_valid_F)), 'Time:',
                             "{:.3f}".format(tm.time() - _timeloop) + 's' +
                             ' LR: ' + "{:.3e}".format(LR) + ' Tol: ' + str(Toler).zfill(3))

                    f.write(output+'\n')
                    f.flush()

                    Avg_t_err_E = 0
                    Avg_v_err_E = 0
                    Avg_t_err_dE = 0
                    Avg_v_err_dE = 0
                    Avg_t_err_F = 0
                    Avg_v_err_F = 0

                Epoch = Epoch + 1

            # Anneal LR
            LR = LR * LA

            # Reset Tol
            Toler = ST

        output = ('Final - Epoch(' + str(Epoch).zfill(4) + ') Errors:'+
                 "{:7.3f}".format(hdn.hatokcal * np.sqrt(best_valid_E))+ ' ' +
                 "{:7.3f}".format(hdn.hatokcal * np.sqrt(best_valid_dE)))

        f.write(output + '\n')
        f.flush()

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
        Fcmp = []
        Nmt = 0

        for data in adl:
            # Extract the data
            xyz = data['coordinates']
            frc = data['forces']
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
                    Fact_t = frc[i1:i2]

                    # Set the conformers in NeuroChem
                    self.nc.setConformers(confs=xyz[i1:i2], types=list(spc))

                    Ecmp_t = self.nc.energy()
                    Fcmp_t = self.nc.force()

                    #print(hdn.hatokcal*np.abs(Fact_t-Fcmp_t).sum()/Ecmp_t.size)

                    Ecmp.append(np.sum(np.power(hdn.hatokcal * Ecmp_t - hdn.hatokcal * Eact_t,2)))
                    Fcmp.append(np.sum(np.power(hdn.hatokcal * Fcmp_t.flatten() - hdn.hatokcal * Fact_t.flatten(), 2))/float(3.0*Fact_t.shape[1]))

                    Nmt = Nmt + Ecmp_t.size
                    #Eact.append(Eact_t)

                    #print(hdn.hatokcal * np.sum(np.abs(Ecmp_t-Eact_t))/float(Ecmp_t.size))

        Ecmp = np.array(Ecmp, dtype=np.float64)
        Fcmp = np.array(Fcmp, dtype=np.float64)

        return np.sqrt(np.sum(Ecmp) / float(Nmt)), np.sqrt(np.sum(Fcmp) / float(Nmt))

    def test_for_bad (self, xyz, Eact_W, spc, index, Emax):
        mNa = 100
        if index.size > 0:

            Nm = index.size
            Na = len(spc)

            bad_l = []
            god_l = []

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
                deltas = deltas/np.sqrt(float(Na))
                bad_l.extend([ n for n, i in zip(index, deltas) if i > Emax ])
                god_l.extend([ n for n, i in zip(index, deltas) if i <= Emax ])

            #print(index.size,len(bad_l),len(god_l),len(bad_l)+len(god_l))

            return np.array(bad_l), np.array(god_l), Nm, deltas
        return np.array([]),np.array([]), 0,np.array([])

    def compute_diverse(self, xyz, spc, index, P, aevsize):
        mNa = 100
        Nk = int(np.floor(P*float(index.size)))
        Nk = Nk if Nk < 500 else 500

        #print('Ndiv:',Nk,'Ni:',index.size)
        if Nk > 4 and index.size > 8:
            # Array of random floats from 0 to 1
            selection = np.random.uniform(low=0.0, high=1.0, size=index.size)

            Pt = 1.0 if index.size < 1000 else 1000 / float(index.size)

            # Obtain the sample
            div_idx = np.array([n for n, i in enumerate(selection) if i <= Pt])
            pas_idx = np.array([n for n, i in enumerate(selection) if i > Pt])

            #print(Nk, div_idx.size, index.size, Pt)

            Inh = [i for i,s in enumerate(spc) if s != 'H']

            Nm = div_idx.size
            Na = len(spc)

            div_l = []

            if Na < mNa:
                mNa = Na

            Nat = Na * Nm

            Nit = int(np.ceil(Nat / 65000.0))
            Nmo = int(65000 / Na)
            Nmx = Nm

            aevs = np.empty([Nm,len(Inh)*aevsize])

            for j in range(0, Nit):
                # Setup idicies
                i1 = j * Nmo
                i2 = min(j * Nmo + Nmo, Nm)

                self.nc.setConformers(confs=xyz[div_idx[i1:i2]], types=list(spc))
                Ecmp_t = self.nc.energy()

                for i,a in enumerate(Inh):
                    for m in range(i1, i2):
                        aevs[m,i*aevsize:(i+1)*aevsize] = self.nc.atomicenvironments(a, m).copy()

            dm = scispc.distance.pdist(aevs, 'cosine')
            picker = rdSimDivPickers.MaxMinPicker()
            ids = list(picker.Pick(dm, aevs.shape[0], Nk))

            cur_index = np.array(div_idx[ids])
            new_index = np.array([k for k in range(div_idx.size) if k not in ids])



            #print(cur_index.size,new_index.size,index.size)

            return cur_index, np.concatenate([new_index, pas_idx])
        elif index.size > 0:
            # Array of random floats from 0 to 1
            selection = np.random.uniform(low=0.0, high=1.0, size=index.size)

            # Obtain the sample
            new_index = np.array([n for n, i in enumerate(selection) if i > P ])
            cur_index = np.array([n for n, i in enumerate(selection) if i <= P ])
            return cur_index, new_index
        else:
            return np.array([]),np.array([])

class ActiveANI (object):

    '''' -----------Constructor------------ '''
    def __init__(self, hdf5files, saef, output, storecac, storetest, Naev):
        self.xyz = []
        self.frc = []
        self.Eqm = []
        self.spc = []
        self.idx = []
        self.gid = []
        self.prt = []

        self.Naev = Naev

        self.kid = [] # list to track data kept

        self.nt = [] # total conformers
        self.nc = [] # total kept

        self.of = open(output,'w')

        self.tf = 0

        for f in hdf5files:
            # Construct the data loader class
            adl = pyt.anidataloader(f)
            print('Loading file:',f)

            # Declare test cache
            if os.path.exists(storetest):
                os.remove(storetest)

            dpack = pyt.datapacker(storetest)

            for i, data in enumerate(adl):

                xyz = data['coordinates']
                frc = data['forces']
                eng = data['energies']
                spc = data['species']
                nme = data['path']

                # Toss out high forces
                Mv = np.max(np.linalg.norm(frc, axis=2), axis=1)
                index = np.where(Mv > 1.75)[0]
                indexk = np.where(Mv <= 1.75)[0]

                # CLear forces
                xyz = xyz[indexk]
                frc = frc[indexk]
                eng = eng[indexk]

                idx = np.random.uniform(0.0, 1.0, eng.size)
                tr_idx = np.asarray(np.where(idx < 0.99))[0]
                te_idx = np.asarray(np.where(idx >= 0.99))[0]

                #print(tr_idx)
                if tr_idx.size > 0:
                    self.prt.append(nme)

                    self.xyz.append( np.ndarray.astype(xyz[tr_idx], dtype=np.float32) )
                    self.frc.append( np.ndarray.astype(frc[tr_idx], dtype=np.float32) )
                    self.Eqm.append( np.ndarray.astype(eng[tr_idx], dtype=np.float64) )
                    self.spc.append(spc)

                    Nd = eng[tr_idx].size
                    #print(Nd)

                    self.idx.append( np.arange(Nd) )
                    self.kid.append( np.array([], dtype=np.int) )
                    self.gid.append( np.array([], dtype=np.int))

                    self.tf = self.tf + Nd

                    self.nt.append(Nd)
                    self.nc.append(0)

                # Prepare and store the test data set
                if xyz[te_idx].size != 0:
                    #t_xyz = xyz[te_idx].reshape(te_idx.size, xyz[te_idx].shape[1] * xyz[te_idx].shape[2])
                    dpack.store_data(nme + '/mol' + str(i), coordinates=xyz[te_idx], forces=frc[te_idx], energies=np.array(eng[te_idx]), species=spc)

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

    def init_dataset(self, P, T=0.9, V=0.1):

        # Declare data cache
        cachet = cg('_train', self.saef, self.storecac, False)
        cachev = cg('_valid', self.saef, self.storecac, False)

        for i,(X,F,E,S) in enumerate(zip(self.xyz,self.frc,self.Eqm,self.spc)):

            N = E.shape[0]

            Tp = int(float(T)*float(P)*float(N))
            Vp = int(float(V)*float(P)*float(N))

            # Randomize index
            np.random.shuffle(self.idx[i])

            # get indicies
            iix = np.random.uniform(0.0, 1.0, self.idx[i].size)
            tr_idx = np.asarray(np.where(iix < T*P))[0]
            vd_idx = np.asarray(np.where(iix >= 1.0-(V*P)))[0]

            idxt = self.idx[i][tr_idx].copy()
            idxv = self.idx[i][vd_idx].copy()

            self.kid[i] = np.concatenate([self.kid[i], idxt, idxv])

            self.nc[i] = self.nc[i] + idxt.shape[0] + idxv.shape[0]

            self.ts = self.ts + idxt.shape[0]
            self.vs = self.vs + idxv.shape[0]

            # Update index list
            self.idx[i] = self.idx[i][Tp+Vp+1:]

            # Add data to the cache
            if idxt.shape[0] != 0:
                cachet.insertdata(X[idxt], F[idxt], E[idxt], list(S))

            if idxv.shape[0] != 0:
                cachev.insertdata(X[idxv], F[idxv], E[idxv], list(S))

        print('Full: ', self.tf)
        print('Used: ', self.ts,':',self.vs, ':', self.ts+self.vs)

        # Make meta data file for caches
        cachet.makemetadata()
        cachev.makemetadata()

    def store_diverse(self, cache, atest, X, F, E, S, index, P, T):
        #print(index.shape, index.size)
        if index.size != 0:
            cur_index, new_index = atest.compute_diverse(X, S, index, P * T, self.Naev)

            if cur_index.shape[0] != 0:
                # Get the new index
                #cur_index = index[cur_index]

                # Add data to the cache
                cache.insertdata(X[cur_index], F[cur_index], E[cur_index], list(S))

            if new_index.size != 0:
                return np.array(new_index), np.array(cur_index), cur_index.size
            else:
                return np.array([]), np.array([]), cur_index.size
        return np.array([]), np.array([]), 0


    def store_random(self, cache, X, F, E, S, index, P, T):
        #print(index, index.shape, index.size)
        if index.size != 0:
            # Array of random floats from 0 to 1
            selection = np.random.uniform(low=0.0, high=1.0, size=index.shape[0])

            # Obtain the sample
            new_index = np.asarray(np.where(selection > P * T))[0]
            cur_index = np.asarray(np.where(selection <=  P * T))[0]

            if cur_index.shape[0] != 0:
                # Get the new index
                cur_index = index[cur_index]

                # Add data to the cache
                cache.insertdata(X[cur_index], F[cur_index], E[cur_index], list(S))

            if new_index.size != 0:
                return np.array(index[new_index]), np.array(cur_index), cur_index.size
            else:
                return np.array([]), np.array([]), cur_index.size
        return np.array([]), np.array([]), 0

    def add_bad_data (self, cnstfile, saefile, nnfdir, gpuid, sinet, P, T=0.9, V=0.1, M=0.3):
        atest = anitester(cnstfile, saefile, nnfdir, gpuid, sinet)

        # Declare data cache
        cachet = cg('_train', self.saef, self.storecac, True)
        cachev = cg('_valid', self.saef, self.storecac, True)

        Nbad = 0
        Nadd = 0
        Ngwd = 0
        Ngto = 0

        Nidx = 0
        Nkid = 0
        Ngid = 0

        for i, (X, F, E, S) in enumerate(zip(self.xyz, self.frc, self.Eqm, self.spc)):

            if self.idx[i].size != 0:
                #print('Parent:', self.prt[i])
                # Check if any "Good" milk went sour
                tmp_idx1, self.gid[i], mt, difft = atest.test_for_bad(X,E,S,self.gid[i],M)

                # Add the soured milk to the pot
                self.idx[i] = np.array(np.concatenate([tmp_idx1,self.idx[i]]),dtype=np.int32)

                # Test the pot for good and bad
                self.idx[i],god_idx,m,diff = atest.test_for_bad(X,E,S,self.idx[i],M)

                # Add good to good index
                self.gid[i] = np.array(np.concatenate([self.gid[i],god_idx]),dtype=np.int32)

                # Add to size of good, good went bad, and total bad
                Ngto = Ngto + self.gid[i].size
                Ngwd = Ngwd + tmp_idx1.size
                Nbad = Nbad + self.idx[i].size

                # Store a random subset of the bad for training
                self.idx[i], kat, Nt = self.store_random(cachet, X, F, E, S, self.idx[i], P, T)
                self.idx[i], kav, Nv = self.store_random(cachev, X, F, E, S, self.idx[i], P, V)

                #self.idx[i], kat, Nt = self.store_diverse(cachet, atest, X, F, E, S, self.idx[i], P, T)
                #self.idx[i], kav, Nv = self.store_diverse(cachev, atest, X, F, E, S, self.idx[i], P, V)

                # Add the training data to kid
                self.kid[i] = np.array(np.concatenate([self.kid[i],kat,kav]),dtype=np.int)

                # Count total in the pot
                Nidx = Nidx + self.idx[i].size
                Nkid = Nkid + self.kid[i].size
                Ngid = Ngid + self.gid[i].size

                # Increment training and validation size
                self.ts = self.ts + Nt
                self.vs = self.vs + Nv

                self.nc[i] = self.nc[i] + Nt + Nv

                Nadd = Nadd + Nt + Nv

        self.Nbad = Nbad

        output = '\n--------Data health intformation---------\n' +\
                 '   -Full: ' + str(self.tf) + ' Percent of full used: ' + "{:.2f}".format(100.0*(self.ts+self.vs)/float(self.tf)) + '%\n' +\
                 '   -Used: ' + str(self.ts) + ' : ' + str(self.vs) + ' : ' + str(self.ts+self.vs) + ' Ngwd: ' + str(Ngwd) + '\n' +\
                 '   -Skip: Ngwd: ' +  str(Ngwd) + ' of ' + str(Ngto) + '\n' +\
                 '   -Size: ' + str(Nkid) + ' : ' + str(Nidx) + ' : ' + str(Ngid) + ' : ' + str(Nkid+Nidx+Ngid) + '\n' +\
                 '   -Added: ' + str(Nadd) + ' bad: ' +str(Nbad) + ' of ' + str(Nidx) + ' ('+"{:.1f}".format(self.get_percent_bad())+'%)' + '\n' +\
                 '-----------------------------------------\n\n'

        print(output)
        self.of.write(output)
        self.of.flush()
        # Make meta data file for caches
        cachet.makemetadata()
        cachev.makemetadata()

    def store_train_h5(self,path):
        if os.path.exists(path):
            os.remove(path)

        dpack = pyt.datapacker(path)

        for j, (i,X,F,E,S,P) in enumerate(zip(self.kid,self.xyz,self.frc,self.Eqm,self.spc,self.prt)):
            xyz = X[i]
            frc = F[i]
            eng = E[i]
            spc = S
            nme = P

            # Prepare and store the test data set
            if xyz.size != 0:
                dpack.store_data(nme + '/mol' + str(j), coordinates=xyz, forces=frc, energies=eng, species=spc)

        # Cleanup the disk
        dpack.cleanup()

    def get_keep_info(self):
        for i,(c,t) in enumerate(zip(self.nc,self.nt)):
            yield (self.spc[i], 100.0 * (c/float(t)))

    def get_diff_kept(self, cnstfile, saefile, nnfdir, gpuid, sinet, M):
        atest = anitester(cnstfile, saefile, nnfdir, gpuid, sinet)
        for k,X,E,S in zip(self.kid,self.xyz,self.Eqm,self.spc):
            index, index2, m, diff = atest.test_for_bad(X, E, S, k, M)
            yield diff
