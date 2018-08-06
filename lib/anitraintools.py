import hdnntools as hdt
import nmstools as nmt
import pyanitools as pyt
import pyanitrainer as atr
import pymolfrag as pmf

from pyNeuroChem import cachegenerator as cg

import numpy as np

from time import sleep
import subprocess
import random
#import pyssh
import re
import os

from multiprocessing import Process
import shutil

import matplotlib.pyplot as plt

def interval(v,S):
    ps = 0.0
    ds = 1.0 / float(S)
    for s in range(S):
        if v > ps and v <= ps+ds:
            return s
        ps = ps + ds

class anitrainerinputdesigner:
    def __init__(self):
        self.params = {"sflparamsfile":None, # AEV parameters file
                       "ntwkStoreDir":"networks/", # Store network dir
                       "atomEnergyFile":None, # Atomic energy shift file
                       "nmax": 0, # Max training iterations
                       "tolr": 50, # Annealing tolerance (patience)
                       "emult": 0.5, # Annealing multiplier
                       "eta": 0.001, # Learning rate
                       "tcrit": 1.0e-5, # eta termination crit.
                       "tmax": 0, # Maximum time (0 = inf)
                       "tbtchsz": 2048, # training batch size
                       "vbtchsz": 2048, # validation batch size
                       "gpuid": 0, # Default GPU id (is overridden by -g flag for HDAtomNNP-Trainer exe)
                       "ntwshr": 0, # Use a single network for all types... (THIS IS BROKEN, DO NOT USE)
                       "nkde": 2, # Energy delta regularization
                       "energy": 1, # Enable/disable energy training
                       "force": 0, # Enable/disable force training
                       "fmult": 1.0, # Multiplier of force cost
                       "pbc": 0, # Use PBC in training (Warning, this only works for data with a single rect. box size)
                       "cmult": 1.0, # Charge cost multiplier (CHARGE TRAINING BROKEN IN CURRENT VERSION)
                       "runtype" : "ANNP_CREATE_HDNN_AND_TRAIN", # DO NOT CHANGE - For NeuroChem backend
                       "adptlrn" : "OFF",
                       "decrate" : 0.9,
                       "moment" : "ADAM",
                       "mu" : 0.99
                       }

        self.layers = dict()

    def add_layer(self, atomtype, layer_dict):
        layer_dict.update({"type":0})
        if atomtype not in self.layers:
            self.layers[atomtype]=[layer_dict]
        else:
            self.layers[atomtype].append(layer_dict)

    def set_parameter(self,key,value):
        self.params[key]=value

    def print_layer_parameters(self):
        for ak in self.layers.keys():
            print('Species:',ak)
            for l in self.layers[ak]:
                print('  -',l)

    def print_training_parameters(self):
        print(self.params)

    def __get_value_string__(self,value):

        if type(value)==float:
            string="{0:10.7e}".format(value)
        else:
            string=str(value)

        return string

    def __build_network_str__(self, iptsize):

        network  = "network_setup {\n"
        network += "    inputsize="+str(iptsize)+";\n"

        for ak in self.layers.keys():
            network += "    atom_net " + ak + " $\n"
            self.layers[ak].append({"nodes":1,"activation":6,"type":0})
            for l in self.layers[ak]:
                network += "        layer [\n"
                for key in l.keys():
                    network += "            "+key+"="+self.__get_value_string__(l[key])+";\n"
                network += "        ]\n"
            network += "    $\n"
        network += "}\n"
        return network

    def write_input_file(self, file, iptsize):
        f = open(file,'w')
        for key in self.params.keys():
            f.write(key+'='+self.__get_value_string__(self.params[key])+'\n')
        f.write(self.__build_network_str__(iptsize))
        f.close()


class alaniensembletrainer():
    def __init__(self, train_root, netdict, h5dir, Nn):
        self.train_root = train_root
        #self.train_pref = train_pref
        self.h5dir = h5dir
        self.Nn = Nn
        self.netdict = netdict

        self.h5file = [f for f in os.listdir(self.h5dir) if f.rsplit('.',1)[1] == 'h5']
        #print(self.h5dir,self.h5file)

    def build_training_cache(self,forces=True):
        store_dir = self.train_root + "cache-data-"
        N = self.Nn

        for i in range(N):
            if not os.path.exists(store_dir + str(i)):
                os.mkdir(store_dir + str(i))

            if os.path.exists(store_dir + str(i) + '/../testset/testset' + str(i) + '.h5'):
                os.remove(store_dir + str(i) + '/../testset/testset' + str(i) + '.h5')

            if not os.path.exists(store_dir + str(i) + '/../testset'):
                os.mkdir(store_dir + str(i) + '/../testset')

        cachet = [cg('_train', self.netdict['saefile'], store_dir + str(r) + '/', False) for r in range(N)]
        cachev = [cg('_valid', self.netdict['saefile'], store_dir + str(r) + '/', False) for r in range(N)]
        testh5 = [pyt.datapacker(store_dir + str(r) + '/../testset/testset' + str(r) + '.h5') for r in range(N)]

        Nd = np.zeros(N, dtype=np.int32)
        Nbf = 0
        for f, fn in enumerate(self.h5file):
            print('Processing file(' + str(f + 1) + ' of ' + str(len(self.h5file)) + '):', fn)
            adl = pyt.anidataloader(self.h5dir+fn)

            To = adl.size()
            Ndc = 0
            Fmt = []
            Emt = []
            for c, data in enumerate(adl):
                Pn = data['path'] + '_' + str(f).zfill(6) + '_' + str(c).zfill(6)

                # Progress indicator
                #sys.stdout.write("\r%d%% %s" % (int(100 * c / float(To)), Pn))
                #sys.stdout.flush()

                # print(data.keys())

                # Extract the data
                X = data['coordinates']
                E = data['energies']
                S = data['species']

                # 0.0 forces if key doesnt exist
                if forces:
                    F = data['forces']
                else:
                    F = 0.0*X

                Fmt.append(np.max(np.linalg.norm(F, axis=2), axis=1))
                Emt.append(E)
                Mv = np.max(np.linalg.norm(F, axis=2), axis=1)

                index = np.where(Mv > 10.5)[0]
                indexk = np.where(Mv <= 10.5)[0]

                Nbf += index.size

                # CLear forces
                X = X[indexk]
                F = F[indexk]
                E = E[indexk]

                Esae = hdt.compute_sae(self.netdict['saefile'],S)

                hidx = np.where(np.abs(E-Esae) > 10.0)
                lidx = np.where(np.abs(E-Esae) <= 10.0)
                if hidx[0].size > 0:
                    print('  -('+str(c).zfill(3)+')High energies detected:\n    ',E[hidx])

                X = X[lidx]
                E = E[lidx]
                F = F[lidx]

                Ndc += E.size

                if (set(S).issubset(self.netdict['atomtyp'])):
                #if (set(S).issubset(['C', 'N', 'O', 'H', 'F', 'S', 'Cl'])):

                    # Random mask
                    R = np.random.uniform(0.0, 1.0, E.shape[0])
                    idx = np.array([interval(r, N) for r in R])

                    # Build random split lists
                    split = []
                    for j in range(N):
                        split.append([i for i, s in enumerate(idx) if s == j])
                        nd = len([i for i, s in enumerate(idx) if s == j])
                        Nd[j] = Nd[j] + nd

                    # Store data
                    for i, t, v, te in zip(range(N), cachet, cachev, testh5):
                        ## Store training data
                        X_t = np.array(np.concatenate([X[s] for j, s in enumerate(split) if j != i]), order='C',
                                       dtype=np.float32)
                        F_t = np.array(np.concatenate([F[s] for j, s in enumerate(split) if j != i]), order='C',
                                       dtype=np.float32)
                        E_t = np.array(np.concatenate([E[s] for j, s in enumerate(split) if j != i]), order='C',
                                       dtype=np.float64)

                        if E_t.shape[0] != 0:
                            t.insertdata(X_t, F_t, E_t, list(S))

                        ## Split test/valid data and store\
                        #tv_split = np.array_split(split[i], 2)

                        ## Store Validation
                        if np.array(split[i]).size > 0:
                            X_v = np.array(X[split[i]], order='C', dtype=np.float32)
                            F_v = np.array(F[split[i]], order='C', dtype=np.float32)
                            E_v = np.array(E[split[i]], order='C', dtype=np.float64)
                            if E_v.shape[0] != 0:
                                v.insertdata(X_v, F_v, E_v, list(S))

                        ## Store testset
                        #if tv_split[1].size > 0:
                            #X_te = np.array(X[split[i]], order='C', dtype=np.float32)
                            #F_te = np.array(F[split[i]], order='C', dtype=np.float32)
                            #E_te = np.array(E[split[i]], order='C', dtype=np.float64)
                            #if E_te.shape[0] != 0:
                            #    te.store_data(Pn, coordinates=X_te, forces=F_te, energies=E_te, species=list(S))


            #sys.stdout.write("\r%d%%" % int(100))
            #print(" Data Kept: ", Ndc, 'High Force: ', Nbf)
            #sys.stdout.flush()
            #print("")

        # Print some stats
        print('Data count:', Nd)
        print('Data split:', 100.0 * Nd / np.sum(Nd), '%')

        # Save train and valid meta file and cleanup testh5
        for t, v, th in zip(cachet, cachev, testh5):
            t.makemetadata()
            v.makemetadata()
            th.cleanup()

    def sae_linear_fitting(self, Ekey='energies', energy_unit=1.0, Eax0sum=False):
        from sklearn import linear_model
        print('Performing linear fitting...')

        datadir = self.h5dir
        sae_out = self.netdict['saefile']

        smap = dict()
        for i,Z in enumerate(self.netdict['atomtyp']):
            smap.update({Z:i})

        Na = len(smap)
        files = os.listdir(datadir)

        X = []
        y = []
        for f in files[0:20]:
            print(f)
            adl = pyt.anidataloader(datadir + f)
            for data in adl:
                # print(data['path'])
                S = data['species']

                if data[Ekey].size > 0:
                    if Eax0sum:
                        E = energy_unit*np.sum(np.array(data[Ekey], order='C', dtype=np.float64), axis=1)
                    else:
                        E = energy_unit*np.array(data[Ekey], order='C', dtype=np.float64)

                    S = S[0:data['coordinates'].shape[1]]
                    unique, counts = np.unique(S, return_counts=True)
                    x = np.zeros(Na, dtype=np.float64)
                    for u, c in zip(unique, counts):
                        x[smap[u]] = c

                    for e in E:
                        X.append(np.array(x))
                        y.append(np.array(e))

        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        lin = linear_model.LinearRegression(fit_intercept=False)
        lin.fit(X, y)

        coef = lin.coef_
        print(coef)

        sae = open(sae_out, 'w')
        for i, c in enumerate(coef[0]):
            sae.write(next(key for key, value in smap.items() if value == i) + ',' + str(i) + '=' + str(c) + '\n')

        sae.close()

        print('Linear fitting complete.')

    def build_strided_training_cache(self,Nblocks,Nvalid,Ntest,build_test=True, forces=True, grad=False, Fkey='forces', forces_unit=1.0, Ekey='energies', energy_unit=1.0, Eax0sum=False, rmhighe=True):
        if not os.path.isfile(self.netdict['saefile']):
            self.sae_linear_fitting(Ekey=Ekey, energy_unit=energy_unit, Eax0sum=Eax0sum)

        h5d = self.h5dir

        store_dir = self.train_root + "cache-data-"
        N = self.Nn
        Ntrain = Nblocks - Nvalid - Ntest

        if Nblocks % N != 0:
            raise ValueError('Error: number of networks must evenly divide number of blocks.')

        Nstride = Nblocks/N

        for i in range(N):
            if not os.path.exists(store_dir + str(i)):
                os.mkdir(store_dir + str(i))

            if build_test:
                if os.path.exists(store_dir + str(i) + '/../testset/testset' + str(i) + '.h5'):
                    os.remove(store_dir + str(i) + '/../testset/testset' + str(i) + '.h5')

                if not os.path.exists(store_dir + str(i) + '/../testset'):
                    os.mkdir(store_dir + str(i) + '/../testset')

        cachet = [cg('_train', self.netdict['saefile'], store_dir + str(r) + '/', False) for r in range(N)]
        cachev = [cg('_valid', self.netdict['saefile'], store_dir + str(r) + '/', False) for r in range(N)]

        if build_test:
            testh5 = [pyt.datapacker(store_dir + str(r) + '/../testset/testset' + str(r) + '.h5') for r in range(N)]

        if rmhighe:
            dE = []
            for f in self.h5file:
                adl = pyt.anidataloader(h5d+f)
                for data in adl:
                    S = data['species']
                    E = data['energies']
                    X = data['coordinates']
    
                    Esae = hdt.compute_sae(self.netdict['saefile'], S)
    
                    dE.append((E-Esae)/np.sqrt(len(S)))
    
            dE = np.concatenate(dE)
            cidx = np.where(np.abs(dE) < 15.0)
            std = np.abs(dE[cidx]).std()
            men = np.mean(dE[cidx])

            print(men,std,men+std)
            idx = np.intersect1d(np.where(dE>=-np.abs(15*std+men))[0],np.where(dE<=np.abs(11*std+men))[0])
            cnt = idx.size
            print('DATADIST: ',dE.size,cnt,(dE.size-cnt),100.0*((dE.size-cnt)/dE.size))

        E = []
        data_count = np.zeros((N,3),dtype=np.int32)
        for f in self.h5file:
            print('Reading data file:',h5d+f)
            adl = pyt.anidataloader(h5d+f)
            for data in adl:
                #print(data['path'],data['energies'].size)

                S = data['species']

                if data[Ekey].size > 0 and (set(S).issubset(self.netdict['atomtyp'])):

                    X = np.array(data['coordinates'], order='C',dtype=np.float32)

                    #print(np.array(data[Ekey].shape),np.sum(np.array(data[Ekey], order='C', dtype=np.float64),axis=1).shape,data[Fkey].shape)

                    if Eax0sum:
                        E = energy_unit*np.sum(np.array(data[Ekey], order='C', dtype=np.float64),axis=1)
                    else:
                        E = energy_unit*np.array(data[Ekey], order='C',dtype=np.float64)

                    if forces and not grad:
                        F = forces_unit*np.array(data[Fkey], order='C', dtype=np.float32)
                    elif forces and grad:
                        F = -forces_unit*np.array(data[Fkey], order='C', dtype=np.float32)
                    else:
                        F = 0.0*X

                    if rmhighe:
                        Esae = hdt.compute_sae(self.netdict['saefile'], S)

                        ind_dE = (E - Esae)/np.sqrt(len(S))

                        hidx = np.union1d(np.where(ind_dE<-(15.0*std+men))[0],np.where(ind_dE>(11.0*std+men))[0])
                        lidx = np.intersect1d(np.where(ind_dE>=-(15.0*std+men))[0],np.where(ind_dE<=(11.0*std+men))[0])

                        if hidx.size > 0:
                            print('  -(' + f + ':' + data['path'] + ')High energies detected:\n    ', (E[hidx]-Esae)/np.sqrt(len(S)))

                        X = X[lidx]
                        E = E[lidx]
                        F = F[lidx]

                    # Build random split index
                    ridx = np.random.randint(0,Nblocks,size=E.size)
                    Didx = [np.argsort(ridx)[np.where(ridx == i)] for i in range(Nblocks)]

                    # Build training cache
                    for nid,cache in enumerate(cachet):
                        set_idx = np.concatenate([Didx[((bid+nid*int(Nstride)) % Nblocks)] for bid in range(Ntrain)])
                        if set_idx.size != 0:
                            data_count[nid,0]+=set_idx.size
                            cache.insertdata(X[set_idx], F[set_idx], E[set_idx], list(S))

                    print('test test',Ntrain)
                    # for nid,cache in enumerate(cachev):
                    #     set_idx = np.concatenate([Didx[((1+bid+nid*int(Nstride)) % Nblocks)] for bid in range(Ntrain)])
                    #     if set_idx.size != 0:
                    #         data_count[nid,0]+=set_idx.size
                    #         cache.insertdata(X[set_idx], F[set_idx], E[set_idx], list(S))

                    for nid,cache in enumerate(cachev):
                        set_idx = np.concatenate([Didx[(Ntrain+bid+nid*int(Nstride)) % Nblocks] for bid in range(Nvalid)])
                        if set_idx.size != 0:
                            data_count[nid, 1] += set_idx.size
                            cache.insertdata(X[set_idx], F[set_idx], E[set_idx], list(S))

                    if build_test:
                        for nid,th5 in enumerate(testh5):
                            set_idx = np.concatenate([Didx[(Ntrain+Nvalid+bid+nid*int(Nstride)) % Nblocks] for bid in range(Ntest)])
                            if set_idx.size != 0:
                                data_count[nid, 2] += set_idx.size
                                th5.store_data(f+data['path'], coordinates=X[set_idx], forces=F[set_idx], energies=E[set_idx], species=list(S))

        # Save train and valid meta file and cleanup testh5
        for t, v in zip(cachet, cachev):
            t.makemetadata()
            v.makemetadata()

        if build_test:
            for th in testh5:
                th.cleanup()

        print(' Train ',' Valid ',' Test ')
        print(data_count)
        print('Training set built.')

    def train_ensemble(self, GPUList, remove_existing=False):
        print('Training Ensemble...')
        processes = []
        indicies = np.array_split(np.arange(self.Nn), len(GPUList))

        for gpu,idc in enumerate(indicies):
            processes.append(Process(target=self.train_network, args=(GPUList[gpu], idc, remove_existing)))
            processes[-1].start()
            #self.train_network(pyncdict, trdict, layers, id, i)

        for p in processes:
            p.join()
        print('Training Complete.')

    def train_network(self, gpuid, indicies, remove_existing=False):
        for index in indicies:
            pyncdict = dict()
            pyncdict['wkdir'] = self.train_root + 'train' + str(index) + '/'
            pyncdict['ntwkStoreDir'] = self.train_root + 'train' + str(index) + '/' + 'networks/'
            pyncdict['datadir'] = self.train_root + "cache-data-" + str(index) + '/'
            pyncdict['gpuid'] = str(gpuid)

            if not os.path.exists(pyncdict['wkdir']):
                os.mkdir(pyncdict['wkdir'])

            if remove_existing:
                shutil.rmtree(pyncdict['ntwkStoreDir'])

            if not os.path.exists(pyncdict['ntwkStoreDir']):
                os.mkdir(pyncdict['ntwkStoreDir'])

            outputfile = pyncdict['wkdir']+'output.opt'

            shutil.copy2(self.netdict['iptfile'], pyncdict['wkdir'])
            shutil.copy2(self.netdict['cnstfile'], pyncdict['wkdir'])
            shutil.copy2(self.netdict['saefile'], pyncdict['wkdir'])

            if "/" in self.netdict['iptfile']:
                nfile = self.netdict['iptfile'].rsplit("/",1)[1]
            else:
                nfile = self.netdict['iptfile']

            command = "cd " + pyncdict['wkdir'] + " && HDAtomNNP-Trainer -i " + nfile + " -d " + pyncdict['datadir'] + " -p 1.0 -m -g " + pyncdict['gpuid'] + " > output.opt"
            proc = subprocess.Popen (command, shell=True)
            proc.communicate()

            print('  -Model',index,'complete')

    def get_train_stats(self):
        #rerr = re.compile('EPOCH\s+?(\d+?)\n[\s\S]+?E \(kcal\/mol\)\s+?(\d+?\.\d+?)\s+?(\d+?\.\d+?)\s+?(\d+?\.\d+?)\n\s+?dE \(kcal\/mol\)\s+?(\d+?\.\d+?)\s+?(\d+?\.\d+?)\s+?(\d+?\.\d+?)\n[\s\S]+?Current best:\s+?(\d+?)\n[\s\S]+?Learning Rate:\s+?(\S+?)\n[\s\S]+?TotalEpoch:\s+([\s\S]+?)\n')
        #rerr = re.compile('EPOCH\s+?(\d+?)\s+?\n[\s\S]+?E \(kcal\/mol\)\s+?(\S+?)\s+?(\S+?)\s+?(\S+?)\n\s+?dE \(kcal\/mol\)\s+?(\S+?)\s+?(\S+?)\s+?(\S+?)\n')
        rblk = re.compile('=+?\n([\s\S]+?=+?\n[\s\S]+?(?:=|Deleting))')
        repo = re.compile('EPOCH\s+?(\d+?)\s+?\n')
        rerr = re.compile('\s+?(\S+?\s+?\(\S+?)\s+?((?:\d|inf)\S*?)\s+?((?:\d|inf)\S*?)\s+?((?:\d|inf)\S*?)\n')
        rtme = re.compile('TotalEpoch:\s+?(\d+?)\s+?dy\.\s+?(\d+?)\s+?hr\.\s+?(\d+?)\s+?mn\.\s+?(\d+?\.\d+?)\s+?sc\.')

        allnets = []
        for index in range(self.Nn):
            print('reading:',self.train_root + 'train' + str(index) + '/' + 'output.opt')
            optfile = open(self.train_root + 'train' + str(index) + '/' + 'output.opt','r').read()
            matches = re.findall(rblk, optfile)

            run = dict({'EPOCH':[],'RTIME':[],'ERROR':dict()})
            for i,data in enumerate(matches):
                run['EPOCH'].append(int(re.search(repo,data).group(1)))

                m = re.search(rtme, data)
                run['RTIME'].append(86400.0*float(m.group(1))+
                                     3600.0*float(m.group(2))+
                                       60.0*float(m.group(3))+
                                            float(m.group(4)))


                err = re.findall(rerr,data)
                for e in err:
                    if e[0] in run['ERROR']:
                        run['ERROR'][e[0]].append(np.array([float(e[1]),float(e[2]),float(e[3])],dtype=np.float64))
                    else:
                        run['ERROR'].update({e[0]:[np.array([float(e[1]), float(e[2]), float(e[3])], dtype=np.float64)]})

            for key in run['ERROR'].keys():
                run['ERROR'][key] = np.vstack(run['ERROR'][key])

            allnets.append(run)
        return allnets
