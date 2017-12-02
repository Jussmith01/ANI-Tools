import hdnntools as hdt
import nmstools as nmt
import pyanitools as pyt
import pyaniasetools as aat
import pyanitrainer as atr
import pymolfrag as pmf

from pyNeuroChem import cachegenerator as cg

import numpy as np

from time import sleep
import subprocess
import random
import pyssh
import re
import os

import pyaniasetools as aat

from multiprocessing import Process
import shutil

class alconformationalsampler():
    def __init__(self, ldtdir, datdir, optlfile, fpatoms, netdict):
        self.ldtdir = ldtdir
        self.datdir = datdir
        self.cdir = ldtdir+datdir+'/confs/'

        self.fpatoms = fpatoms

        self.optlfile = optlfile
        self.idir = [f for f in open(optlfile).read().split('\n') if f != '']

        if not os.path.exists(self.cdir):
            os.mkdir(self.cdir)

        self.netdict = netdict

    def run_sampling_nms(self, nmsparams, gpus=[0]):
        print('Running NMS sampling...')
        p = Process(target=self.normal_mode_sampling, args=(nmsparams['T'],
                                                            nmsparams['Ngen'],
                                                            nmsparams['Nkep'],
                                                            gpus[0]))
        p.start()
        p.join()

    def run_sampling_md(self, mdparams, gpus=[0]):
        md_work = []
        for di, id in enumerate(self.idir):
            files = os.listdir(id)
            for f in files:
                if len(f) > 4:
                    if ".ipt" in f[-4:]:
                        md_work.append(id+f)
                    else:
                        print('Incorrect extension:',id+f)

        md_work = np.array(md_work)
        np.random.shuffle(md_work)
        md_work = np.array_split(md_work,len(gpus))
	
        proc = []
        for i,(md,g) in enumerate(zip(md_work,gpus)):
            proc.append(Process(target=self.mol_dyn_sampling, args=(md,i,
                                                                    mdparams['N'],
                                                                    mdparams['T1'],
                                                                    mdparams['T2'],
                                                                    mdparams['dt'],
                                                                    mdparams['Nc'],
                                                                    mdparams['Ns'],
                                                                    g)))
        print('Running MD Sampling...')
        for i,p in enumerate(proc):
            p.start()

        for p in proc:
            p.join()
        print('Finished sampling.')

    def run_sampling_dimer(self, dmparams, gpus=[0]):

        proc = []
        for i,g in enumerate(gpus):
            proc.append(Process(target=self.dimer_sampling, args=(i, int(dmparams['Nr']/len(gpus)),
                                                                  dmparams,
                                                                  g)))
        print('Running Dimer-MD Sampling...')
        for i,p in enumerate(proc):
            p.start()

        for p in proc:
            p.join()
        print('Finished sampling.')

    def normal_mode_sampling(self, T, Ngen, Nkep, gpuid):
        of = open(self.ldtdir + self.datdir + '/info_data_nms.nfo', 'w')

        aevsize = self.netdict['aevsize']

        anicv = aat.anicrossvalidationconformer(self.netdict['cnstfile'],
                                                self.netdict['saefile'],
                                                self.netdict['nnfprefix'],
                                                self.netdict['num_nets'],
                                                gpuid, False)

        dc = aat.diverseconformers(self.netdict['cnstfile'],
                                   self.netdict['saefile'],
                                   self.netdict['nnfprefix']+'0/networks/',
                                   aevsize,
                                   gpuid, False)

        Nkp = 0
        Nkt = 0
        Ntt = 0
        idx = 0
        for di,id in enumerate(self.idir):
            of.write(str(di)+' of '+str(len(self.idir))+') dir: '+ str(id) +'\n')
            #print(di,'of',len(self.idir),') dir:', id)
            files = os.listdir(id)
            files.sort()

            Nk = 0
            Nt = 0
            for fi,f in enumerate(files):
                data = hdt.read_rcdb_coordsandnm(id+f)

                #print(id+f)
                spc = data["species"]
                xyz = data["coordinates"]
                nmc = data["nmdisplacements"]
                frc = data["forceconstant"]

                nms = nmt.nmsgenerator(xyz,nmc,frc,spc,T,minfc=5.0E-2)
                conformers = nms.get_Nrandom_structures(Ngen)

                ids = dc.get_divconfs_ids(conformers, spc, Ngen, Nkep, [])
                conformers = conformers[ids]
                #print('    -',f,len(ids),conformers.shape)

                sigma = anicv.compute_stddev_conformations(conformers,spc)
                #print(sigma)
                sid = np.where( sigma >  0.25 )[0]
                #print(sid)
                #print('  -', fi, 'of', len(files), ') File:', f, 'keep:', sid.size,'percent:',"{:.2f}".format(100.0*sid.size/Ngen))


                Nt += sigma.size
                Nk += sid.size
                if 100.0*sid.size/float(Ngen) > 0:
                    Nkp += sid.size
                    cfn = f.split('.')[0].split('-')[0]+'_'+str(idx).zfill(5)+'-'+f.split('.')[0].split('-')[1]+'_2.xyz'
                    hdt.writexyzfile(self.cdir+cfn,conformers[sid],spc)
                idx += 1

            Nkt += Nk
            Ntt += Nt
            of.write('    -Total: '+str(Nk)+' of '+str(Nt)+' percent: '+"{:.2f}".format(100.0*Nk/Nt)+'\n')
            of.flush()
            #print('    -Total:',Nk,'of',Nt,'percent:',"{:.2f}".format(100.0*Nk/Nt))

        del anicv
        del dc

        of.write('\nGrand Total: '+ str(Nkt)+ ' of '+ str(Ntt)+' percent: '+"{:.2f}".format(100.0*Nkt/Ntt)+ ' Kept '+str(Nkp)+'\n')
        #print('\nGrand Total:', Nkt, 'of', Ntt,'percent:',"{:.2f}".format(100.0*Nkt/Ntt), 'Kept',Nkp)
        of.close()

    def mol_dyn_sampling(self,md_work, i, N, T1, T2, dt, Nc, Ns, gpuid):
        activ = aat.moldynactivelearning(self.netdict['cnstfile'],
                                         self.netdict['saefile'],
                                         self.netdict['nnfprefix'],
                                         self.netdict['num_nets'],
                                         gpuid)

        difo = open(self.ldtdir + self.datdir + '/info_data_mdso-'+str(i)+'.nfo', 'w')
        Nmol = 0
        dnfo = 'MD Sampler running: ' + str(md_work.size)
        difo.write(dnfo + '\n')
        Nmol = md_work.size
        ftme_t = 0.0
        for di, id in enumerate(md_work):
            data = hdt.read_rcdb_coordsandnm(id)
            print(di, ') Working on', id, '...')
            S = data["species"]

            # Set mols
            activ.setmol(data["coordinates"], S)

            # Generate conformations
            X = activ.generate_conformations(N, T1, T2, dt, Nc, Ns, dS=0.25)

            ftme_t += activ.failtime

            nfo = activ._infostr_
            m = id.rsplit('/',1)[1].rsplit('.',1)[0]
            difo.write('  -' + m + ': ' + nfo + '\n')
            difo.flush()
            #print(nfo)

            if X.size > 0:
                hdt.writexyzfile(self.cdir + 'mds_' + m.split('.')[0] + '_' + str(i).zfill(2) + str(di).zfill(4) + '.xyz', X, S)
        difo.write('Complete mean fail time: ' + "{:.2f}".format(ftme_t / float(Nmol)) + '\n')
        print(Nmol)
        del activ
        difo.close()

    def dimer_sampling(self, tid, Nr, dparam, gpuid):
        mds_select = dparam['mdselect']
        N = dparam['N']
        T = dparam['T']
        L = dparam['L']
        V = dparam['V']
        dt = dparam['dt']
        Nm = dparam['Nm']

        Ni = dparam['Ni']
        
        mols = []
        difo = open(self.ldtdir + self.datdir + '/info_data_mddimer-'+str(tid)+'.nfo', 'w')
        for di,id in enumerate(dparam['mdselect']):
            files = os.listdir(self.idir[id[1]])
            random.shuffle(files)

            dnfo = str(di) + ' of ' + str(len(dparam['mdselect'])) + ') dir: ' + str(self.idir[id[1]]) + ' Selecting: '+str(id[0]*len(files))
            #print(dnfo)
            difo.write(dnfo+'\n')
        
            for i in range(id[0]):
                for n,m in enumerate(files):
                        data = hdt.read_rcdb_coordsandnm(self.idir[id[1]]+m)
                        mols.append(data)

        dgen = pmf.dimergenerator(self.netdict['cnstfile'], 
                                  self.netdict['saefile'], 
                                  self.netdict['nnfprefix'], 
                                  self.netdict['num_nets'], 
                                  mols, gpuid)

        difo.write('Beginning dimer generation...\n')
        
        Nt = 0
        Nd = 0
        for i in range(Nr):
            dgen.init_dynamics(Nm, V, L, dt, T)
 
            fname = self.cdir + 'dimer-'+str(tid).zfill(2)+str(i).zfill(2)+'_'
       
            dgen.run_dynamics(Ni)
            dgen.__fragmentbox__(fname)
        
            Nt += dgen.Nt
            Nd += dgen.Nd
        
            #print('Step (',tid,',',i,') [', str(dgen.Nd), '/', str(dgen.Nt),'] generated ',len(dgen.frag_list), 'dimers...')
            difo.write('Step ('+str(i)+') ['+ str(dgen.Nd)+ '/'+ str(dgen.Nt)+'] generated '+str(len(dgen.frag_list))+'dimers...\n')

        difo.write('Generated '+str(Nd)+' of '+str(Nt)+' tested dimers. Percent: ' + "{:.2f}".format(100.0*Nd/float(Nt)))
        difo.close()

def interval(v,S):
    ps = 0.0
    ds = 1.0 / float(S)
    for s in range(S):
        if v > ps and v <= ps+ds:
            return s
        ps = ps + ds

class alaniensembletrainer():
    def __init__(self, train_root, netdict, train_pref, h5dir, Nn):
        self.train_root = train_root
        self.train_pref = train_pref
        self.h5dir = h5dir
        self.Nn = Nn
        self.netdict = netdict

        self.h5file = [f for f in os.listdir(self.h5dir) if f.rsplit('.',1)[1] == 'h5']
        #print(self.h5dir,self.h5file)

    def build_training_cache(self):
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
                F = data['forces']
                S = data['species']

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

                Ndc += E.size

                if (set(S).issubset(['C', 'N', 'O', 'H', 'F', 'S', 'Cl'])):

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
                        tv_split = np.array_split(split[i], 2)

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

    def train_ensemble(self, GPUList):
        print('Training Ensemble...')
        processes = []
        for i,id in enumerate(GPUList):
            processes.append(Process(target=self.train_network, args=(id, i)))
            processes[-1].start()
            #self.train_network(pyncdict, trdict, layers, id, i)

        for p in processes:
            p.join()
        print('Training Complete.')

    def train_network(self, gpuid, index):
        pyncdict = dict()
        pyncdict['wkdir'] = self.train_root + 'train' + str(index) + '/'
        pyncdict['ntwkStoreDir'] = self.train_root + 'train' + str(index) + '/' + 'networks/'
        pyncdict['datadir'] = self.train_root + "cache-data-" + str(index) + '/'
        pyncdict['gpuid'] = str(gpuid)

        if not os.path.exists(pyncdict['wkdir']):
            os.mkdir(pyncdict['wkdir'])

        if not os.path.exists(pyncdict['ntwkStoreDir']):
            os.mkdir(pyncdict['ntwkStoreDir'])

        outputfile = pyncdict['wkdir']+'output.opt'

        shutil.copy2(self.netdict['iptfile'], pyncdict['wkdir'])
        shutil.copy2(self.netdict['cnstfile'], pyncdict['wkdir'])
        shutil.copy2(self.netdict['saefile'], pyncdict['wkdir'])

        command = "cd " + pyncdict['wkdir'] + " && HDAtomNNP-Trainer -i inputtrain.ipt -d " + pyncdict['datadir'] + " -p 1.0 -m -g " + pyncdict['gpuid'] + " > output.opt"
        proc = subprocess.Popen (command, shell=True)
        proc.communicate()

#    def train_ensemble(self, GPUList, pyncdict, trdict, layers):
#        print('Training Ensemble...')
#        processes = []
#        for i,id in enumerate(GPUList):
#            processes.append(Process(target=self.train_network, args=(pyncdict, trdict, layers, id, i)))
#            processes[-1].start()
#            #self.train_network(pyncdict, trdict, layers, id, i)
#
#        for p in processes:
#            p.join()
#        print('Training Complete.')

#    def train_network(self, pyncdict, trdict, layers, gpuid, index):
#        pyncdict['wkdir'] = self.train_root + 'train' + str(index) + '/'
#        pyncdict['ntwkStoreDir'] = self.train_root + 'train' + str(index) + '/' + 'networks/'
#        pyncdict['datadir'] = self.train_root + "cache-data-" + str(index) + '/'
#        pyncdict['gpuid'] = str(gpuid)
#
#        if not os.path.exists(pyncdict['wkdir']):
#            os.mkdir(pyncdict['wkdir'])
#
#        if not os.path.exists(pyncdict['ntwkStoreDir']):
#            os.mkdir(pyncdict['ntwkStoreDir'])
#
#        outputfile = pyncdict['wkdir']+'output.opt'
#
#        # Setup trainer
#        tr = atr.anitrainer(pyncdict, layers)
#
#        # Train network
#        tr.train_network(trdict['learningrate'],
#                         trdict['lrannealing'],
#                         trdict['lrconvergence'],
#                         trdict['ST'],
#                         outputfile,
#                         trdict['printstep'])
