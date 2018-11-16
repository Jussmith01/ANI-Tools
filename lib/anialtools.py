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
#import pyssh
import re
import os

import pyaniasetools as aat

from multiprocessing import Process
import shutil

import matplotlib.pyplot as plt

class alconformationalsampler():

    # Constructor
    def __init__(self, ldtdir, datdir, optlfile, fpatoms, netdict):
        self.ldtdir = ldtdir # local working dir
        self.datdir = datdir # working data dir
        self.cdir = ldtdir+datdir+'/confs/' # confs store dir (the data gen code looks here for conformations to run QM on)

        self.fpatoms = fpatoms # atomic species being sampled

        self.optlfile = optlfile # Optimized molecules store path file

        self.idir = [f for f in open(optlfile).read().split('\n') if f != ''] # read and store the paths to the opt files

        # create cdir if it does not exist
        if not os.path.exists(self.cdir):
            os.mkdir(self.cdir)

        # store network parameters dictionary
        self.netdict = netdict


    # Runs NMS sampling (single GPU only currently)
    def run_sampling_nms(self, nmsparams, gpus=[0]):
        print('Running NMS sampling...')
        p = Process(target=self.normal_mode_sampling, args=(nmsparams['T'],
                                                            nmsparams['Ngen'],
                                                            nmsparams['Nkep'],
                                                            nmsparams['maxd'],
                                                            nmsparams['sig'],
                                                            gpus[0]))
        p.start()
        p.join()

    # Run MD sampling on N GPUs. This code will automatically run 2 mds per GPU for better utilization
    def run_sampling_md(self, mdparams, perc=1.0, gpus=[0]):
        md_work = []
        for di, id in enumerate(self.idir):
            files = os.listdir(id)
            for f in files:
                if len(f) > 4:
                    if ".ipt" in f[-4:]:
                        md_work.append(id+f)
                    else:
                        print('Incorrect extension:',id+f)

        gpus2 = gpus

        md_work = np.array(md_work)
        np.random.shuffle(md_work)
        md_work = md_work[0:int(perc*md_work.size)]
        md_work = np.array_split(md_work,len(gpus2))
	
        proc = []
        for i,(md,g) in enumerate(zip(md_work,gpus2)):
            proc.append(Process(target=self.mol_dyn_sampling, args=(md,i,
                                                                    mdparams['N'],
                                                                    mdparams['T1'],
                                                                    mdparams['T2'],
                                                                    mdparams['dt'],
                                                                    mdparams['Nc'],
                                                                    mdparams['Ns'],
                                                                    mdparams['sig'],
                                                                    g)))
        print('Running MD Sampling...')
        for i,p in enumerate(proc):
            p.start()

        for p in proc:
            p.join()
        print('Finished sampling.')

    # Run the dimer sampling code on N gpus
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

    # Run cluster sampling on N gpus
    def run_sampling_cluster(self, gcmddict, gpus=[0]):

        Nmols = np.random.randint(low=gcmddict['MolLow'],
                                  high=gcmddict['MolHigh'],
                                  size=gcmddict['Nr'])

        Ntmps = np.random.randint(low=gcmddict['T']-200,
                                  high=gcmddict['T']+200,
                                  size=gcmddict['Nr'])

        print('Box Sizes:',Nmols)
        print('Sim Temps:',Ntmps)

        seeds = np.random.randint(0,1000000,len(gpus))

        mol_sizes = np.array_split(Nmols, len(gpus))
        mol_temps = np.array_split(Ntmps, len(gpus))

        proc = []
        for i,g in enumerate(gpus):
            proc.append(Process(target=self.cluster_sampling, args=(i, int(gcmddict['Nr']/len(gpus)),
                                                                    mol_sizes[i],
                                                                    mol_temps[i],
                                                                    seeds[i],
                                                                    gcmddict,
                                                                    g)))
        print('Running Cluster-MD Sampling...')
        for i,p in enumerate(proc):
            p.start()

        for p in proc:
            p.join()
        print('Finished sampling.')

    # Normal mode sampler function
    def normal_mode_sampling(self, T, Ngen, Nkep, maxd, sig, gpuid):
        of = open(self.ldtdir + self.datdir + '/info_data_nms.nfo', 'w')

        aevsize = self.netdict['aevsize']

        anicv = aat.anicrossvalidationconformer(self.netdict['cnstfile'],
                                                self.netdict['saefile'],
                                                self.netdict['nnfprefix'],
                                                self.netdict['num_nets'],
                                                [gpuid], False)

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
                print(f)
                data = hdt.read_rcdb_coordsandnm(id+f)

                #print(id+f)
                spc = data["species"]
                xyz = data["coordinates"]
                nmc = data["nmdisplacements"]
                frc = data["forceconstant"]

                if "charge" in data and "multip" in data:
                    chg = data["charge"]
                    mlt = data["multip"]
                else:
                    chg = "0"
                    mlt = "1"

                nms = nmt.nmsgenerator(xyz,nmc,frc,spc,T,minfc=5.0E-2,maxd=maxd)
                conformers = nms.get_Nrandom_structures(Ngen)

                if conformers.shape[0] > 0:
                    if conformers.shape[0] > Nkep:
                        ids = dc.get_divconfs_ids(conformers, spc, Ngen, Nkep, [])
                        conformers = conformers[ids]

                    sigma = anicv.compute_stddev_conformations(conformers,spc)
                    sid = np.where( sigma >  sig )[0]


                    Nt += sigma.size
                    Nk += sid.size
                    if 100.0*sid.size/float(Ngen) > 0:
                        Nkp += sid.size
                        cfn = f.split('.')[0].split('-')[0]+'_'+str(idx).zfill(5)+'-'+f.split('.')[0].split('-')[1]+'_2.xyz'
                        cmts = [' '+chg+' '+mlt for c in range(Nk)]
                        hdt.writexyzfilewc(self.cdir+cfn,conformers[sid],spc,cmts)
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

    # MD Sampling function
    def mol_dyn_sampling(self,md_work, i, N, T1, T2, dt, Nc, Ns, sig, gpuid):
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
            #print(di, ') Working on', id, '...')
            S = data["species"]

            # Set mols
            activ.setmol(data["coordinates"], S)

            # Generate conformations
            X = activ.generate_conformations(N, T1, T2, dt, Nc, Ns, dS=sig)

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

    # Dimer sampling function
    def dimer_sampling(self, tid, Nr, dparam, gpuid):
        mds_select = dparam['mdselect']
        #N = dparam['N']
        T = dparam['T']
        L = dparam['L']
        V = dparam['V']
        maxNa = dparam['maxNa']
        dt = dparam['dt']
        sig = dparam['sig']
        Nm = dparam['Nm']
        Ni = dparam['Ni']
        Ns = dparam['Ns']
        
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
                        if len(data['species']) < maxNa:
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
 
            for j in range(Ns):
                if j != 0:
                    dgen.run_dynamics(Ni)

                fname = self.cdir + 'dimer-'+str(tid).zfill(2)+str(i).zfill(2)+'-'+str(j).zfill(2)+'_'
                max_sig = dgen.__fragmentbox__(fname,sig)
                print('MaxSig:',max_sig)
                #difo.write('Step ('+str(i)+',',+str(j)+') ['+ str(dgen.Nd)+ '/'+ str(dgen.Nt)+']\n')
                difo.write('Step ('+str(i)+','+str(j)+') ['+ str(dgen.Nd)+ '/'+ str(dgen.Nt)+'] max sigma: ' + "{:.2f}".format(max_sig) + ' generated '+str(len(dgen.frag_list))+' dimers...\n')

                Nt += dgen.Nt
                Nd += dgen.Nd
        
                #print('Step (',tid,',',i,') [', str(dgen.Nd), '/', str(dgen.Nt),'] generated ',len(dgen.frag_list), 'dimers...')
                #difo.write('Step ('+str(i)+') ['+ str(dgen.Nd)+ '/'+ str(dgen.Nt)+'] generated '+str(len(dgen.frag_list))+'dimers...\n')
                if max_sig > 3.0*sig:
                    difo.write('Terminating dynamics -- max sigma: '+"{:.2f}".format(max_sig)+' Ran for: '+"{:.2f}".format(j*Ni*dt)+'fs\n')
                    break

        difo.write('Generated '+str(Nd)+' of '+str(Nt)+' tested dimers. Percent: ' + "{:.2f}".format(100.0*Nd/float(Nt)))
        difo.close()

    # Cluster sampling function
    def cluster_sampling(self, tid, Nr, mol_sizes, mol_temps, seed, gcmddict, gpuid):
        os.environ["OMP_NUM_THREADS"] = "2"

        dictc = gcmddict.copy()
        solv_file = dictc['solv_file']
        solu_dirs = dictc['solu_dirs']

        np.random.seed(seed)

        dictc['Nr'] = Nr
        dictc['molfile'] = self.cdir + 'clst'
        dictc['dstore'] = self.ldtdir + self.datdir + '/'

        solv = [hdt.read_rcdb_coordsandnm(solv_file)]

        if solu_dirs:
            solu = [hdt.read_rcdb_coordsandnm(solu_dirs+f) for f in os.listdir(solu_dirs)]
        else:
            solu = []

        dgen = pmf.clustergenerator(self.netdict['cnstfile'],
                                    self.netdict['saefile'],
                                    self.netdict['nnfprefix'],
                                    self.netdict['num_nets'],
                                    solv, solu, gpuid)

        dgen.generate_clusters(dictc, mol_sizes, mol_temps, tid)


    # Run the TS sampler
    def run_sampling_TS(self, tsparams, gpus=[0], perc=1.0):
        TS_infiles = []
        for di, id in enumerate(tsparams['tsfiles']):
            files = [fl for fl in os.listdir(id) if '.xyz' in fl]
            for f in files:
                TS_infiles.append(id+f)

        gpus2 = gpus

        TS_infiles = np.array(TS_infiles)
        np.random.shuffle(TS_infiles)
        TS_infiles = TS_infiles[0:int(perc*len(TS_infiles))]
        TS_infiles = np.array_split(TS_infiles,len(gpus2))
	
        proc = []
        for i,g in enumerate(gpus2):
            proc.append(Process(target=self.TS_sampling, args=(i, TS_infiles[i], tsparams, g)))
        print('Running MD Sampling...')
        for p in proc:
            p.start()

        for p in proc:
            p.join()

        print('Finished sampling.')

    # TS sampler function
    def TS_sampling(self, tid, TS_infiles, tsparams, gpuid):
        activ = aat.MD_Sampler(TS_infiles,
                               self.netdict['cnstfile'],
                               self.netdict['saefile'],
                               self.netdict['nnfprefix'],
                               self.netdict['num_nets'],
                               gpuid)
        T=tsparams['T']
        sig=tsparams['sig']
        Ns=tsparams['n_samples']
        n_steps=tsparams['n_steps']
        steps=tsparams['steps']
        min_steps=tsparams['min_steps']
        nm=tsparams['normalmode']
        displacement=tsparams['displacement']
        difo = open(self.ldtdir + self.datdir + '/info_tssampler-'+str(tid)+'.nfo', 'w')
        for f in TS_infiles:
            X = []
            ftme_t = 0.0
            fail_count=0
            sumsig = 0.0
            for i in range(Ns):
                #x, S, t, stddev, fail, temp = activ.run_md(f, T, steps, n_steps, nmfile=f.rsplit(".",1)[0]+'.log', displacement=displacement, min_steps=min_steps, sig=sig, nm=nm)
                x, S, t, stddev, fail, temp = activ.run_md(f, T, steps, n_steps, min_steps=min_steps, sig=sig, nm=nm)
                sumsig += stddev
                if fail:
                    #print('Job '+str(i)+' failed in '+"{:.2f}".format(t)+' Sigma: ' + "{:.2f}".format(stddev)+' SetTemp: '+"{:.2f}".format(temp))
                    difo.write('Job '+str(i)+' failed in '+"{:.2f}".format(t)+'fs Sigma: ' + "{:.2f}".format(stddev) + ' SetTemp: ' + "{:.2f}".format(temp) + '\n')
                    X.append(x[np.newaxis,:,:])
                    fail_count+=1
                else:
                    #print('Job '+str(i)+' succeeded.')
                    difo.write('Job '+str(i)+' succeeded.\n')
                ftme_t += t
            print('Complete mean fail time: ' + "{:.2f}".format(ftme_t / float(Ns)) + ' failed ' + str(fail_count) + '/' + str(Ns) + '\n')
            difo.write('Complete mean fail time: ' + "{:.2f}".format(ftme_t / float(Ns)) + ' failed ' + str(fail_count) + '/' + str(Ns) + ' MeanSig: ' + "{:.2f}".format(sumsig / float(Ns)) + '\n')
            X = np.vstack(X)
            hdt.writexyzfile(self.cdir + os.path.basename(f), X, S)
        del activ
        difo.close()

    # Run the dihedral sampler
    def run_sampling_dhl(self, dhlparams, gpus):
        dhlparams['Nmol'] = int(np.ceil(dhlparams['Nmol']/len(gpus)))

        seeds = np.random.randint(0,1000000,len(gpus))

        proc = []
        for i,g in enumerate(gpus):
            proc.append(Process(target=self.DHL_sampling, args=(i, dhlparams, self.fpatoms, g, seeds[i])))

        print('Running DHL Sampling...')
        for p in proc:
            p.start()

        for p in proc:
            p.join()

        print('Finished sampling.')

    # Dihedral sampling function
    def DHL_sampling(self, i, dhlparams, fpatoms, gpuid, seed):
        activ = aat.aniTortionSampler(self.netdict,
                                      self.cdir,
                                      dhlparams['smilefile'],
                                      dhlparams['Nmol'],
                                      dhlparams['Nsamp'],
                                      dhlparams['sig'],
                                      dhlparams['rng'],
                                      fpatoms,
                                      seed,
                                      gpuid)

        activ.generate_dhl_samples(MaxNa=dhlparams['MaxNa'], fpref='dhl_scan-'+str(i).zfill(2), freqname='vib'+str(i)+'.')

    def run_sampling_pDynTS(self, pdynparams, gpus=0):

        gpus2 = gpus
        proc = []
        for g in enumerate(gpus2):
            proc.append(Process(target=self.pDyn_QMsampling, args=(pdynparams, g)))
        print('Running pDynamo Sampling...')
        for p in proc:
            p.start()

        for p in proc:
            p.join()

        print('Finished pDynamo sampling.')


    def pDyn_QMsampling(self, pdynparams, gpuid):       
                                                                  #Call subproc_pDyn class in pyaniasetools as activ
        activ = aat.subproc_pDyn(self.netdict['cnstfile'],
                                         self.netdict['saefile'],
                                         self.netdict['nnfprefix'],
                                         self.netdict['num_nets'],
                                         gpuid)
        pDyn_dir=pdynparams['pDyn_dir']                         #Folder to write pDynamo input file
        num_rxn=pdynparams['num_rxn']                           #Number of input rxn
        logfile_OPT=pdynparams['logfile_OPT']                   #logfile for FIRE OPT output
        logfile_TS=pdynparams['logfile_TS']                     #logfile for ANI TS output
        logfile_IRC=pdynparams['logfile_IRC']                   #logfile for ANI IRC output
        sbproc_cmdOPT=pdynparams['sbproc_cmdOPT']               #Subprocess commands to run pDyanmo
        sbproc_cmdTS=pdynparams['sbproc_cmdTS']
        sbproc_cmdIRC=pdynparams['sbproc_cmdIRC']
        IRCdir=pdynparams['IRCdir']                             #path to get pDynamo saved IRC points
        indir=pdynparams['indir']                               #path to save XYZ files of IRC points to check stddev
        XYZfile=pdynparams['XYZfile']                           #XYZ file with high standard deviations structures
        l_val=pdynparams['l_val']                               #Ri --> randomly perturb in the interval [+x,-x]
        h_val=pdynparams['h_val']                               
        n_points=pdynparams['n_points']                         #Number of points along IRC (forward+backward+1 for TS)
        sig=pdynparams['sig']
        N=pdynparams['N']
        wkdir=pdynparams['wkdir']
        cnstfilecv=pdynparams['cnstfilecv']
        saefilecv=pdynparams['saefilecv']
        Nnt=pdynparams['Nnt']

        # --------------------------------- Run pDynamo ---------------------------
        # auto-TS ---> FIRE constraint OPT of core C atoms ---> ANI TS ---> ANI IRC

        activ.write_pDynOPT(num_rxn, pDyn_dir, wkdir, cnstfilecv, saefilecv, Nnt)                              #Write pDynamo input file in pDyndir
        activ.write_pDynTS(num_rxn, pDyn_dir, wkdir, cnstfilecv, saefilecv, Nnt)
        activ.write_pDynIRC(num_rxn, pDyn_dir, wkdir, cnstfilecv, saefilecv, Nnt)

        chk_OPT = activ.subprocess_cmd(sbproc_cmdOPT, False, logfile_OPT)
        if chk_OPT == 0:                                                                                  #Wait until previous subproc is done!!
            chk_TS = activ.subprocess_cmd(sbproc_cmdTS, False, logfile_TS)
            if chk_TS == 0:
                chk_IRC = activ.subprocess_cmd(sbproc_cmdIRC, False, logfile_IRC)
                
        # ----------------------- Save points along ANI IRC ------------------------
        IRCfils=os.listdir(IRCdir)
        IRCfils.sort()

        for f in IRCfils:
            activ.getIRCpoints_toXYZ(n_points, IRCdir+f, f, indir)
        infils=os.listdir(indir)
        infils.sort()
        
        # ------ Check for high standard deviation structures and get vib modes -----
        for f in infils:
            stdev = activ.check_stddev(indir+f, sig)
            if stdev > sig:                      #if stddev is high then get modes for that point
                nmc = activ.get_nm(indir+f)      #save modes in memory for later use
            
            activ.write_nm_xyz(XYZfile)          #writes all the structures with high standard deviations to xyz file

        # ----------------------------- Read XYZ for NM -----------------------------      
        X, spc, Na, C = hdt.readxyz3(XYZfile)

        # --------- NMS for XYZs with high stddev --------
        for i in range(len(X)):
            for j in range (len(nmc)):
                gen = nmt.nmsgenerator_RXN(X[i],nmc[j],spc[i],l_val,h_val)      # xyz,nmo,fcc,spc,T,Ri_-x,Ri_+x,minfc = 1.0E-3

                N = 500
                gen_crd = np.zeros((N, len(spc[i]),3),dtype=np.float32)
                for k in range(N):
                    gen_crd[k] = gen.get_random_structure()

                hdt.writexyzfile(self.cdir + 'nms_TS%i.xyz' %N, gen_crd, spc[i])
                
        del activ


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

    def build_strided_training_cache(self,Nblocks,Nvalid,Ntest,build_test=True, build_valid=False, forces=True, grad=False, Fkey='forces', forces_unit=1.0, Ekey='energies', energy_unit=1.0, Eax0sum=False, rmhighe=True):
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

        if build_valid:
            valdh5 = [pyt.datapacker(store_dir + str(r) + '/../testset/valdset' + str(r) + '.h5') for r in range(N)]

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
                            if build_valid:
                                valdh5[nid].store_data(f+data['path'], coordinates=X[set_idx], forces=F[set_idx], energies=E[set_idx], species=list(S))


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

        if build_valid:
            for vh in valdh5:
                vh.cleanup()

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
