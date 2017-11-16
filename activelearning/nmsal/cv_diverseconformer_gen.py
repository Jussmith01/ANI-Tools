import hdnntools as hdn
import nmstools as nmt
import numpy as np
import os

import random

# pyneurochem
import pyNeuroChem as pync
import pyaniasetools as aat

#wkdir    = '/home/jujuman/Dropbox/ChemSciencePaper.AER/networks/ANI-c08f-ntwk/'
#cnstfile = wkdir + 'rHCNO-4.6A_16-3.1A_a4-8.params'
#saefile  = wkdir + 'sae_6-31gd.dat'
#nnfdir   = wkdir + 'networks/'

wkdir    = '/home/jujuman/Research/DataReductionMethods/modelCNOSFCl/ANI-AL-0605/ANI-AL-0605.0000/cv7/'
cnstfile = wkdir + 'rHCNOSFCl-4.6A_16-3.1A_a4-8.params'
saefile  = wkdir + 'sae_wb97x-631gd.dat'
nnfdir   = wkdir + 'train0/networks/'


# reaction center atoms
fpatoms = ['C', 'N', 'O', 'S', 'F', 'Cl']
aevsize = 1008
T = 1000.0
Ngen = 400
Nkep = 100
atmlist = []

idir = [#'/home/jujuman/Research/GDB-11-AL-wB97x631gd/chemmbl22/config_2/inputs/',
        #'/home/jujuman/Research/GDB-11-AL-wB97x631gd/chemmbl22/config_1/inputs/',
	    #'/home/jujuman/Research/GDB-11-AL-wB97x631gd/h2o_cluster/inputs/',
        #'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s06/config_1/inputs/',
        #'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s06/config_2/inputs/',
        #'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s06/config_3/inputs/',
        #'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s07/config_1/inputs/',
        #'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s07/config_2/inputs/',
        #'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s07/config_3/inputs/',
        #'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s07/config_4/inputs/',
        #'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s08/config_1/inputs/',
        #'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s08/config_2/inputs/',
        #'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s08/config_3/inputs/',
        #'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s08/config_4/inputs/',
        #'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s09/config_1/inputs/',
        #'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s01/inputs/',
        #'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s02/inputs/',
        #'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s03/inputs/',
        #'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s04/inputs/',
        #'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s05/inputs/',
        #'/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_red/dnntsgdb11_03_red/inputs/',
        #'/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_red/dnntsgdb11_04_red/inputs/',
        #'/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_red/dnntsgdb11_05_red/inputs/',
        #'/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_red/dnntsgdb11_06_red/inputs/',
        '/home/jujuman/Research/GDB-11-AL-wB97x631gd/elements_SFCl/ANI-AL-SFCl/ANI-AL-0605/ANI-AL-0605.0001/config/inputs/',
        '/home/jujuman/Research/GDB-11-AL-wB97x631gd/elements_SFCl/gdb11_size1/inputs/',
        '/home/jujuman/Research/GDB-11-AL-wB97x631gd/elements_SFCl/gdb11_size2/inputs/',
        '/home/jujuman/Research/GDB-11-AL-wB97x631gd/elements_SFCl/gdb11_size3/inputs/',
        '/home/jujuman/Research/GDB-11-AL-wB97x631gd/elements_SFCl/gdb11_size4/inputs/',
        '/home/jujuman/Research/GDB-11-AL-wB97x631gd/elements_SFCl/gdb11_size5/inputs/',
        ]

#cdir = '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb09_1/confs_5/'
cdir = '/home/jujuman/Research/GDB-11-AL-wB97x631gd/elements_SFCl/ANI-AL-SFCl/ANI-AL-0605/ANI-AL-0605.0001/confs_1/'

dc = aat.diverseconformers(cnstfile, saefile, nnfdir, aevsize, 0, False)

#wkdircv = '/home/jujuman/Research/DataReductionMethods/model6r/model3-5/cv1/cv32/'
#cnstfilecv = wkdircv + '../rHCNO-4.6A_16-3.1A_a4-8.params'
wkdircv = '/home/jujuman/Research/DataReductionMethods/modelCNOSFCl/ANI-AL-0605/ANI-AL-0605.0000/cv7/'
cnstfilecv = wkdircv + 'rHCNOSFCl-4.6A_16-3.1A_a4-8.params'
saefilecv  = wkdircv + 'sae_wb97x-631gd.dat'
nnfprefix   = wkdircv + 'train'

anicv = aat.anicrossvalidationconformer(cnstfilecv,saefilecv,nnfprefix,5,0,False)

if not os.path.exists(cdir):
    os.mkdir(cdir)

of = open(cdir+'info_data_nms.nfo', 'w')

Nkp = 0
Nkt = 0
Ntt = 0
idx = 0
for di,id in enumerate(idir):
    of.write(str(di)+' of '+str(len(idir))+') dir: '+ str(id) +'\n')
    print(di,'of',len(idir),') dir:', id)
    files = os.listdir(id)
    files.sort()

    Nk = 0
    Nt = 0
    for fi,f in enumerate(files):
        data = hdn.read_rcdb_coordsandnm(id+f)

        #print(id+f)
        spc = data["species"]
        xyz = data["coordinates"]
        nmc = data["nmdisplacements"]
        frc = data["forceconstant"]

        print(spc)

        nms = nmt.nmsgenerator(xyz,nmc,frc,spc,T,minfc=5.0E-2)
        conformers = nms.get_Nrandom_structures(Ngen)

        ids = dc.get_divconfs_ids(conformers, spc, Ngen, Nkep, atmlist)
        conformers = conformers[ids]
        print('    -',f,len(ids),conformers.shape)

        sigma = anicv.compute_stddev_conformations(conformers,spc)
        #print(sigma)
        sid = np.where( sigma >  0.3 )[0]
        print('  -', fi, 'of', len(files), ') File:', f, 'keep:', sid.size,'percent:',"{:.2f}".format(100.0*sid.size/Ngen))


        Nt += sigma.size
        Nk += sid.size
        if 100.0*sid.size/float(Ngen) > 0:
            Nkp += sid.size
            cfn = f.split('.')[0].split('-')[0]+'_'+str(idx).zfill(5)+'-'+f.split('.')[0].split('-')[1]+'_2.xyz'
            hdn.writexyzfile(cdir+cfn,conformers[sid],spc)
        idx += 1

    Nkt += Nk
    Ntt += Nt
    of.write('    -Total: '+str(Nk)+' of '+str(Nt)+' percent: '+"{:.2f}".format(100.0*Nk/Nt)+'\n')
    of.flush()
    print('    -Total:',Nk,'of',Nt,'percent:',"{:.2f}".format(100.0*Nk/Nt))

of.write('\nGrand Total: '+ str(Nkt)+ ' of '+ str(Ntt)+' percent: '+"{:.2f}".format(100.0*Nkt/Ntt)+ ' Kept '+str(Nkp)+'\n')
print('\nGrand Total:', Nkt, 'of', Ntt,'percent:',"{:.2f}".format(100.0*Nkt/Ntt), 'Kept',Nkp)
of.close()
