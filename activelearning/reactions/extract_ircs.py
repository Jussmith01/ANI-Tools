import hdnntools as hdt
import pygau09tools as pyg
import os
import matplotlib.pyplot as plt
import pyNeuroChem as pync
import numpy as np

os.environ["PYTHONPATH"] = "../../lib"

d = '/home/jujuman/Dropbox/IRC_DBondMig/auto-TS/auto-TS1/IRC/'
c = '/home/jujuman/Dropbox/IRC_DBondMig/auto-TS/auto-TS1/XYZ/'
r = '/home/jujuman/Dropbox/IRC_DBondMig/auto-TS/auto-TS1/DataGen/'
fp = 'DA_IRC'

#wkdir1 = '/home/jujuman/Dropbox/ChemSciencePaper.AER/networks/ANI-c08f-ntwk-cv/'
#cnstfile = 'rHCNO-4.6A_16-3.1A_a4-8.params'
#saefile = 'sae_6-31gd.dat'

wkdir1 = '/home/jujuman/Research/ReactionGeneration/cv_single2/'
cnstfile = 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefile = 'sae_6-31gd.dat'

Nnc = 3

files = os.listdir(d)
files.sort()

# Construct pyNeuroChem classes
nc1 =  [pync.conformers(wkdir1 + cnstfile, wkdir1 + saefile, wkdir1 + 'cv_train_' + str(l) + '/networks/', 0, True) for l in range(Nnc)]

comp_xyz = []

for f in files:
    Eact, xyz, spc, Rc = pyg.read_irc(d+f)
    s_idx = f.split('IRC')[1].split('.')[0]
    hdt.writexyzfile(c+f.split('.')[0]+'.xyz',xyz,spc)
    #print(f.split('IRC')[1].split('.')[0],Rc.shape)
    if Rc.size > 10:
        #------------ CV NETWORKS 1 -----------
        energies = []
        N = 0
        for comp in nc1:
            comp.setConformers(confs=xyz, types=list(spc))
            energies.append(hdt.hatokcal*comp.energy())
            N = N + 1

        energies = np.vstack(energies)
        modl_std = np.std(energies,axis=0) / float(len(spc))

        bad_cnt = 0
        bad_xyz = []
        bad_idx = []
        for i,(X,s) in enumerate(zip(xyz,modl_std)):
            if s > 0.05:
                bad_cnt = bad_cnt + 1
                if i%3 == 0:
                    bad_xyz.append(X)
                    bad_idx.append(i)

        #for j,(X,i) in enumerate(zip(bad_xyz,bad_idx)):
        #    idx = s_idx + str(j).zfill(3)
        #    hdt.write_rcdb_input(X, spc, int(idx), r, fp, 5, 'wb97x/6-31g*', '500', freq='1', opt='0', comment=' index: '+ str(i))

        print(s_idx,' ',bad_cnt,'of',modl_std.shape[0],' Bad kept: ', len(bad_xyz))

        #print("CV1:", modl_std)
        #plt.plot(Rc[:, 1], np.abs(hdt.hatokcal*Eact[::-1] - energies[0,:][::-1]))
        #plt.plot(Rc[:, 1], energies[0,:][::-1]-energies[0,:][::-1].min(),color='green')
        #plt.plot(Rc[:, 1], modl_std[::-1],color='blue')
        #plt.plot(Rc[:, 1], hdt.hatokcal*(Rc[:, 0]-Rc[:, 0].min()),color='Black')

        #plt.show()
