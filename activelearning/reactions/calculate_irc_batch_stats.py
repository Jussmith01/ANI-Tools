import hdnntools as hdt
import pygau09tools as pyg
import os
import matplotlib.pyplot as plt
import pyNeuroChem as pync
import numpy as np
import random

def plot_irc(axes, i, d, f):
    #print(f)
    Eact, xyz, spc, Rc = pyg.read_irc(d+f)
    Eact = hdt.hatokcal * Eact

    xyz = xyz[1:]
    Eact = Eact[1:]
    Rc = Rc[:-1]

    #print(Rc[:,1])
    #print(Eact-Eact.min() - Rc[:,1]-Rc[:,1].min())
    s_idx = f.split('IRC')[1].split('.')[0]
    hdt.writexyzfile(c+f.split('.')[0]+'.xyz',xyz,spc)
    #print(f.split('IRC')[1].split('.')[0],Rc.shape)
    if Rc.size > 10:
        #------------ CV NETWORKS 1 -----------
        energies1 = []
        N = 0
        for comp in nc1:
            comp.setConformers(confs=xyz, types=list(spc))
            energies1.append(hdt.hatokcal*comp.energy())
            N = N + 1

        energies2 = []
        N = 0
        for comp in nc2:
            comp.setConformers(confs=xyz, types=list(spc))
            energies2.append(hdt.hatokcal*comp.energy())
            N = N + 1

        modl_std1 = np.std(energies1, axis=0)[::-1]
        energies1 = np.mean(np.vstack(energies1),axis=0)

        modl_std2 = np.std(energies2, axis=0)[::-1]
        energies2 = np.mean(np.vstack(energies2),axis=0)

        rmse1 = hdt.calculaterootmeansqrerror(energies1,Eact)
        rmse2 = hdt.calculaterootmeansqrerror(energies2,Eact)

        dba = Eact.max()-Eact[0]
        db1 = energies1.max() - energies1[0]
        db2 = energies2.max() - energies2[0]

        rpa = Eact[0] - Eact[-1]
        rp1 = energies1[0] - energies1[-1]
        rp2 = energies2[0] - energies2[-1]

        bar1.append(abs(db1-dba))
        bar2.append(abs(db2-dba))

        rmp1.append(abs(rpa-rp1))
        rmp2.append(abs(rpa-rp2))

        Ec1.append(energies1)
        Ec2.append(energies2)
        Ea.append(Eact)

        print(i,')',f,':',len(spc),':',rmse1,rmse2,'R/P1: ',abs(rpa-rp1),'R/P2: ',abs(rpa-rp2),'Barrier1:',abs(db1-dba),'Barrier2:',abs(db2-dba))

        Rce = hdt.hatokcal * (Rc[:, 0] - Rc[:, 0][0])
        Rce1 = energies2[::-1]-energies2[::-1][0]

        axes.set_xlim([Rc.min(), Rc.max()])
        axes.set_ylim([Rce.min()-1.0, Rce1.max()+20.0])

        axes.plot(Rc[:, 1], hdt.hatokcal*(Rc[:, 0]-Rc[:, 0][0]),color='Black', label='DFT')

        axes.errorbar(Rc[:, 1], energies2[::-1]-energies2[::-1][0], yerr=modl_std2, fmt='--',color='red',label="ANI-1: "+"{:.1f}".format(bar2[-1]),linewidth=2)
        axes.errorbar(Rc[:, 1], energies1[::-1]-energies1[::-1][0], yerr=modl_std1, fmt='--',color='blue',label="["+str(i)+"]: "+"{:.1f}".format(bar1[-1]),linewidth=2)
        #axes.set_xlabel("Reaction Coordinate $\AA$")
        #axes.set_ylabel(r"$\Delta E$ $ (kcal \times mol^{-1})$")
        #axes.plot(Rc[:, 1], energies2[::-1]-energies2[::-1][0],'--',color='red',label="["+str(i)+"]: "+"{:.1f}".format(bar2[-1]),linewidth=3)
        #axes.plot(Rc[:, 1], energies1[::-1]-energies1[::-1][0],'--',color='green',label="["+str(i)+"]: "+"{:.1f}".format(bar1[-1]),linewidth=3)

        axes.legend(loc="upper left",fontsize=10)
        axes.set_title(str(f), color='black', fontdict={'weight': 'bold'}, x=0.8, y=0.85)


os.environ["PYTHONPATH"] = "../../lib"

d = '/home/jujuman/Dropbox/IRC_DBondMig/auto-TS/auto-TS2/testset/IRC/'
c = '/home/jujuman/Dropbox/IRC_DBondMig/auto-TS/auto-TS2/testset/XYZ/'
r = '/home/jujuman/Dropbox/IRC_DBondMig/auto-TS/auto-TS2/testset/DataGen/'
fp = 'DA_IRC'

#d = '/home/jujuman/Dropbox/IRC_DBondMig/Cope/Try1/IRC/'
#c = '/home/jujuman/Dropbox/IRC_DBondMig/Cope/Try1/XYZ/'
#r = '/home/jujuman/Dropbox/IRC_DBondMig/Cope/Try1/DataGen/'
#fp = 'CP_IRC'

#d = '/home/jujuman/Dropbox/IRC_DBondMig/Claisen/Try3/testset/IRC/'
#c = '/home/jujuman/Dropbox/IRC_DBondMig/Claisen/Try3/testset/XYZ/'
#r = '/home/jujuman/Dropbox/IRC_DBondMig/Claisen/Try3/testset/DataGen/'
#fp = 'CL_IRC'


#wkdircv1 = '/home/jujuman/Research/ForceTrainTesting/train_full_al1/'
#cnstfile1 = 'rHCNO-4.6A_16-3.1A_a4-8.params'
#saefile1 = 'sae_6-31gd.dat'

wkdircv2 = '/home/jujuman/Research/ANI-validation/'
cnstfile2 = 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefile2 = 'sae_6-31gd.dat'

#wkdircv2 = '/home/jujuman/Research/DataReductionMethods/model6r/model-gdb_r06_comb08_2/cv2/'
#cnstfile2 = 'rHCNO-4.6A_16-3.1A_a4-8.params'
#saefile2 = 'sae_6-31gd.dat'

wkdircv1 = '/home/jujuman/Research/DataReductionMethods/model6r/model-gdb_r06_comb09_1/cv4_2/'
cnstfile1 = 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefile1 = 'sae_6-31gd.dat'

#wkdircv2 = '/home/jujuman/Gits/ANI-Networks/networks/ANI-c08f-ntwk-cv/'
#cnstfile2 = 'rHCNO-4.6A_16-3.1A_a4-8.params'
#saefile2 = 'sae_6-31gd.dat'

Nnc = 5

files = os.listdir(d)[0:3]
files.sort()
#random.shuffle(files)
#files = files[0:9]
#print(files)
# Construct pyNeuroChem classes
nc1 =  [pync.conformers(wkdircv1 + cnstfile1, wkdircv1 + saefile1, wkdircv1 + 'train' + str(l) + '/networks/', 0, False) for l in range(Nnc)]
nc2 =  [pync.conformers(wkdircv2 + cnstfile2, wkdircv2 + saefile2, wkdircv2 + 'train' + str(l) + '/networks/', 0, False) for l in range(Nnc)]

comp_xyz = []

print('N-IRC',len(files))
X = int(np.ceil(np.sqrt(len(files))))
Y = int(np.round(np.sqrt(len(files))))
print(X,Y)
f, axarr = plt.subplots(Y, X, sharey=False, sharex=False)

Ec1 = []
Ec2 = []
Ea  = []
rmp1 = []
bar1 = []
rmp2 = []
bar2 = []
for i,f in enumerate(files):
    plot_irc(axarr[int(np.floor(i / X)), i % X], i, d, f)

plt.show()

bar1 = np.array(bar1) # Barrier 1
bar2 = np.array(bar2) # Barrier 2

rmp1 = np.array(rmp1) # Reactant product 1
rmp2 = np.array(rmp2) # Reactant product 2

Ec1 = np.concatenate(Ec1)
Ec2 = np.concatenate(Ec2)
Ea  = np.concatenate(Ea)

#plt.suptitle(str(len(files)) + " Diels-Alder reactions (x axis=$R_c$;y-axis=relative E [kcal/mol])\n"+cts+"\n"+cds+"\n"+cbs,fontsize=14,fontweight='bold',y=0.99)

print('Barrier   - ANI retrain:',bar1.sum()/bar1.size,'Original ANI:',bar2.sum()/bar2.size)
print('Reac/prod - ANI retrain:',rmp1.sum()/rmp1.size,'Original ANI:',rmp2.sum()/rmp2.size)
print('IRC RMSE  - ANI Retrain:',hdt.calculaterootmeansqrerror(Ec1,Ea),'Original ANI:',hdt.calculaterootmeansqrerror(Ec2,Ea))