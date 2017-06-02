# Import pyNeuroChem
import pyNeuroChem as pync
import numpy as np
import hdnntools as hdt
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pygau09tools as pyg
from itertools import chain
import os

def plot_irc_data(axes, file, title, ntwl, cnstfile, saefile, dir, trained):
    Eact, xyz, typ, Rc = pyg.read_irc(file)
    Rc = Rc[:,1]
    Rc = Rc[::-1]

    print(Eact.shape, Rc.shape, xyz.shape)

    # Shift reference to reactant
    #Eact = Eact[::-1]
    Eact = hdt.hatokcal * (Eact - Eact[-1])

    # Plot reference results
    axes.scatter (Rc,Eact, color='black',  linewidth=3)

    # Plot ANI results
    color = cm.rainbow(np.linspace(0, 1, len(ntwl)))
    terr = np.zeros(len(ntwl))
    derr = np.zeros(len(ntwl))
    berr = np.zeros(len(ntwl))
    for i, (nt,c) in enumerate(zip(ntwl,color)):
        ncr = pync.conformers(dir + nt[0] + cnstfile, dir + nt[0] + saefile, rcdir + nt[0] + 'networks/', 0, True)

        # Set the conformers in NeuroChem
        ncr.setConformers(confs=xyz, types=list(typ))

        # Compute Energies of Conformations
        E1 = ncr.energy()

        # Shift ANI E to reactant
        E1 = hdt.hatokcal * (E1 - E1[-1])

        # Calculate error
        errn = hdt.calculaterootmeansqrerror(E1,Eact)

        terr[i] = errn
        derr[i] = np.abs(np.abs((E1[0] - E1[-1])) - np.abs((Eact[0] - Eact[-1])))
        berr[i] = np.abs(E1.max() - Eact.max())

        # Plot
        axes.plot(Rc,E1, 'r--', color=c, label="["+str(i)+"]: "+"{:.1f}".format(berr[i]), linewidth=2)

    #axes.set_xlim([Rc.min(), Rc.max()])
    #axes.set_ylim([-15, 70])
    axes.legend(loc="upper left",fontsize=7)
    if trained:
        axes.set_title(title,color='green',fontdict={'weight':'bold'},x=0.83,y=0.70)
    else:
        axes.set_title(title, color='red', fontdict={'weight':'bold'},x=0.83,y=0.70)
    return terr,derr,berr

main_dir = '/home/jujuman/Dropbox/IRC_DBondMig/Benzene_rxn/'

# Set required files for pyNeuroChem
#rcdir  = '/home/jujuman/Research/ANI-DATASET/RXN1_TNET/training/'
rcdir  = '/home/jujuman/Scratch/Dropbox/ChemSciencePaper.AER/networks/'
cnstfile = 'rHCNOFS-4.6A_16-3.1A_a4-8.params'
saefile  = 'sae_wb97x-631gd_HCNOFS.dat'
#rcdir = '/home/jujuman/Dropbox/ChemSciencePaper.AER/networks/ANI-c08f09div-ntwk-cv/'
#cnstfile = '../rHCNO-4.6A_16-3.1A_a4-8.params'
#saefile  = '../sae_6-31gd.dat'

ntwl = [#('ANI-c08f-ntwk/', 'Org'),
        #('rxn1/ani_benz_rxn_ntwk/', '1'),
        #('rxn2/ani_benz_rxn_ntwk/', '1,2'),
        #('rxn-1-2-5-6/ani_benz_rxn_ntwk/', '1,2,5,6'),
        #('rxn1to6/ani_benz_rxn_ntwk/','6Rxn'),
        #('train_04/', 'AllAtomNet'),
        #('cv_c08e_ntw_0/','1'),
        #('cv_c08e_ntw_1/','2'),
        #('cv_c08e_ntw_2/','3'),
        #('cv_c08e_ntw_3/','4'),
        #('cv_c08e_ntw_4/','5'),
        ('ANI-SN_CHNOSF-1/','1'),]

t_list = []

# THE PROGRAM!
sub_dir = list(chain.from_iterable([[main_dir+d+'/'+i+'/IRC.log' for i in os.listdir(main_dir+d)] for d in os.listdir(main_dir) if '.docx' not in d]))

#print(lambda x:x.split('/')[-2])
sub_dir.sort(key=lambda x:(int(x.split('/')[-2].split('-')[0]), int(x.split('/')[-2].split('-')[1])))

#sub_dir = sub_dir[:-1]

print(len(sub_dir))
X = int(np.ceil(np.sqrt(len(sub_dir))))
Y = int(np.round(np.sqrt(len(sub_dir))))
print(X,Y)
f, axarr = plt.subplots(Y, X,sharey=False,sharex=False)

terrs = []
derrs = []
berrs = []
for i,d in enumerate(sub_dir):
    print(i,'): ',d)
    if X == 1 and Y ==1:
        t_te, t_de, t_be = plot_irc_data(axarr, d, d.split('/')[-2], ntwl, cnstfile,saefile, rcdir, d.split('/')[-2] in t_list)
    else:
        t_te, t_de, t_be = plot_irc_data(axarr[int(np.floor(i/X)), i % X], d, d.split('/')[-2], ntwl, cnstfile, saefile, rcdir, d.split('/')[-2] in t_list)
    terrs.append(t_te)
    derrs.append(t_de)
    berrs.append(t_be)

terrs = np.vstack(terrs)
derrs = np.vstack(derrs)
berrs = np.vstack(berrs)

cts = "         Average RMSE: "
cds = "Average R/P Delta E: "
cbs = "         Barrier height: "
for i, (ct, cd, cb) in enumerate(zip(terrs.T, derrs.T, berrs.T)):
    cts += "[" + ntwl[i][1] + "] " + "{:.2f}".format(np.mean(ct)) + " "
    cds += "[" + ntwl[i][1] + "] " + "{:.2f}".format(np.mean(cd)) + " "
    cbs += "[" + ntwl[i][1] + "] " + "{:.2f}".format(np.mean(cb)) + " "
    print("[" + ntwl[i][1] + "] " + "{:.2f}".format(np.mean(ct)) + " ")
    print("[" + ntwl[i][1] + "] " + "{:.2f}".format(np.mean(cd)) + " ")
    print("[" + ntwl[i][1] + "] " + "{:.2f}".format(np.mean(cb)) + " ")

plt.suptitle(str(len(sub_dir)) + " Diels-Alder reactions (x axis=$R_c$;y-axis=relative E [kcal/mol])\n"+cts+"\n"+cds+"\n"+cbs,fontsize=14,fontweight='bold',y=0.99)

plt.show()
#for d in
#print(sub_dir)

#en, cd, ty, Rc = pyg.read_irc(dtdir)