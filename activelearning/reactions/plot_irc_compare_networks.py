# Import pyNeuroChem
import pyNeuroChem as pync
import numpy as np
import hdnntools as hdt
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

def plot_irc_data(axes, file, rcf, title, ntwl, cnstfile, saefile, dir, idx):
    xyz,typ,Eact = hdt.readncdat(file,np.float32)
    Rc = np.load(rcf)

    # Shift reference to reactant
    #Eact = Eact[::-1]
    Eact = hdt.hatokcal * (Eact - Eact[0])

    # Plot reference results
    axes.plot (Rc['x'][:,1],Eact, color='black',  linewidth=3)

    # Plot ANI results
    color = cm.rainbow(np.linspace(0, 1, len(ntwl)))
    terr = np.zeros(len(ntwl))
    derr = np.zeros(len(ntwl))
    berr = np.zeros(len(ntwl))
    for i, (nt,c) in enumerate(zip(ntwl,color)):
        ncr = pync.conformers(dir + cnstfile, dir + saefile, rcdir + nt[0] + 'networks/', 0)

        # Set the conformers in NeuroChem
        ncr.setConformers(confs=xyz, types=list(typ))

        # Compute Energies of Conformations
        E1 = ncr.energy()

        # Shift ANI E to reactant
        E1 = hdt.hatokcal * (E1 - E1[0])

        # Calculate error
        errn = hdt.calculaterootmeansqrerror(E1,Eact)

        terr[i] = errn
        derr[i] = np.abs(np.abs((E1[0] - E1[-1])) - np.abs((Eact[0] - Eact[-1])))
        berr[i] = np.abs(E1.max() - Eact.max())

        # Plot
        axes.plot(Rc['x'][:,1],E1, 'r--', color=c, label="["+nt[1]+"]: "+"{:.2f}".format(errn), linewidth=2)
        #axes.plot([Rc['x'][:,1].min(),Rc['x'][:,1].max()],[E1[-1],E1[-1]], 'r--', color=c)
        #axes.plot([Rc['x'][:,1].min(),Rc['x'][:,1].max()],[E1[0],E1[0]], 'r--', color=c)

    axes.set_xlim([Rc['x'][:,1].min(), Rc['x'][:,1].max()])
    axes.legend(loc="upper left",fontsize=12)
    if idx < 6:
        axes.set_title(title,color='green',fontdict={'weight':'bold'})
    else:
        axes.set_title(title, color='red', fontdict={'weight': 'bold'})
    return terr,derr,berr


# Set required files for pyNeuroChem
rcdir  = '/home/jujuman/Research/ANI-DATASET/RXN1_TNET/training/'
cnstfile = 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefile  = 'sae_6-31gd.dat'

ntwl = [('ANI-c08f-ntwk/', 'N'),
        #('rxn1/ani_benz_rxn_ntwk/', '1'),
        #('rxn2/ani_benz_rxn_ntwk/', '1,2'),
        #('rxn-1-2-5-6/ani_benz_rxn_ntwk/', '1,2,5,6'),
        ('rxn1to6/ani_benz_rxn_ntwk/','1-6'),
        ]

X = 4
Y = 3

f, axarr = plt.subplots(Y, X,sharey=False,sharex=False)
terrs = []
derrs = []
berrs = []
for i in range(0,10):
    file = '/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnnts_rxns/benz_dbm_rxns_full/irc_dbm_benz_' + str(i+1) + '/irc.dat'
    rcf  = '/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnnts_rxns/benz_dbm_rxns_full/irc_dbm_benz_' + str(i+1) + '/reaction_coordinate.npz'
    t_te,t_de, t_be = plot_irc_data(axarr[int(np.floor(i/X)), i % X], file, rcf, 'rxn' + str(i + 1), ntwl, cnstfile, saefile, rcdir, i)
    terrs.append(t_te)
    derrs.append(t_de)
    berrs.append(t_be)

terrs = np.vstack(terrs)
derrs = np.vstack(derrs)
berrs = np.vstack(berrs)

cts = "Average RMSE:        "
cds = "Average R/P Delta E: "
cbs = "Barrier height:      "
for i, (ct, cd, cb) in enumerate(zip(terrs.T, derrs.T, berrs.T)):
    cts += "[" + ntwl[i][1] + "] " + "{:.2f}".format(np.mean(ct[6:])) + " "
    cds += "[" + ntwl[i][1] + "] " + "{:.2f}".format(np.mean(cd[6:])) + " "
    cbs += "[" + ntwl[i][1] + "] " + "{:.2f}".format(np.mean(cb[6:])) + " "
    print("[" + ntwl[i][1] + "] " + "{:.2f}".format(np.mean(ct[6:])) + " ")
    print("[" + ntwl[i][1] + "] " + "{:.2f}".format(np.mean(cd[6:])) + " ")
    print("[" + ntwl[i][1] + "] " + "{:.2f}".format(np.mean(cb[6:])) + " ")

plt.suptitle("(x axis=step;y-axis=relative E [kcal/mol])\n"+cts+"\n"+cds+"\n"+cbs,fontsize=12)
#plt.suptitle("(x-axis=Rc ($\AA$);y-axis=relative E [kcal/mol]) Green = training set; Red = test set",fontsize=18)

#plt.ylabel('E (kcal/mol)')
#plt.xlabel('Distance $\AA$')
#plt.legend(bbox_to_anchor=(0.05, 0.95), loc=2, borderaxespad=0.,fontsize=16)

plt.show()