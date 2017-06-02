# Import pyNeuroChem
import pyNeuroChem as pync
import numpy as np
import hdnntools as hdt
import matplotlib.pyplot as plt

def plot_irc_data(axes, file, rcf, title):
    xyz,typ,Eact = hdt.readncdat(file,np.float32)
    Rc = np.load(rcf)

    # Set required files for pyNeuroChem
    wkdir  = '/home/jujuman/Dropbox/ChemSciencePaper.AER/networks/ANI-c08f-ntwk-cv/'
    cnstfile = 'rHCNO-4.6A_16-3.1A_a4-8.params'
    saefile  = 'sae_6-31gd.dat'

    nc =  [pync.conformers(wkdir + cnstfile, wkdir + saefile, wkdir + 'cv_c08e_ntw_' + str(l) + '/networks/', 0) for l in range(5)]

    rcdir  = '/home/jujuman/Research/ANI-DATASET/RXN1_TNET/training/rxn1to6/ani_benz_rxn_ntwk/'
    ncr1 = pync.conformers(rcdir + '../../' + cnstfile, rcdir + '../../' + saefile, rcdir + '/networks/', 0)
    ncr2 = pync.molecule(rcdir + '../../' + cnstfile, rcdir + '../../'  + saefile, rcdir + '/networks/', 0)
    ncr3 = pync.molecule(rcdir + '../../' + cnstfile, rcdir + '../../'  + saefile, rcdir + '/networks/', 0)

    # Compute reactant E
    ncr2.setMolecule(coords=xyz[0], types=list(typ))
    Er = ncr2.energy()

    # Compute product E
    ncr3.setMolecule(coords=xyz[-1], types=list(typ))
    Ep = ncr3.energy()

    #Eact = Eact[::-1]

    dE_ani = hdt.hatokcal * (Er - Ep)
    dE_dft = hdt.hatokcal * (Eact[0] - Eact[-1])
    print('Delta E R/P ANI:', dE_ani,'Delta E R/P ANI:', dE_dft,'Diff:',abs(dE_ani-dE_dft))

    # Set the conformers in NeuroChem
    ncr1.setConformers(confs=xyz, types=list(typ))

    # Compute Energies of Conformations
    E1 = ncr1.energy()

    # Shift
    E1 = E1 - E1[0]
    Eact = Eact - Eact[0]

    # Plot
    errn = hdt.calculaterootmeansqrerror(hdt.hatokcal*E1,hdt.hatokcal*Eact)
    axes.plot(Rc['x'][:,1],hdt.hatokcal * (E1), color='red', label="{:.2f}".format(errn), linewidth=2)

    axes.plot (Rc['x'][:,1],hdt.hatokcal*(Eact), 'r--', color='black',  linewidth=3)

    err = []

    for n,net in enumerate(nc):
        # Set the conformers in NeuroChem
        net.setConformers(confs=xyz,types=list(typ))

        # Compute Energies of Conformations
        E1 = net.energy()
        E1 = E1 - E1[0]

        err.append(hdt.calculaterootmeansqrerror(hdt.hatokcal*E1,hdt.hatokcal*Eact))

        # Plot
        if n == len(nc)-1:
            mean = np.mean(np.asarray(err))
            axes.plot (Rc['x'][:,1],hdt.hatokcal*(E1), color='blue',  label="{:.2f}".format(mean),  linewidth=1)
        else:
            axes.plot (Rc['x'][:,1],hdt.hatokcal*(E1), color='blue',  linewidth=1)

            axes.plot (Rc['x'][:,1],hdt.hatokcal*(E1), color='blue',  linewidth=1)


    axes.set_xlim([Rc['x'][:,1].min(), Rc['x'][:,1].max()])
    axes.legend(loc="upper right",fontsize=8)
    axes.set_title(title)
    return np.array([errn,np.mean(err)])

X = 4
Y = 3

#plt.plot (x, morse(x,1.0,1.9,1.53), color='grey',  label='SRC+ANI',  linewidth=2)
f, axarr = plt.subplots(Y, X,sharey=False,sharex=False)
errors = []
for i in range(0,10):
    file = '/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnnts_rxns/benz_dbm_rxns_full/irc_dbm_benz_' + str(i+1) + '/irc.dat'
    rcf  = '/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnnts_rxns/benz_dbm_rxns_full/irc_dbm_benz_' + str(i+1) + '/reaction_coordinate.npz'
    errors.append(plot_irc_data(axarr[int(np.floor(i/X)), i % X], file, rcf, 'rxn' + str(i + 1)))
errors = np.vstack(errors)

er1 = np.mean(errors[:,0])
er2 = np.mean(errors[:,1])

plt.suptitle("Double bond migration IRCs\n(Blue: old cv networks; red: network trained to new data; black DFT)\n(New data rxn index: 1 through 6)\nx-axis=Rc ($\AA$);y-axis=relative E; Avg New net error:" + "{:.2f}".format(er1) + "; Avg. CV error: " + "{:.2f}".format(er2))

#plt.ylabel('E (kcal/mol)')
#plt.xlabel('Distance $\AA$')
#plt.legend(bbox_to_anchor=(0.05, 0.95), loc=2, borderaxespad=0.,fontsize=16)

plt.show()