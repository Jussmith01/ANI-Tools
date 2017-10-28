import matplotlib.pyplot as plt
import pyNeuroChem as pync
import hdnntools as hdt
import pygau09tools as pyg

import numpy as np

#--------parameters------------
ircfile = '/home/jujuman/Dropbox/IRC_DBondMig/Diels–Alder_reaction/exo_endo/rxn'
file = 'IRC_endo.log'
ids = [1,3,6,7,8,9]

wkdir1 = '/home/jujuman/Research/ReactionGeneration/cv_multip/'
cnstfile1 = 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefile1 = 'sae_6-31gd.dat'
nnfile1 = 'train2/networks/'

wkdir2 = '/home/jujuman/Gits/ANI-Networks/networks/ANI-c08f-ntwk/'
cnstfile2 = 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefile2 = 'sae_6-31gd.dat'
nnfile2 = '/networks/'
#------------------------------

# Declare neurochem
nc1 = pync.conformers(wkdir1 + cnstfile1, wkdir1 + saefile1, wkdir1 + nnfile1, 0, False)
nc2 = pync.conformers(wkdir2 + cnstfile2, wkdir2 + saefile2, wkdir2 + nnfile2, 0, False)

barriers = []
reprdelt = []

irc1_t = []
irc2_t = []
irca_t = []

for id in ids:
    # Load IRC
    Eact, xyz, spc, Rc = pyg.read_irc(ircfile+str(id)+'/'+file)

    nc1.setConformers(confs=xyz, types=list(spc))
    Ecmp1 = nc1.energy().copy()

    nc2.setConformers(confs=xyz, types=list(spc))
    Ecmp2 = nc2.energy().copy()

    hdt.writexyzfile(ircfile+str(id)+'/'+file+'.xyz',xyz,spc)
    print(hdt.hatokcal * hdt.calculaterootmeansqrerror(Ecmp1,Eact))
    print(hdt.hatokcal * hdt.calculaterootmeansqrerror(Ecmp2,Eact))

    irc1 = hdt.hatokcal*(Ecmp1[::-1])
    irc2 = hdt.hatokcal*(Ecmp2[::-1])
    irca = hdt.hatokcal*(Eact[::-1])

    irc1_t.append(irc1)
    irc2_t.append(irc2)
    irca_t.append(irca)

    print(irc2[0],irc1[0],irca[0])

    reprdelt.append(np.array([(irc1[0] - irc1[-1]),
                              (irc2[0] - irc2[-1]),
                              (irca[0] - irca[-1])]))

    barriers.append(np.array([irc1.max(),
                              irc2.max(),
                              irca.max()]))

    print('R/P Delt: ', np.abs((irc1[0]-irc1[-1])-(irca[0]-irca[-1]))
                      , np.abs((irc2[0]-irc2[-1])-(irca[0]-irca[-1])))
    print('Barriers: ', np.abs(irc1.max()-irca.max()), np.abs(irc2.max()-irca.max()))

    plt.plot(Rc[:, 1], irc1,color='green',label='ANI-Retrain')
    plt.plot(Rc[:, 1], irc2,color='red',label='ANI-Original')
    plt.plot(Rc[:, 1], irca,color='Black', label='DFT')
    plt.legend(bbox_to_anchor=(0.01, 0.99), loc=2, borderaxespad=0., fontsize=14)
    plt.show()

irc1_t = np.concatenate(irc1_t)
irc2_t = np.concatenate(irc2_t)
irca_t = np.concatenate(irca_t)

reprdelt = np.vstack(reprdelt)
barriers = np.vstack(barriers)

print('                   ANI-rt  ANI-Org')
print('IRC MAE        : ', "{:6.2f}".format(hdt.calculatemeanabserror(irc1_t,irca_t)),
                           "{:7.2f}".format(hdt.calculatemeanabserror(irc2_t,irca_t)))

print('IRC RMSE       : ', "{:6.2f}".format(hdt.calculaterootmeansqrerror(irc1_t,irca_t)),
                           "{:7.2f}".format(hdt.calculaterootmeansqrerror(irc2_t,irca_t)))

print('Barrier MAE    : ', "{:6.2f}".format(hdt.calculatemeanabserror(barriers[:,0],barriers[:,2])),
                           "{:7.2f}".format(hdt.calculatemeanabserror(barriers[:,1],barriers[:,2])))

print('Barrier RMSE   : ', "{:6.2f}".format(hdt.calculaterootmeansqrerror(barriers[:,0],barriers[:,2])),
                           "{:7.2f}".format(hdt.calculaterootmeansqrerror(barriers[:,1],barriers[:,2])))

print('R/P Delta MAE  : ', "{:6.2f}".format(hdt.calculatemeanabserror(reprdelt[:,0],reprdelt[:,2])),
                           "{:7.2f}".format(hdt.calculatemeanabserror(reprdelt[:,1],reprdelt[:,2])))

print('R/P Delta RMSE : ', "{:6.2f}".format(hdt.calculaterootmeansqrerror(reprdelt[:,0],reprdelt[:,2])),
                           "{:7.2f}".format(hdt.calculaterootmeansqrerror(reprdelt[:,1],reprdelt[:,2])))