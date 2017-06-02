import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('/home/jujuman/Python/PycharmProjects/HD-AtomNNP/ANI_ASE_Programs/temp.dat',dtype=np.float32,delimiter=' ')
data = np.loadtxt('/home/jujuman/Research/CrossValidation/MD_CV/md-peptide-cv.dat',dtype=np.float32,delimiter=' ')

print(data)

#t = (data[:,0] * 0.25) / 1000.0
t = data[:,0]

plt.figure()
f, axes = plt.subplots(2, 1)

axes[0].plot(t,data[:,1],color='blue',label='E0')
axes[0].plot(t,data[:,2],color='red',label='E1')
axes[0].plot(t,data[:,3],color='green',label='E2')
axes[0].plot(t,data[:,4],color='orange',label='E3')
axes[0].plot(t,data[:,5],color='black',label='E4')

axes[0].set_ylabel('E (kcal/mol)')
plt.legend(bbox_to_anchor=(0.2, 0.98), loc=2, borderaxespad=0., fontsize=14)

axes[1].plot(t,data[:,6],color='black',label='E4')
axes[1].set_ylabel('Std. Dev. (kcal/mol/atom)')
axes[1].set_xlabel('t (ps)')

plt.suptitle("Benzene MD cross-validation")

font = {'family' : 'Bitstream Vera Sans',
        'weight' : 'normal',
        'size'   : 14}

plt.rc('font', **font)

plt.show()

