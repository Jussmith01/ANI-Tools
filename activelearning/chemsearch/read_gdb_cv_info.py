import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


Nsamp = 313684
f1 = open('/home/jujuman/Research/CrossValidation/GDB-09-High-sdev/gdb-09-1.0sdev.dat','r')
f2 = open('/home/jujuman/Research/CrossValidation/GDB-09-High-sdev/gdb-09-1.0sdev_fix.dat','r')

fo = open("/home/jujuman/Research/CrossValidation/GDB-09-High-sdev/gdb-09-1.0",'w')
sigma1 = []
for i in f1.readlines():
    data = i.split(":")

    mid = int(data[0].split('(')[0])

    sd = float(data[3].split('=')[1])
    sigma1.append(sd)
    print(mid,' ',sd,' ',data[4])
    #fo.write(mid,' ', sd, ' ', )


sigma2 = []
for i in f2.readlines():
    data = i.split(":")

    mid = int(data[0].split(' ')[2])

    sd = float(data[3].split('=')[1])
    sigma2.append(sd)
    print(mid,' ',sd)



x1 = np.sort(np.array(sigma1))
x2 = np.sort(np.array(sigma2))
#x1 = x[:x.shape[0]-2500]
#x2 = x[x.shape[0]-2500:]

#print(x)

f, axarr = plt.subplots(2)

n, bins, patches = axarr[0].hist(x1, 1000, normed=0, facecolor='green', alpha=0.5)
n, bins, patches = axarr[0].hist(x2, 1000, normed=0, facecolor='blue', alpha=0.5)

axarr[0].grid(True)
axarr[1].grid(True)

axarr[0].set_xlabel('Standard deviation (kcal/mol)')
axarr[0].set_ylabel('Count')

axarr[1].set_xlabel('Standard deviation (kcal/mol)')
axarr[1].set_ylabel('Count')

#print('Total: ', x.shape[0], ' Split: ', x.shape[0]-2500)

#plt.suptitle(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
#plt.axis([40, 160, 0, 0.03])

plt.show()
