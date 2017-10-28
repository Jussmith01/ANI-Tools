import hdnntools as hdt
import pyanitools as pyt
import os

file = '/home/jujuman/Research/DataReductionMethods/model6/model0.05me/ani_red_c06.h5'
sdir = '/home/jujuman/Research/GDB-11-AL-wB97x631gd/'

aload = pyt.anidataloader(file)

for data in aload:

    X = data['coordinates']
    S = data['species']
    P = data['path']

    parent = P.split('/')[1]
    index  = P.split('/')[2].split('mol')[1].zfill(7)

    path = sdir+parent
    if not os.path.exists(path):
        os.mkdir(path)

    print(path + '/' + parent + '-' + index + '.xyz','DATA:',X.shape[0])
    hdt.writexyzfile(path+'/'+parent+'-'+index+'.xyz',X,S)
