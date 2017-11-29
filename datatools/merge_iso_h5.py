import pyanitools as pyt
import re

h5_list = ['/home/jsmith48/scratch/auto_al/h5files/ANI-AL-0605.0001.0001.h5',
           '/home/jsmith48/scratch/auto_al/h5files/ANI-AL-0605.0001.0002.h5',
           '/home/jsmith48/scratch/auto_al/h5files/ANI-AL-0605.0001.0003.h5',
           '/home/jsmith48/scratch/auto_al/h5files/ANI-AL-0605.0001.0004.h5',
           '/home/jsmith48/scratch/auto_al/h5files/ANI-AL-0605.0001.0005.h5',
           '/home/jsmith48/scratch/auto_al/h5files/ANI-AL-0605.0001.0006.h5',
           '/home/jsmith48/scratch/auto_al/h5files/ANI-AL-0605.0001.0007.h5',
           '/home/jsmith48/scratch/auto_al/h5files/ANI-AL-0605.0001.0008.h5',
           '/home/jsmith48/scratch/auto_al/h5files/ANI-AL-0605.0001.0009.h5',
           '/home/jsmith48/scratch/auto_al/h5files/ANI-AL-0605.0001.0010.h5',
           '/home/jsmith48/scratch/auto_al/h5files/ANI-AL-0605.0001.0011.h5',
          ]

r = re.compile('(.+?)(\d+?)')
comb = dict()
for h5 in h5_list:
    adl = pyt.anidataloader(h5)
    for data in adl:
        key = data['path'].split('_')[1].split('/')[0]
        items = r.findall(key)
        #print(key, items, sorted(items))
        #print(data.keys())
        if key in comb:
            comb[key].append(data)
        else:
            comb[key] = [data]

print(len(list(comb.keys())))
for k in comb.keys():
    data = comb[k]
    data_new = dict()
    data_new['energies'] = data

    print(k,data_new['energies'])
