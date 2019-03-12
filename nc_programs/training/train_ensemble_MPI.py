#
# coding: utf-8

# In[1]:


import pyanitools as pyt
import anitraintools as alt

import numpy as np

import sys
import os

from multiprocessing import Process

from mpi4py import MPI

# Run MPI ranks
comm = MPI.COMM_WORLD

if len(sys.argv) != 6:
    print('Error: missing arguments!')
    exit(1)

ensemble_size = 8

print('Storing in: ',sys.argv[1])

ndir = str(sys.argv[1])
root = str(sys.argv[2])
Rcr = float(sys.argv[3])
Rca = float(sys.argv[4])
seed = str(sys.argv[5])

mpi_rank = comm.Get_rank()

if mpi_rank == 0:
    os.mkdir(root+'/'+ndir)

prm = alt.anitrainerparamsdesigner(['Sn'], 32, 8, 8, Rcr, Rca, 2.0)
prm.create_params_file(root+ndir)

ipt = alt.anitrainerinputdesigner()
ipt.set_parameter('atomEnergyFile','sae_linfit.dat')
ipt.set_parameter('sflparamsfile',prm.get_filename())
ipt.set_parameter('eta','0.001')
ipt.set_parameter('energy','1')
ipt.set_parameter('force','1')
ipt.set_parameter('fmult','0.01')
ipt.set_parameter('feps','0.001')
ipt.set_parameter('dipole','0')
ipt.set_parameter('cdweight','2.0')
ipt.set_parameter('tolr','100')
ipt.set_parameter('tbtchsz','128')
ipt.set_parameter('vbtchsz','128')
ipt.set_parameter('nkde','0')
ipt.set_parameter('pbc','1')
ipt.set_parameter('boxx','16.0')
ipt.set_parameter('boxy','16.0')
ipt.set_parameter('boxz','16.0')

# Set network layers
ipt.add_layer('Sn',{"nodes":96,"activation":9,"type":0,"l2norm":0,"l2valu":0.000001})
ipt.add_layer('Sn',{"nodes":96 ,"activation":9,"type":0,"l2norm":0,"l2valu":0.000001})
ipt.add_layer('Sn',{"nodes":64 ,"activation":9,"type":0,"l2norm":0,"l2valu":0.000001})

#ipt.add_layer('Sn',{"nodes":96 ,"activation":9,"type":0,"l2norm":0,"l2valu":0.000001})
#ipt.add_layer('Sn',{"nodes":80 ,"activation":9,"type":0,"l2norm":0,"l2valu":0.000001})
#ipt.add_layer('Sn',{"nodes":64 ,"activation":9,"type":0,"l2norm":0,"l2valu":0.000001})

wdir = root

#mpi_rank = comm.Get_rank()

if mpi_rank == 0:
    ipt.print_layer_parameters()

netdict = {'cnstfile':wdir+'/'+ndir+'/'+prm.get_filename(),
           'saefile':wdir+'/'+ndir+'/sae_linfit.dat',
           'iptsize':prm.get_aev_size(),
           'atomtyp':prm.params['elm']}

if mpi_rank == 0:
    rank_seeds = np.random.randint(0, 2**32, comm.Get_size())
    for r in range(1,rank_seeds.size):
        comm.Send(rank_seeds, dest=r, tag=13)
    rank_seed = rank_seeds[mpi_rank]
else:
    rank_seeds = np.empty(comm.Get_size(),dtype=np.int)
    comm.Recv(rank_seeds, source=0, tag=13)
    rank_seed = rank_seeds[mpi_rank]

# Seed this generator with the rank seed
np.random.seed(rank_seed)

local_seeds = np.random.randint(0, 2**32, size=2)

# Declare the training class for all ranks
ens = alt.alaniensembletrainer(wdir+'/'+ndir+'/',
                               netdict,
                               ipt,
                               wdir+'../h5files_gen14/',
                               #wdir+'h5files-Sn-conly_gen1/',
                               ensemble_size,random_seed=local_seeds[0])

# Have rank 0 build the training caches
if mpi_rank == 0:
    ens.build_strided_training_cache(16,2,1,Ekey='energy',
                                     forces=True,Fkey='force',forces_unit=1.0,grad=True,
                                     dipole=False,
                                     rmhighe=True,rmhighf=2.5,pbc=True)

# Wait for master to catchup after building the caches
comm.Barrier()

if comm.Get_size() > ensemble_size:
    print("Error: more ranks requested than required for training this ensemble.")
    exit(1)

Nnet_per_rank = int(ensemble_size/comm.Get_size())

# Train the model corresponding to this rank
host = MPI.Get_processor_name()
hosts = comm.allgather(host)

gpuid = mpi_rank%4
print(mpi_rank,host,' GPU: ',gpuid,local_seeds[1])

ens.train_ensemble_single(gpuid, [mpi_rank], False, local_seeds[1])
