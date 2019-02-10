#
# coding: utf-8

# In[1]:


import pyanitools as pyt
import anitraintools as alt


# In[5]:

ndir = 'model_test17'
wdir = '/home/jujuman/Research/dipole_training/test_ani-1x/'
Nn = 4
Rcr = 5.0
Rca = 3.5

prm = alt.anitrainerparamsdesigner(['H','O'], 16, 8, 4, Rcr, Rca, 0.9)
prm.create_params_file(wdir+ndir)

ipt = alt.anitrainerinputdesigner()

ipt.set_parameter('atomEnergyFile','sae_linfit.dat')
ipt.set_parameter('sflparamsfile',prm.get_filename())
ipt.set_parameter('eta','0.001')
ipt.set_parameter('energy','1')
ipt.set_parameter('dipole','0')
ipt.set_parameter('cdweight','0.1')
#ipt.set_parameter('freezelayer','0,1,2')
ipt.set_parameter('tolr','100')
ipt.set_parameter('tbtchsz','1024')
ipt.set_parameter('vbtchsz','1024')
ipt.set_parameter('nkde','0')

ipt.add_layer('H',{"nodes":96,"activation":9,"type":0,"l2norm":1,"l2valu":0.0000025})
ipt.add_layer('H',{"nodes":80,"activation":9,"type":0,"l2norm":1,"l2valu":0.0000025})
ipt.add_layer('H',{"nodes":64,"activation":9,"type":0,"l2norm":1,"l2valu":0.0000025})
#ipt.add_layer('H',{"nodes":2, "activation":6,"type":0})

ipt.add_layer('O',{"nodes":80,"activation":9,"type":0,"l2norm":1,"l2valu":0.0000025})
ipt.add_layer('O',{"nodes":64,"activation":9,"type":0,"l2norm":1,"l2valu":0.0000025})
ipt.add_layer('O',{"nodes":48,"activation":9,"type":0,"l2norm":1,"l2valu":0.0000025})
#ipt.add_layer('O',{"nodes":2, "activation":6,"type":0})

ipt.print_layer_parameters()

# In[6]:
netdict = {'cnstfile':wdir+'/'+ndir+'/'+prm.get_filename(),
           'saefile':wdir+'/'+ndir+'/sae_linfit.dat',
           'iptfile':wdir+'/'+ndir+'/inputtrain.ipt',
           'iptsize':prm.get_aev_size(),
           'atomtyp':prm.params['elm']}


# In[7]:
ens = alt.alaniensembletrainer(wdir+'/'+ndir+'/',
                               netdict,
                               ipt,
                               wdir+'h5files_water/',
                               Nn)


# In[8]:
#ens.build_strided_training_cache(10,1,1,forces=False,dipole=False,Dkey='hirdipole')
#ens.build_strided_training_cache(10,1,1,forces=False,dipole=True,Dkey='dipoles')
ens.build_strided_training_cache(16,1,1,Ekey='mp2_tz_energies',Eax0sum=True,forces=True,Fkey='mp2_tz_grad',grad=True,dipole=True,Dkey='mp2_tz_dipoles',dipole_unit=0.529177249)
#ens.build_strided_training_cache(16,1,1,forces=False,dipole=True,Dkey='mp2_dz_dipoles',dipole_unit=1.0/1.889725989)
#ens.build_strided_training_cache(10,1,1,forces=False,dipole=True,Dkey='Nccsd(t)_tz_dipoles',dipole_unit=1.0)
#ens.build_strided_training_cache(16,1,1,forces=False,dipole=True,Dkey='mp2_tz_dipoles',dipole_unit=0.529177249)
#ens.build_strided_training_cache(8,1,1,forces=False,dipole=True,Dkey='dipoles',dipole_unit=1.0,charge=True)


# In[9]:
ens.train_ensemble([0])

#ens.train_ensemble([0,1,6,7])
#ens.train_ensemble([2,3,4,5])

