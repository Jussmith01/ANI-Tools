import hdnntools as hdn
import pyanitools as pyt
import os

#path = '/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnnts_nms_resample/confs_cv_gdb01-05_red03-05/confs_cv_gdb01-06_rs2.h5'
#path = '/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnnts_nms_resample/confs_cv_gdb01-05_red03-05/confs_cv_gdb01-06_rs4.h5'
path = '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-08/confs_cv_gdb01-08_rs4.h5'
#path = '/home/jujuman/Research/GDB-11-AL-wB97x631gd/mdal.h5'

dtdirs = [#'/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_mdal_resample/aminoacids/mdal1/data/',
          #'/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_mdal_resample/h2o/mdal1/data/',
          '/home/jujuman/Scratch/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-08/confs_4/data/',
          #'/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-06/confs_4/data/',
          #"/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnnts_nms_resample/confs_cv_gdb01-05_red03-05/confs_3/data/",
          #"/home/jujuman/Scratch/Research/GDB-11-wB97X-6-31gd/dnnts_nms_resample/confs_cv_gdb01-03_red03-05/data_cv_1/",
          #"/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnnts_red/dnntsgdb11_04_red/data/",
          #"/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnnts_red/dnntsgdb11_05_red/data/",
          #"/home/jujuman/Research/GDB-11-test-LOT/dnntsgdb11_02/confs/data/",
          #"/home/jujuman/Research/GDB-11-test-LOT/dnntsgdb11_03/confs/data/",
          ]

if os.path.exists(path):
    os.remove(path)

#open an HDF5 for compressed storage.
#Note that if the path exists, it will open whatever is there.
dpack = pyt.datapacker(path)

Nd = 0
Nf = 0
for d in dtdirs:

    files = [f for f in os.listdir(d) if ".dat" in f]
    files.sort()
    Nf += len(files)

    for n,f in enumerate(files):
        print(d+f)
        data = hdn.readncdatall(d+f)

        Ne = data['energies'].size
        Nd += Ne

        f = f.rsplit("-",1)

        #print(f)
        fn = f[0] + "/mol" + f[1].split(".")[0]
        #fn = f[1] + "/mol" + f[2].split(".")[0]

        #print(fn)
        print('Storing: ',fn,' data:',Ne)

        dpack.store_data(fn, **data)

print('Total data:',Nd,'from',Nf,'files.')
dpack.cleanup()

