import hdnntools as hdn
import pyanitools as pyt
import os

path = '/home/jujuman/Research/GDB-11-test-LOT/ani-gdb-c02.h5'

dtdirs = ["/home/jujuman/Research/GDB-11-test-LOT/dnntsgdb11_01/confs/data/",
          "/home/jujuman/Research/GDB-11-test-LOT/dnntsgdb11_02/confs/data/"]

if os.path.exists(path):
    os.remove(path)

#open an HDF5 for compressed storage.
#Note that if the path exists, it will open whatever is there.
dpack = pyt.datapacker(path)

for d in dtdirs:

    files = [f for f in os.listdir(d) if ".dat" in f]
    files.sort()

    for n,f in enumerate(files):
        data = hdn.readncdatall(d+f)
        f = f.split("-")
        print('Storing: ',f[0] + "/mol" + f[1].split(".")[0])

        dpack.store_data(f[0] + "/mol" + f[1].split(".")[0], **data)

dpack.cleanup()

