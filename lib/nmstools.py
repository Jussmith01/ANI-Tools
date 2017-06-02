import numpy as np

mDynetoMet = 1.0e-5 * 1.0e-3 * 1.0e10
Kb = 1.38064852e-23
MtoA = 1.0e10

class nmsgenerator():
    # xyz = initial min structure
    # nmo = normal mode displacements
    # fcc = force constants
    # spc = atomic species list
    # T   = temperature of displacement
    def __init__(self,xyz,nmo,fcc,spc,T,minfc = 1.0E-3):
        self.xyz = xyz
        self.nmo = nmo
        self.fcc = np.array([i if i > minfc else minfc for i in fcc])
        self.chg = np.array([self.__getCharge__(i) for i in spc])
        self.T = T
        self.Na = xyz.shape[0]
        self.Nf = nmo.shape[0]

    # returns atomic charge
    def __getCharge__(self,type):
        if type is 'H':
            return 1.0
        elif type is 'C':
            return 6.0
        elif type is 'N':
            return 7.0
        elif type is 'O':
            return 8.0
        else:
            print('Unknown atom type! ',type)
            exit(1)

    # Checks for small atomic distances
    def __check_atomic_distances__(self,rxyz):
        for i in range(0,self.Na):
            for j in range(i+1,self.Na):
                Rij = np.linalg.norm(rxyz[i]-rxyz[j])
                if Rij < 0.0075 * (self.chg[i] * self.chg[j]) + 0.65:
                    return True
        return False

    # Generate a structure
    def __genrandomstruct__(self):
        rdt = np.random.random(self.Nf+1)
        rdt[0] = 0.0
        norm = np.random.random(1)[0]
        rdt = norm*np.sort(rdt)
        rdt[self.Nf] = norm

        oxyz = self.xyz.copy()

        for i in range(self.Nf):
            Ki = mDynetoMet * self.fcc[i]
            ci = rdt[i+1]-rdt[i]
            Sn = -1.0 if np.random.binomial(1,0.5,1) else 1.0
            Ri = Sn * MtoA * np.sqrt((3.0 * ci * float(self.Na) * Kb * self.T)/Ki)
            oxyz = oxyz + Ri * self.nmo[i]
        return oxyz

    # Call this to return a random structure
    def get_random_structure(self):
        gs = True
        while gs:
            rxyz = self.__genrandomstruct__()
            gs = self.__check_atomic_distances__(rxyz)
        return rxyz
