import numpy as np

class molfrag():
    def __init__(self,xyz,spc):
        self.xyz = xyz
        self.spc = spc

    def fragment(self, i, cut1, cut2):
        Xc = self.xyz[i]
        new_spc = [self.spc[i]]
        new_xyz = [self.xyz[i]-Xc]

        for j,x in enumerate(self.xyz):
            if i != j:
                dij = np.linalg.norm(self.xyz[i]-x)
                if dij <= cut1:
                    new_spc.append(self.spc[j])
                    new_xyz.append(x-Xc)
                elif dij > cut1 and dij < cut2 and self.spc[j] is 'H':
                    #print('Add shell:', j)
                    new_spc.append(self.spc[j])
                    new_xyz.append(x-Xc)

        del_elm = []
        for j,x1 in enumerate(new_xyz):
            if new_spc[j] == 'H':
                lone = True
                for k,x2 in enumerate(new_xyz):
                    if j != k and np.linalg.norm(x1-x2) < 1.1:
                        lone = False
                if not lone:
                    del_elm.append(j)
            else:
                del_elm.append(j)

        new_xyz = np.vstack(new_xyz)[del_elm]
        new_spc = np.array(new_spc)[del_elm]

        return new_xyz,new_spc

    def order_by_type(self,frags):
        for i,f in enumerate(frags):
            sort_idx = np.argsort(f[0])
            srt_spc = f[0][sort_idx]
            srt_xyz = f[1][sort_idx]
            frags[i] = (srt_spc,srt_xyz)


    def get_all_frags(self,type, cut1, cut2):
        fragments = []
        for i,s in enumerate(self.spc):
            if s is type:
                xyz, spc = self.fragment(i,cut1,cut2)
                fragments.append((spc,xyz))

        self.order_by_type(fragments)
        return fragments

