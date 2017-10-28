import numpy as np
# Import theano
import theano
import theano.tensor as T

xyz = np.array([[ 0.6, 0.5, 0.1]
               ,[-0.55,-0.3,0.3]])

# ref calculated using w = 1.0
ref = np.array([[ 0.33733496,  0.2346678, -0.05866695]
               ,[-0.33733496, -0.2346678,  0.05866695]])

w = 0.1
e = 0.001
r0 = 1.0
print(xyz)

X = T.matrix('X')
f = T.matrix('f')
W = T.scalar('W')
R0 = T.scalar('R0')

R    = T.sqrt(T.power(X[0] - X[1],2).sum())
E    = (W/2.0)*T.power(R-R0,2)
dEdX = T.grad(E,X)
dEdW = T.grad(E,W)

# Analytic part
Cf   = T.power(dEdX-f,2)
d2EdWdX = T.jacobian(Cf.flatten(), W)

F = theano.function([X,W,R0],E)
G = theano.function([X,W,R0],dEdX)
C = theano.function([X,W,R0],dEdW)
H = theano.function([X,W,R0,f],d2EdWdX)

print(F(xyz,w,r0))
print(G(xyz,w,r0))
print(H(xyz,w,r0,ref))

for i in range(30):
    # Cost of forces error wrt W analytic
    Can = H(xyz,w,r0,ref).sum()

    # Cost of forces error wrt W numeric
    frc = G(xyz,w,r0)
    Cnu = (C(xyz+e*(frc-ref),w,r0)-C(xyz-e*(frc-ref),w,r0))/e
    w_prev = w
    w = w - 0.5*Cnu
    print(str(i).zfill(2),') Analytic:',"{:.5f}".format(Can),'Numerical:',"{:.5f}".format(Cnu),'Oldw:',"{:.5f}".format(w_prev),'eww:',"{:.5f}".format(w),'frcrmse:',"{:.5f}".format(np.sqrt(np.power(frc-ref,2).mean())))