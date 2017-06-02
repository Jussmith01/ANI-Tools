__author__ = 'Justin Smith'

from matplotlib import pyplot as plt
import numpy as np
import scipy.interpolate
import matplotlib as mpl

#--------------------------------
#           Functions
#--------------------------------

# ------------------------------------------
#          Radial Function Cos
# ------------------------------------------
def cutoffcos(X,Rc):
    Xt = X
    for i in range(0,Xt.shape[0]):
        if Xt[i] > Rc:
            Xt[i] = Rc

    return 0.5 * (np.cos((np.pi * Xt)/Rc) + 1.0)

# ------------------------------------------
#          Radial Function Cos
# ------------------------------------------
def radialfunctioncos(X,eta,Rc,Rs):
    F = np.exp(-eta*(X-Rs)**2.0) * cutoffcos(X,Rc)
    return F

# ------------------------------------------
#          Radial Function Cos
# ------------------------------------------
def angularradialfunctioncos(X,eta,Rc,Rs):
    F = np.sqrt(np.exp(-eta*(X-Rs)**2.0) * cutoffcos(X,Rc))
    return F

# ------------------------------------------
#          Radial Function Cos
# ------------------------------------------
def angularradialfunctioncos2(X,Y,eta,Rc,Rs):
    F = np.exp(-eta*((X + Y)/2.0 - Rs)**2) * np.sqrt(cutoffcos(X,Rc) * cutoffcos(Y,Rc))
    return F


# ------------------------------------------
#               Angular Function
# ------------------------------------------
def angularfunction(T,zeta,lam,Ts):
    F = 0.5 * (2.0**(1.0-zeta)) * ((1.0 + lam * np.cos(T-Ts))**zeta)
    return F

# ------------------------------------------
# Calculate The Steps for a Radial Dataset-
# ------------------------------------------
def computecutoffdataset(x1,x2,pts,Rc,plt,scolor,slabel):

    X = np.linspace(x1, x2, pts, endpoint=True)
    F = cutoffcos(X,Rc)
    plt.plot(X, F, label=slabel, color=scolor, linewidth=2)

# ------------------------------------------
# Calculate The Steps for a Radial Dataset-
# ------------------------------------------
def computeradialdataset(x1,x2,pts,eta,Rc,Rs,plt,scolor,slabel):

    X = np.linspace(x1, x2, pts, endpoint=True)
    F = radialfunctioncos(X,eta,Rc,Rs)
    plt.plot(X, F, label=slabel, color=scolor, linewidth=2)

# ------------------------------------------
# Calculate The Steps for a Radial Dataset
# ------------------------------------------
def computeangularradialdataset(x1,x2,pts,eta,Rc,Rs,plt,scolor,slabel):

    X = np.linspace(x1, x2, pts, endpoint=True)
    F = angularradialfunctioncos2(X,X,eta,Rc,Rs)
    plt.plot(X, F, label=slabel, color=scolor, linewidth=2)

# ------------------------------------------
# Calculate The Steps for an angular Dataset
# ------------------------------------------
def computeangulardataset(t1,t2,pts,zeta,lam,Ts,plt,scolor,slabel):

    T = np.linspace(t1, t2, pts, endpoint=True)
    F = angularfunction(T,zeta,lam,Ts)
    plt.plot(T, F, label=slabel, color=scolor, linewidth=2)

# ------------------------------------------
# Calculate The Steps for an angular Dataset
# ------------------------------------------
def expcost(X,tau):
    F = 2/tau * X * np.exp((X*X)/tau)
    return F

def msecost(X):
    F = X
    return F

def graphexpcost(t1,t2,pts,tau,plt,scolor,slabel):

    T = np.linspace(t1, t2, pts, endpoint=True)
    F = expcost(T,tau)
    plt.plot(T, F, label=slabel, color='red', linewidth=2)

    F = expcost(T,0.5)
    plt.plot(T, F, label=slabel, color='green', linewidth=2)

    G = msecost(T)
    plt.plot(T, G, label=slabel, color='blue', linewidth=2)

# ------------------------------------------
# Calculate The Steps for an angular Dataset
# ------------------------------------------
def printdatatofile(f,title,X,N):
    f.write(title + ' = [')
    for i in range(0,N):
        if i < N-1:
            s = "{:.7e}".format(X[i]) + ','
        else:
            s = "{:.7e}".format(X[i])
        f.write(s)
    f.write(']\n')

# ------------------------------------------
#         Simple Addition Function
# ------------------------------------------
def add (x,y):
    return x+y

# ----------------------------------------------------
# Show a 2d Contour Plot of the Angular Env Functions
# ----------------------------------------------------
def show2dcontangulargraph (ShfA,ShfZ,eta,zeta,Rc,func,title):
    N = 200000
    x, y = 2.0 * Rc * np.random.random((2, N)) - Rc

    #print(x)

    R = np.sqrt(x**2 + y**2)
    T = np.arctan2(x,y)

    z = np.zeros(N)

    for i in ShfZ:
        for j in ShfA:
            print( 'ShfZ: ' + str(i) + ' ShfA: ' + str(j) )
            #zt = angularfunction(T,zeta,1.0,i) * angularradialfunctioncos(R,eta,Rc,j) * angularradialfunctioncos(R,eta,Rc,j)
            zt = angularfunction(T,zeta,1.0,i) * angularradialfunctioncos2(R,R,eta,Rc,j)
            #zt = angularradialfunctioncos(R1,R2,eta,Rc,j)

            for k in range(1,z.shape[0]):
                z[k] = func(z[k],zt[k])
                #print(z[k])

    # Set up a regular grid of interpolation points
    xi, yi = np.linspace(x.min(), y.max(), 600), np.linspace(x.min(), y.max(), 600)
    xi, yi = np.meshgrid(xi, yi)

    zi = scipy.interpolate.griddata((x, y), z, (xi, yi), method='linear')

    plt.imshow(zi, vmin=z.min(), vmax=z.max(), origin='lower',
           extent=[x.min(), y.max(), x.min(), y.max()])

    plt.title(title)
    plt.ylabel('Distance ($\AA$)')
    plt.xlabel('Distance ($\AA$)')

    font = {'family': 'Bitstream Vera Sans',
            'weight': 'normal',
            'size': 18}
    plt.rc('font', **font)

    plt.colorbar()
    plt.show()

# ----------------------------------------------------
# Show a 2d Contour Plot of the Radial Env Functions
# ----------------------------------------------------
def show2dcontradialgraph (ShfR,eta,Rc,func,title):
    N = 200000
    x, y = 2.0 * Rc * np.random.random((2, N)) - Rc

    print(x)

    R = np.sqrt(x**2 + y**2)
    T = np.arctan2(x,y)

    z = np.zeros(N)

    for j in ShfR:
        print( 'ShfZ: ' + str(i) + ' ShfA: ' + str(j) )
        zt = radialfunctioncos(R,eta,Rc,j)

        for k in range(1,z.shape[0]):
            z[k] = func(z[k],zt[k])

    # Set up a regular grid of interpolation points
    xi, yi = np.linspace(x.min(), y.max(), 300), np.linspace(x.min(), y.max(), 300)
    xi, yi = np.meshgrid(xi, yi)

    zi = scipy.interpolate.griddata((x, y), z, (xi, yi), method='linear')

    plt.imshow(zi, vmin=z.min(), vmax=z.max(), origin='lower',
           extent=[x.min(), x.max(), y.min(), y.max()])

    font = {'family': 'Bitstream Vera Sans',
            'weight': 'normal',
            'size': 18}
    plt.rc('font', **font)

    plt.title(title)
    plt.ylabel('Distance ($\AA$)')
    plt.xlabel('Distance ($\AA$)')

    plt.colorbar()
    plt.show()

# ****************************************************
#--------------------------------
#         Set Parameters
#--------------------------------
#File name
pf = 'rHCNOSF-4.6A_8-3.1A_a4-8.params' # Output filename

Nrr = 16 # Number of shifting radial functions
Na = 6 # Number of atom types
Nar = 4 # Number of shifting angular/radial parameters
Nzt = 8 # Number of angular shifting parameters

TM = 1
Rcr = 4.6 # radial cutoff
Rca = 3.1 # Angular cutoff

xs = 1.3

#Atyp = '[H,C,O,N]'
Atyp = '[H,C,N,O,S,F]'
EtaR = np.array([8.0]) # Radial eta parameters
EtaA = np.array([18.0]) # Angular/Radial eta parameters
Zeta = np.array([12.0]) # Angular zeta parameters

# ****************************************************
cmap = mpl.cm.brg

#graphexpcost(-2.0,2.0,400,2.0,plt,'red','test')
#plt.show()

#computecutoffdataset(0.0,Rc,1000,Rc,plt,'blue','cutoff function')
#plt.show()

fontsize = 18

#--------------------------------
#         Main Program
#    (Build Env Params File)
#--------------------------------
Nrt = Nrr * Na
ShfR = np.zeros(Nrr)

#Now instead of multiple etaR we use multiple shifts with a single large EtaR
for i in range(0,Nrr):
    stepsize = (Rcr-xs) / float(Nrr)
    step = i * stepsize + xs
    color = i/float(Nrr)
    computeradialdataset(0.0, Rcr, 1000, EtaR[0], Rcr,step, plt, cmap(color), '$R_s$ = '+ "{:.2f}".format(step))
    ShfR[i] = step

plt.title('Radial environment functions (REF) \n' + r"${\eta}$ = " + "{:.2f}".format(EtaR[0]))
plt.ylabel('REF Output')
plt.xlabel('Angstroms')
plt.legend(bbox_to_anchor=(0.8, 0.98), loc=2, borderaxespad=0.)
font = {'family': 'Bitstream Vera Sans',
        'weight': 'normal',
        'size': fontsize}
plt.rc('font', **font)
plt.show()

#Uncomment for pretty contour plots of the radial environments using a sum and then max function
#show2dcontradialgraph(ShfR,EtaR,Rc,add,'Sum Radial Output')
#show2dcontradialgraph(ShfR,EtaR,Rcr,max,'Maximum radial function output')

ShfZ = np.zeros(Nzt)

Nat = Nar * (Na*(Na+1)/2) * Nzt

for i in range(0,Nzt):
    stepsize = (2.0 * np.pi) / (float(Nzt))
    step = i*stepsize
    color = i/float(Nrr)
    computeangulardataset(-np.pi,np.pi,1000,Zeta[0],1.0,step,plt, cmap(color), r"${\theta}_s$ = " + "{:.2f}".format(step))
    ShfZ[i] = step

#for i in range(0,Nzt):
#    stepsize = (2.0 * np.pi) / (float(Nzt))
#    step = i*stepsize
#    stepp = 0
#    if i is 0:
#        stepp = 0.1
#    else:
#        stepp = 1
#    color = i/float(Nrr)
#    computeangulardataset(-np.pi,np.pi,1000,2.0,1.0,step,plt, cmap(color), r"${\lambda}$ = " + "{:.2f}".format(stepp))
#    ShfZ[i] = step

plt.title('Modified Angular Environment Functions (AEF) \n' + r"${\zeta}$ = " + "{:.2f}".format(Zeta[0]))
# plt.title('Original Angular Environment Functions (OAEF) \n' + r"${\zeta}$ = " + "{:.2f}".format(2.0))
plt.ylabel('OAEF Output')
plt.xlabel('Radians')
plt.legend(bbox_to_anchor=(0.7, 0.95), loc=2, borderaxespad=0.)
font = {'family': 'Bitstream Vera Sans',
        'weight': 'normal',
        'size': fontsize}
plt.rc('font', **font)
plt.show()


ShfA = np.zeros(Nar)

for i in range(0,Nar):
    stepsize = (Rca-xs) / float(Nar)
    step = (i * stepsize + xs)
    color = i/float(Nrr)
    computeangularradialdataset(0.0, Rca, 1000, EtaA[0], Rca,step, plt, cmap(color), r"${R_s}$ = " + "{:.2f}".format(step))
    ShfA[i] = step

plt.title('Angular (Only Radial) Environment Functions (AREF)')
plt.ylabel('AREF Output')
plt.xlabel('Angstroms')
plt.legend(bbox_to_anchor=(0.7, 0.95), loc=2, borderaxespad=0.)
font = {'family': 'Bitstream Vera Sans',
        'weight': 'normal',
        'size': fontsize}
plt.rc('font', **font)
plt.show()

#Uncomment for pretty contour plots of the angular environments using a sum and then max function
#show2dcontangulargraph(ShfA,ShfZ,EtaA[0],Zeta[0],Rca,add,'Sum Angular Output')
#show2dcontangulargraph(ShfA,ShfZ,EtaA[0],Zeta[0],Rca,max,'Maximum angular output')

Nt = Nat + Nrt
print('Total Environmental Vector Size: ',int(Nt))

# Open File
f = open(pf,'w')

#Write data to parameters file
f.write('TM = ' + str(TM) + '\n')
f.write('Rcr = ' + "{:.4e}".format(Rcr) + '\n')
f.write('Rca = ' + "{:.4e}".format(Rca) + '\n')
#f.write('EtaR = ' + "{:.4e}".format(EtaR) + '\n')
printdatatofile(f,'EtaR',EtaR,EtaR.shape[0])
printdatatofile(f,'ShfR',ShfR,Nrr)
#f.write('Zeta = ' + "{:.4e}".format(Zeta) + '\n')
printdatatofile(f,'Zeta',Zeta,Zeta.shape[0])
printdatatofile(f,'ShfZ',ShfZ,Nzt)
#f.write('EtaA = ' + "{:.4e}".format(EtaA1) + '\n')
printdatatofile(f,'EtaA',EtaA,EtaA.shape[0])
printdatatofile(f,'ShfA',ShfA,Nar)
f.write('Atyp = ' + Atyp + '\n')

f.close()
