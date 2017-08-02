import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn import metrics

import hdnntools as hdt

data = np.loadtxt('/home/jujuman/Downloads/BMI_ML.txt')

X = data[:,3:]
Y = data[:, 1].reshape(-1, 1)

print(X)

#plt.hist(X.flatten(), bins=100)
#plt.show()

means = np.mean(X, axis=0)
print(means)
X = X - means
print(X)

maxes = np.max(X,axis=0)
X = X/maxes

print(X)
#np.random.shuffle(X)

print(X)

scaler = preprocessing.StandardScaler().fit(Y)
Y = scaler.transform(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    train_size=13000,
                                                    random_state=10)

nnr = MLPRegressor(activation='relu', solver='adam', batch_size=50, max_iter=5000, hidden_layer_sizes=(20,20), early_stopping=True, verbose=True)
#svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1, verbose=True)
#rf = RandomForestRegressor(n_estimators=10)

print('Fitting...')
nnr.fit(X_train, y_train.flatten())
pms = nnr.get_params()
print(pms)


params = []
for p in nnr.coefs_:
    params.append(p.flatten())

Np = np.concatenate(params).size
print(Np)

print('Predicting...')
P = nnr.predict(X_train)
P = scaler.inverse_transform(P)
A = scaler.inverse_transform(y_train.flatten())

print(hdt.calculaterootmeansqrerror(P,A))
print(hdt.calculatemeanabserror(P,A))

#plt.plot(A,A, color='black')
#plt.scatter(P,A,color='blue')
#plt.show()

print('Predicting...')
P = nnr.predict(X_test)
P = scaler.inverse_transform(P)
A = scaler.inverse_transform(y_test.flatten())

print('RMSE:',hdt.calculaterootmeansqrerror(P,A))
print('MAE: ',hdt.calculatemeanabserror(P,A))

print('r^2:', metrics.r2_score(A, P, sample_weight=None, multioutput=None))

print('Max/min:',A.max(),A.min())

plt.plot(A,A, color='black')
plt.scatter(P,A,color='blue')
plt.xlabel('BMI Actual')
plt.ylabel('BMI Predicted')
plt.suptitle('Neural network regression test set performance')
plt.title('['+str(Np)+' parameters]')
plt.show()