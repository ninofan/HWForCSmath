import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.stats import norm
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

def point_prediction(num_sample,d,value_regularization):
    x = np.arange(0, 1, 1/num_sample)
    y = norm.rvs(0, size=num_sample, scale=0.1)
    z = np.arange(0, 1, 1/1000)
    plt.plot(z,np.sin(2*sp.pi*z) , linewidth=2, color = 'red')
    y = norm.rvs(0,0.1,size = num_sample)
    y = y + np.sin(2*sp.pi*x)
    plt.scatter(x, y, s=5)
    if value_regularization == 0:
        clf = Pipeline([('poly', PolynomialFeatures(degree=d)),
                    ('linear', LinearRegression(fit_intercept=False))])
    else:
        clf = Pipeline([('poly', PolynomialFeatures(degree=d)),
                    ('linear',Ridge(fit_intercept=False,alpha = value_regularization/2/num_sample))])
    clf.fit(x[:,np.newaxis],y)
    y_test = clf.predict(x[:, np.newaxis])
    plt.plot(x, y_test, linewidth=2)
    plt.grid()
    if value_regularization == 0:
        plt.legend(['original curve','fitting curve'], loc='upper right')
    else:
        plt.legend(['original curve','fitting curve'+'\nln(lambda)=-18'], loc='upper right')
plt.figure(1)
point_prediction(10,3,0)
plt.figure(2)
point_prediction(10,9,0)
plt.figure(3)
point_prediction(15,9,0)
plt.figure(4)
point_prediction(100,9,0)
plt.figure(5)
point_prediction(10,9,1.5230e-08)
plt.show()
