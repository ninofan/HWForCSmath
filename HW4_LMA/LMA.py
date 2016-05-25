from scipy import *
import numpy as np
import theano.tensor as T
import theano
from scipy.linalg import solve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

x = T.dvector('x')

# define the objective function to optimized
y = x ** 2
y = y.sum()
fig = plt.figure()
ax = Axes3D(fig)
x1 = np.arange(-2,2,0.1)
y1 = np.arange(-2,2,0.1)
x1, y1 = np.meshgrid(x1, y1)
z = -x1**2 - y1**2
ax.plot_surface(x1, y1, z, rstride=1, cstride=1, color = 'grey')




# define the gradient function and the Hessen matrix function
gy = T.grad(y, x)
hy, updates = theano.scan(lambda i, gy,x : T.grad(gy[i], x), sequences=T.arange(gy.shape[0]), non_sequences=[gy, x])
fy = theano.function([x], y)
gy = theano.function([x], gy)
hy = theano.function([x], hy)

# the initial point and miu
xk = np.array([1.8,-1.8])
orbitx = np.array([1.8,-1.8])
I = np.eye(2)
miu = 1

gk = gy(xk)
Gk = hy(xk)


while np.linalg.norm(gk) >= 1e-8:
    Gk = Gk + miu * I
    while is_pos_def(Gk) == 0:
        miu = 4 * miu
        Gk = Gk + miu * I
    s = solve(Gk,-gk)
    deltaf = fy(xk + s) - fy(xk)
    deltaq = dot(gy(xk + s),s) + 0.5 * dot(dot((s),Gk),np.transpose((s)))
    rk = deltaf / deltaq
    if rk > 0.75:
        miu = miu / 2
    elif rk < 0.25:
        miu = miu * 4
    if rk > 0:
        xk = xk + s
    gk = gy(xk)
    Gk = hy(xk)
    orbitx = np.append(orbitx,xk,axis=0)

size_orbit = orbitx.size
orbitx = np.reshape(orbitx,[size_orbit/2,2])
orbity = orbitx[:,0]**2 + orbitx[:,1]**2
plt.plot(orbitx[:,0],orbitx[:,1],-orbity,color = 'r')

print(xk)
print(fy(xk))
plt.show()









