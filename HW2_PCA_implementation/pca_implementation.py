import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def zeroMean(dataMat):
    meanVal = np.mean(dataMat,axis = 0)     #按列求均值，即求各个特征的均值
    newData = dataMat - meanVal
    return newData, meanVal

def pca(dataMat,n):
    newData,meanVal = zeroMean(dataMat)
    covMat=np.cov(newData,rowvar = 0)           #求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))#求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
    eigValIndice=np.argsort(eigVals)            #对特征值从小到大排序
    n_eigValIndice=eigValIndice[-1:-(n+1):-1]   #最大的n个特征值的下标
    n_eigVect=eigVects[:,n_eigValIndice]        #最大的n个特征值对应的特征向量
    lowDDataMat=newData*n_eigVect               #低维特征空间的数据
    reconMat=(lowDDataMat*n_eigVect.T)+meanVal  #重构数据
    return meanVal,n_eigVect,lowDDataMat,reconMat

f = open(".\optdigits-orig.tra")             # 返回一个文件对象
line = f.readline()             # 调用文件的 readline()方法

idx_line = 0
#rawdata = np.zeros((1,1934*1025))
rawdata = np.array([])
while idx_line < 21:
    idx_line += 1
    line = f.readline()

idx_line = 0

while line:
    line = line.strip('\n')
    line = line.strip(' ')
    content = (list(line))
    content = [int(i) for i in content]
    rawdata = np.concatenate((rawdata,content))
    #print(content)
    idx_line = idx_line + 1
    line = f.readline()

print(idx_line)
rawdata = np.reshape(rawdata,(1934,32 * 32 + 1))
f.close()

data_image = rawdata[:,0 : 32 * 32]
data_class = rawdata[:,32 * 32]
idx_for_3 = np.where(data_class==3)[0]
data_for_3 = data_image[idx_for_3,:]
dataMat = data_for_3
meanvalue,eigenvect,lowDDataMat,reconMat = pca(dataMat,2)
m = np.reshape(meanvalue,(32,32))
v0 = eigenvect[:,0]
v1 = eigenvect[:,1]
v0 = np.reshape(v0,(32,32))
v1 = np.reshape(v1,(32,32))
v0 = v0.real
v1 = v1.real

m = 256-m * 256
v0 = 256-v0 * 256
v1 = 256-v1 * 256

m_im = Image.fromarray(m)
v0_im = Image.fromarray(v0)
v1_im = Image.fromarray(v1)

m_im.show()
v0_im.show()
v1_im.show()



plt.scatter(lowDDataMat[:,0],lowDDataMat[:,1],s=5)
plt.grid()
plt.show()
