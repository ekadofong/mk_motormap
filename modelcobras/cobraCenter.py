import sys
import numpy as np
from scipy import optimize
import xml.etree.cElementTree as ET
import math


from astropy.io import ascii
import matplotlib.pyplot as plt

def calc_R(x,y, xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) 
    """
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def f(c, x, y):
    """ 
    calculate the algebraic distance between the data points and the mean 
        circle centered at c=(xc, yc) 
    """
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()

def leastsq_circle(x,y):
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    center, ier = optimize.leastsq(f, center_estimate, args=(x,y))
    xc, yc = center
    Ri       = calc_R(x, y, *center)
    R        = Ri.mean()
    residu   = np.sum((Ri - R)**2)
    return xc, yc, R, residu

def findCenter(x, y):
    tbdata = ascii.read(catalog)
    xp=tbdata['col1'].data
    yp=tbdata['col2'].data
    theta=tbdata['col3'].data
    da=abs(np.array([x - y for x, y in zip(theta, theta[1:])]))
    xc1,yc1,rbest1,res1=leastsq_circle(tbdata['col1'].data,tbdata['col2'].data)
    r=np.sqrt((tbdata['col1'].data-xc1)**2+(tbdata['col2'].data-yc1)**2)

    ind=np.where(da > 3)

    xc2,yc2,rbest2,res2=leastsq_circle(tbdata['col1'].data[ind],tbdata['col2'].data[ind])
    
    return xc2, yc2, rbest2

path='/Volumes/Disk/CaltechCobraData/17_12_11_10_07_01_msimCenters/Log/'
file='thetaFwMap_mId_1_pId_3.txt'
catalog=path+file

tbdata = ascii.read(catalog)
xp=tbdata['col1'].data
yp=tbdata['col2'].data
theta=tbdata['col3'].data

substring=file[file.index('_')+1:]
substring=substring[substring.index('_')+1:]
substring=substring[substring.index('_')+1:]
substring=substring[substring.index('_')+1:]

inx=int(substring[0:substring.index('.')])
#print(inx)

xc2, yc2, rbest2=findCenter(xp, yp)

#da=abs(np.array([x - y for x, y in zip(theta, theta[1:])]))

#xc1,yc1,rbest1,res1=leastsq_circle(tbdata['col1'].data,tbdata['col2'].data)
#r=np.sqrt((tbdata['col1'].data-xc1)**2+(tbdata['col2'].data-yc1)**2)

#ind=np.where(da > 3)

#xc2,yc2,rbest2,res2=leastsq_circle(tbdata['col1'].data[ind],tbdata['col2'].data[ind])

xml='/Volumes/Disk/CaltechCobraData/17_12_11_10_07_01_msimCenters/Log/usedXMLFile.xml'
tree = ET.ElementTree(file=xml)
tree.getroot()
root=tree.getroot()

#print(root[0][1][0].text,root[0][1][1].text)

xold=float(root[inx-1][1][0].text)
yold=float(root[inx-1][1][1].text)

print(xold,yold)
print(xc2,yc2)
fig= plt.figure(figsize=(10, 10))
ax = plt.subplot()
plt.scatter(tbdata['col1'].data, tbdata['col2'].data,s=8)
plt.scatter(tbdata['col1'].data[ind], tbdata['col2'].data[ind],s=8,color='red')

rr=1.5*math.floor(rbest2)

plt.plot(xc2,yc2,'ro') 
plt.plot(xold,yold,'x') 
plt.xlim(xc2-rr,xc2+rr)
plt.ylim(yc2-rr,yc2+rr)
c1=plt.Circle((xc1,yc1),rbest1,fill=False,linestyle='--')
c2=plt.Circle((xc2,yc2),rbest2,fill=False)

ax.add_artist(c1)
ax.add_artist(c2)

plt.show()

