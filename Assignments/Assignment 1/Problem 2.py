import numpy as np
import matplotlib.pyplot as plt
from VandermondeMat import VDM
from linsolve import *
from Interpolate import *
import time

def lagrangepolynomial(x, coeff):
    lagrangepoly = 0
    for num, ele in enumerate(coeff):
        lagrangepoly += ele * x **num
    return lagrangepoly


#import data
vander = VDM("Vandermonde.txt")

#calculate interpolated values using LU
vander = VDM("Vandermonde.txt")
vandersolve=  LinalgSolve(vander.A, vander.y)
solution = vandersolve.CroutSolve()
x_eval_points = np.linspace(min(vander.x),max(vander.x),1000)


interpolated_values_LU = lagrangepolynomial(x_eval_points,solution) #Lu values
interpolated_values_LU_error = np.subtract(lagrangepolynomial(vander.x,solution),vander.y) #difference in values


#calculate interpolated values using Neville

interpolated_values_Neville = [interpol(vander.x, vander.y).interpolate(i,method="neville") for i in x_eval_points]
interpolated_values_Neville_error = np.subtract([interpol(vander.x, vander.y).interpolate(i,method="neville") for i in vander.x],vander.y) #difference in values


#plot data
fig,ax=plt.subplots(1,2, figsize=(10, 5))

ax[0].scatter(x_eval_points,interpolated_values_Neville,marker='o',linewidth=0, s=3, label = "neville")
ax[0].scatter(x_eval_points,interpolated_values_LU,marker='o',linewidth=0, s=3, label = "LU")
ax[0].scatter(vander.x,vander.y,marker='o',linewidth=0,label="original data")
ax[0].grid()
ax[0].legend()
ax[0].set_xlabel('$x$')
ax[0].set_ylabel('$y$')
ax[0].set_ylim(-100,175)
ax[0].set_title("Interpolated values")


#ax[1].scatter(x_eval_points,interpolated_values_Neville,marker='o',linewidth=0, s=5, label = "neville")
ax[1].scatter(vander.x,interpolated_values_LU_error,marker='o',linewidth=0,  label = "LU error")
ax[1].scatter(vander.x,interpolated_values_Neville_error,marker='o',linewidth=0, label = "Neville error")
ax[1].grid()
ax[1].legend()
ax[1].set_xlabel('$x$')
ax[1].set_ylabel('$y error$')
ax[1].set_title("Interpolation Error")

plt.show()


for i in range(20):
    start= time.time()
    vandersolve.CroutSolve()
    print(time.time()- start )
