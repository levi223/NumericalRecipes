import numpy as np
import matplotlib.pyplot as plt
from VandermondeMat import VDM
from linsolve import *
from Interpolate import *
import time as time

def lagrangepolynomial(x, coeff):
    lagrangepoly = 0
    for num, ele in enumerate(coeff):
        lagrangepoly += ele * x **num
    return lagrangepoly

timeititerN = 1000

#import data
vander = VDM("Vandermonde.txt")
vander = VDM("Vandermonde.txt")
vandersolve=  LinalgSolve(vander.A, vander.y)
x_eval_points = np.linspace(min(vander.x),max(vander.x),1000)

#measure runtime - interpolated singularly iterated LU decomp
Crout1xruntime = []
for i in range(timeititerN):
    start= time.time()
    solutionCrout = vandersolve.CroutSolve()
    interpolated_values_LU = lagrangepolynomial(x_eval_points,solutionCrout) #Lu values
    Crout1xruntime.append(time.time() - start)
    interpolated_values_LU_error = np.subtract(lagrangepolynomial(vander.x,solutionCrout),vander.y) #difference in values
print("Average time to run LU decomp interpolater 1x: ",np.average(Crout1xruntime))


#measure runtime -  interpolated 10x iterated LU decomp
Crout10xruntime =[]
for i in range(timeititerN):
    start= time.time()
    solutionitertaiveCrout = vandersolve.CroutSolveIterative(iterations=10)
    interpolated_values_iterative_LU =lagrangepolynomial(x_eval_points,solutionitertaiveCrout)
    Crout10xruntime.append(time.time() - start)
    interpolated_values_iterative_LU_error = np.subtract(lagrangepolynomial(vander.x,solutionCrout),vander.y) #difference in values
print("Average time to run LU decomp interpolater 10x: ",np.average(Crout10xruntime))



#measure runtime - Neville interpolation
Nevilleruntime = []
for i in range(timeititerN):
    start=  time.time()
    interpolated_values_Neville = [interpol(vander.x, vander.y).interpolate(i,method="neville") for i in x_eval_points]
    interpolated_values_Neville_error = np.subtract([interpol(vander.x, vander.y).interpolate(i,method="neville") for i in vander.x],vander.y) #difference in values
    Nevilleruntime.append(time.time() - start)
print("Time to run Neville interpolater: ", np.average(Nevilleruntime))


#plot data
fig,ax=plt.subplots(1,2, figsize=(10, 5))

ax[0].scatter(x_eval_points,interpolated_values_Neville,marker='p',linewidth=0, s=3, label = "neville")
ax[0].scatter(x_eval_points,interpolated_values_LU,marker='D',linewidth=0, s=3, label = "Iterative LU 1x")
ax[0].scatter(vander.x,vander.y,marker='o',linewidth=0,label="original data")
ax[0].scatter(x_eval_points,interpolated_values_iterative_LU,marker="v", s=1, label="Iterative LU 10x")
ax[0].grid()
ax[0].legend()
ax[0].set_xlabel('$x$')
ax[0].set_ylabel('$y$')
ax[0].set_ylim(-100,175)
ax[0].set_title("Interpolated values")


#ax[1].scatter(x_eval_points,interpolated_values_Neville,marker='o',linewidth=0, s=5, label = "neville")
ax[1].scatter(vander.x,interpolated_values_LU_error,marker='D',linewidth=0,  label = "Iterative LU 1x error")
ax[1].scatter(vander.x,interpolated_values_Neville_error,marker='p',linewidth=0, label = "Neville error")
ax[1].scatter(vander.x,interpolated_values_iterative_LU_error,marker='v',linewidth=0, label = "Iterative LU 10x error ")
ax[1].grid()
ax[1].legend()
ax[1].set_xlabel('$x$')
ax[1].set_ylabel('$y error$')
ax[1].set_title("Interpolation Error")
plt.show()


