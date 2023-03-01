import numpy as np
import os 
import sys

class VDM():
    
    def __init__(self, file="Vandermonde.txt") -> None:
        self.data=np.genfromtxt(os.path.join(sys.path[0],"Vandermonde.txt"),comments='#',dtype=np.float64)
        self.x = self.data[:,0]
        self.y = self.data[:,1]
        self.xx = np.linspace(self.x[0],self.x[-1],1001) #x values to interpolate at
        self.A = np.vander(self.x, increasing=True) #ask question if allowed
        self.b = self.y
    
    


