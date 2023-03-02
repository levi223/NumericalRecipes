import numpy as np

class LinalgSolve():
    
    def croutdecomp(self,A):
        #initialize U and L
        n = np.shape(A)[0]
        L = np.zeros((n,n))
        U = np.eye(n)
        A = np.array(A)
        
        for j in range(0,n): #loop over columns but not the first one
            #computing the lower value L# loop over all rows below the diagonal
            sum =  L[j:,:j] @ U [:j,j] #calculating the
            L[j:,j] = A[j:,j] - sum

            #computing the upper value U
            sum = L[j,:j] @ U [:j,j:] 
            U[j][j:] = (A[j][j:] - sum) / L[j][j]

        return np.array(L), np.array(U)
    
    def __init__(self, A, b) -> None:
        self.A = np.array(A)
        self.b = np.array(b)
        self.n = len(self.A)
        self.I = np.eye(self.n)
        self.L, self.U = self.croutdecomp(self.A)

    
    def forward_sub(self,L,b):
        n = len(b)
        y = np.array(b).copy()
        for i in range(n): # iterate over common shape
                 # setting new array values to be same as b
            for j in range(i): # iterating over all values under the 
                y[i]=y[i]-(L[i,j]*y[j])
            y[i]=y[i]/L[i,i]

        return y

    def backward_sub(self, U, y):
        n = len(y)
        x = np.zeros(n)
        for i in range(n-1, -1, -1): #traverse matrix in reverse
            sum = 0
            for j in range(i+1, n): #iterate over rows excluding the diagonal
                sum += U[i][j] * x[j]
            x[i] = (y[i] - sum) / U[i][i]

        return x
    
    
    def CroutSolve(self):
        Y = self.forward_sub(self.L, self.b)
        X = self.backward_sub(self.U,Y)
        return X
    
    def CroutSolveIterative(self, iterations):
        solution = self.CroutSolve()
        b = self.b.copy()
        for i in range(1,iterations):
            db = self.A @ solution - b
            intermediate = self.forward_sub(self.L, db)
            dx = self.backward_sub(self.U,intermediate)
            solution -=dx
        return solution
           
            
        