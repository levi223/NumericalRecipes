import numpy as np

class interpol(): 
    def __init__(self,x,y) -> None:
        self.x = np.array(x) 
        self.y= np.array(y)

    def neville(self,x,neighbours) -> float:
        xvalues =self.x[neighbours]
        n = len(xvalues)
        
        #calculation
        pvals = self.y.copy()[neighbours]  
        for j in range(1, n):
            for i in range(n - j):
                pvals[i] = ((x - xvalues[i + j]) * pvals[i] + (xvalues[i] - x) * pvals[i + 1]) / (xvalues[i] - xvalues[i + j])
        return pvals[0]
    
    def neighbours(self, x, M) -> list:
        slicing_list = self.x 
        index_list = [index for index in range(0,len(self.x))]
        list_len= len(slicing_list)
        while list_len > 1: 
            middle_index = int(np.math.floor(list_len/2))
            #update condition
            if x > slicing_list[middle_index]:
                slicing_list = slicing_list[middle_index:]
                index_list = index_list[middle_index:]
            else: 
                slicing_list = slicing_list[:middle_index]
                index_list = index_list[:middle_index]
            list_len = len(slicing_list)
        finallist = list(range(int(np.ceil(index_list[0]-M/2)+1), int(np.floor(index_list[0]+M/2)+1)))
        
        if max(finallist) >= len(self.x):
            finallist = np.array(finallist) - (max(finallist)-len(self.x))-1
            
        if min(finallist) < 0:
            finallist = np.array(finallist) + min(finallist)
    
        return finallist
        
        
    def interpolate(self, x, M=4, method= "neville") -> float:
        #neville
        if method == "neville":
            neighbours = self.neighbours(x,M)
            try: 
                return self.neville(x, neighbours)
            except (IndexError, ValueError) as e:
                print("Overflow error happened")
                print(f"{e}, {e.__class__}")
        