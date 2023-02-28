import numpy as np

def poisson(lam,k) -> float:
    """
    because the standard definition of poisson requires the calculation of exponential terms that exceed
    we can rewrite the formula to log space and do our calculations. The result would be ln(P(k))
    using the series representation of the exponential we can regain our original P(x)
    rewriting the equation in log space we get kln(lambda) - ln(k!) - lambda
    We can simplify the ln(k!) form to ln(k)+ ln(k-1) + ln(k-2)... we can rewrite this as np.sum(np.log(np.arange(1, k + 1))))
    """
    k = np.round(k).astype(np.int32) #amount of occurances thus an whole natural number and an integer
    lam = np.float32(lam) #expected value of X (X=k) and thus also the variance of k. Variance is continuous and not discrete thus float
    return np.exp(k * np.log(lam) - lam - np.sum(np.log(np.arange(1, k + 1), dtype=np.float32)), dtype=np.float32)


testlist = np.array([[1,0],[5,10],[3,21],[2.6,40],[101,200]], dtype=np.float32)

for i in testlist:
  print(f"val poisson (k:{i[1]} , lambda:{i[0]}):" ,poisson(i[0],i[1]), type(poisson(i[0],i[1])))

