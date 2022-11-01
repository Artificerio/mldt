# Cholesky decomposition 

import numpy as np

def cholesky(a):
    """Performs Cholesky decomposition of a matrix which must be
    a symmetric and positive definite matrix. The function returns 
    the lower variant triangular matrix L"""
    a = np.array(a, float)
    L = np.zeros_like(a)
    n, _ = np.shape(a) # since a is square, enough to know number of columns
    for j in range(n):
        for i in range(j,n):
            if i == j:
                L[i,j] = np.sqrt(a[i,j] - np.sum(L[i,:j]) ** 2)
            else:
                L[i,j] = (a[i,j] - np.sum(L[i,:j] * L[j, :j])) / L[j,j]
    return L


H = [[5.2,3,0.5,1,2],
     [3,6.3,-2,4,0],
     [0.5,-2,8,-3.1,3],
     [1,4,-3.1,7.6,2.6],
     [2,0,3,2.6,15]]


L = cholesky(H)
print("Cholesky decomposition of matrix H is: ")
print(L)