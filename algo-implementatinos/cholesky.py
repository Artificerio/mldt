import numpy as np

def symmetric(a, rtol=1e-05, atol=1e-08) -> bool:
    assert(a.shape[0] == a.shape[1]), "Not a square matrix"
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def positive_definite(a) -> bool:
    return np.all(np.linalg.eigvals(a) > 0)

def cholesky(a) -> np.array:
    """Performs Cholesky decomposition of a matrix which must be
    a symmetric and positive definite matrix. The function returns 
    the lower variant triangular matrix L"""

    a = np.array(a, float)
    L = np.zeros_like(a)
    n, _ = np.shape(a)

    if (not symmetric(a) and not positive_definite(a)):
        raise Exception("Input Matrix should be symmetric and positive-definite")

    for j in range(n):
        for i in range(j,n):
            if i == j:
                L[i,j] = np.sqrt(a[i,j] - np.sum(L[i,:j])**2)
            else:
                L[i,j] = (a[i,j] - np.sum(L[i,:j] * L[j, :j])) / L[j,j]
    return L

H = [[3.2,3,0.5,1,2],
     [3,6.3,-2,4,0],
     [0.5,-2,8,-3.1,3],
     [1,4,-3.1,7.6,2.6],
     [2,0,3,2.6,15]]

A = np.array([[1,2],[2,3]], dtype=float)

H = np.array(H, dtype=float)
print(cholesky(A))