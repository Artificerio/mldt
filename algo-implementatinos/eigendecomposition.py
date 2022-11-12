import numpy as np

A = np.array([[5,-6,-6],[-1,4,2],[3,-6,-4]], dtype=int)

D,P = np.linalg.eig(A)
P_T = np.linalg.inv(P)
print("Eigendecomposition of A in orthonormal basis")
print("Basis transformation matrix P\n", P)
print("Scaling matrix\n", D)
print("Backward basis transformation matrix\n", P_T)
