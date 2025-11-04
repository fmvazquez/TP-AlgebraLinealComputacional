from alc import calculaLU
import numpy as np

def calculaCholesky(A,atol=1e-10):
    N = A.shape[0]    
    res = calculaLU(A)
    if (res == None):
        print("No se puede hacer Cholesky, no existe desc LU")
        return
    L, U, _ = res
    D = np.zeros((N,N))
    for i in range(N):
        D[i,i] = U[i,i]
        for j in range(i, N):
            U[i,j] = U[i,j] / U[i,i]

    

A = np.array([
    [25,15,-5],
    [15,18,0],
    [-5,0,11]
])

calculaCholesky(A)



