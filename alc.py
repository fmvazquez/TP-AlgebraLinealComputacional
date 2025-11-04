import numpy as np


# Labo03
def norma(x, p):
    sum = 0
    if p == "inf": 
        return np.abs(np.array(x)).max()
    for xi in x:
        sum += np.abs(xi)**p
    return sum ** (1/p)

# Labo04
def calculaLU(A):
    """
    Calcula la factorización LU de la matriz A y retorna las matrices L y U, 
    junto con el número de operaciones realizadas. 

    En caso de que la matriz no pueda factorizarse, retorna None.
    """

    cant_op = 0
    M=A.shape[0]
    N=A.shape[1]
    Ac = A.copy()
    
    if M!=N:
        print('Matriz no cuadrada')
        return
    
    for i in range(N):
        piv = Ac[i][i]
        for j in range(i+1, N):
            
            mult = Ac[j][i] / piv
            cant_op+=1
            for k in range(i, N):
                Ac[j][k] -= mult * Ac[i][k]
                cant_op+=2
            Ac[j][i] = mult

    L = np.eye(N,N)
    U = np.eye(N,N)

    for i in range(N):
        for j in range(N):
            if (i <= j): U[i, j] = Ac[i, j]
            else: L[i, j] = Ac[i, j]

    return L, U, cant_op

def res_tri(L,b,inferior=True):
    """
    Resuelve el sistema Lx = b, donde L es triangular. 

    Se puede indicar si es triangular inferior o superior usando el argumento 
    `inferior` (por defecto se asume que es triangular inferior).
    """
    N = L.shape[0]
    if L.shape[0] != b.shape[0]:
        print("Dimensiones no compatibles en res_tri")
        return
    if L.shape[0] != L.shape[1]:
        print("Matriz no cuadrada en res_tri")
        return
    
    res = b.copy()
    inter_fila = range(N)
    if (not inferior): inter_fila = reversed(inter_fila)

    for i in inter_fila:
        if (inferior): inter_col = range(i+1)
        else: inter_col = range(N-1, i-1, -1)
        for j in inter_col:
            if i == j: res[i] /= L[i, i]
            else: res[i] -= L[i, j] * res[j]    
    return res
