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
    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()

    """
    Calcula la factorización LU de la matriz A y retorna las matrices L y U, 
    junto con el número de operaciones realizadas. 

    En caso de que la matriz no pueda factorizarse, retorna None.
    """
    
    if m!=n:
        print('Matriz no cuadrada')
        return
    
    L = np.zeros((m,m))
    U = np.zeros((m,m))
    for k in range (m):
        pivot = Ac[k,k]
        if pivot == 0: return None, None, 0
        L[k,k] = 1
        U[k,k] = Ac[k,k]
        for i in range(k+1,m):
            Ac[i,k] = Ac[i,k]/pivot
            L[i,k] = Ac[i,k]
            U[i,k] = 0
            cant_op += 1
            for j in range (k+1,m):
                if i == 1: U[i-1,j] = Ac[i-1,j]
                Ac[i,j] = Ac[i,j] - Ac[i,k]*Ac[k,j]
                U[i,j] = Ac[i,j]
                cant_op += 2
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
