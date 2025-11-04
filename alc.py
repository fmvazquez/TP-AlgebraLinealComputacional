import numpy as np

def prodMat(A, B):
    if A.shape[1] != B.shape[0]:
        print("No se puede hacer el producto matricial")
        return
    res = np.zeros((A.shape[0], B.shape[1]))

    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            tot = 0
            for k in range(A.shape[1]):
                tot += A[i, k] * B[k, j]
            res[i, j] = tot

    return res

def aplicTrans(A, v):
    """
    Asume que el vector es columna
    """

    if A.shape[1] != v.shape[0]:
        print("No se puede aplicar la matriz al vector")
        return
    res = np.zeros((A.shape[0]))

    for i in range(res.shape[0]):
        tot = 0            
        for k in range(A.shape[1]):
            tot += A[i, k] * v[k]
        res[i] = tot

    return res

def prodPunto(v, w):
    """
    Asume que el vector es columna
    """

    if v.shape[0] != w.shape[0]:
        print("No se puede calcular el prodcuto punto")
        return
    
    res = 0
    for k in range(v.shape[0]):
        res += v[k] * w[k]

    return res

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
    Ac = A.copy().astype(np.float32)
    
    if M!=N:
        print('Matriz no cuadrada')
        return
    
    for j in range(N):
        piv = Ac[j][j]
        for i in range(j+1, N):
            
            mult = Ac[i][j] / piv
            cant_op+=1
            for k in range(j+1, N):
                Ac[i][k] -= mult * Ac[j][k]
                cant_op+=2
            Ac[i][j] = mult
    
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

# Labo06
def metpot2k(A, tol=1e-15, K=1000):
    """
    A: matriz de tamaño n x n
    tol: tolerancia en la diferencia entre un paso y el siguiente de la estimación del autovector
    K: número máximo de iteraciones a realizar

    Retorna:
    v: autovector estimado
    lambda: autovalor estimado
    k: número de iteraciones realizadas
    """

    N = A.shape[0]

    v = np.random.rand(N)
    v = v / norma(v,2)

    v_monio = aplicTrans(A, v) 
    v_monio = v_monio / norma(v_monio,2)

    e = prodPunto(v_monio, v)
    k = 0
    while abs(e - 1) > tol and k < K:
        v = v_monio

        # Aplicar fA dos veces sobre v
        w = aplicTrans(A, v)
        w = w / norma(w, 2)

        v_monio = aplicTrans(A, w)
        v_monio = v_monio / norma(v_monio, 2)

        e = prodPunto(v_monio, v)
        k += 1
    
    v_monio = np.expand_dims(v_monio, axis=0).T

    lambd = prodMat(prodMat(v_monio.T, A), v_monio)[0][0]
    eps = e-1

    return v_monio, lambd, eps

def diagRH(A, tol=1e-15, K=1000):
    """
    A: Matriz simétrica de tamaño n x n
    tol: Tolerancia en la diferencia entre un paso y el siguiente de la estimación del autovector
    K: Número máximo de iteraciones a realizar

    Retorna:
    - Matriz de autovectores S
    - Matriz de autovalores D tal que A = S D S.T
    
    Si la matriz A no es simétrica, debe retornar None
    """

    if (A.shape[0] != A.shape[1]):
        print("Matriz no cuadrada")
        return
    
    N = A.shape[0]


    v1, lambda1, _ = metpot2k(A, tol, K)
    e1 = np.zeros((N,1))
    e1[0][0] = 1


    factor = 2 / (norma(np.squeeze(e1 - v1),2)**2)
    Hv1 = np.eye(N) - factor * prodMat(e1-v1, (e1-v1).T)
    if N == 2:
        S = Hv1
        D = prodMat(prodMat(Hv1, A), Hv1.T)
    else: 
        B = prodMat(prodMat(Hv1, A), Hv1.T)
        Ahat = B[1:N,1:N]
        Shat, Dhat = diagRH(Ahat, tol=1e-15, K=1000)
        D = np.zeros((N, N))
        D[0][0] = lambda1
        for i in range(1,N):
            for j in range(1,N):
                D[i][j] = Dhat[i-1][j-1]
        Hprod = np.zeros((N, N))
        Hprod[0][0] = 1
        for i in range(1,N):
            for j in range(1,N):
                Hprod[i][j] = Shat[i-1][j-1]
        S = prodMat(Hv1, Hprod)
    
    return S, D