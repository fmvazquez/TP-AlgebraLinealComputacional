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

# Labo01
def error(x,y):
    """
    Recibe dos numeros x e y, y calcula el error de aproximar x usando y en float64
    """
    return np.abs(np.float64(x) - np.float64(y))

def matricesIguales(A,B):
    """    
    Devuelve True si ambas matrices son iguales y False en otro caso.
    """
    A = np.array(A)
    B = np.array(B)

    if A.shape != B.shape: return False
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if error(A[i][j], B[i][j]) > np.finfo(np.float64).eps: return False
    return True

# Labo03
# BIEN
def norma(x, p):
    sum = 0
    if p == "inf": 
        return np.abs(np.array(x)).max()
    for xi in x:
        sum += np.abs(xi)**p
    return sum ** (1/p)

# Labo04
def calculaLU(A):
    # BIEN SEGUN TESTS NUESTROS
    """
    Calcula la factorización LU de la matriz A y retorna las matrices L y U, 
    junto con el número de operaciones realizadas. 

    En caso de que la matriz no pueda factorizarse, retorna None.
    """
    
    cant_op = 0
    M=A.shape[0]
    N=A.shape[1]
    Ac = A.copy().astype(np.float64)
    
    if M!=N:
        return None
    
    finfo = np.finfo(Ac.dtype)
    
    for j in range(N):
        piv = Ac[j][j]
        if abs(piv) < finfo.eps:  # pivote nulo o casi nulo
            return None
        for i in range(j+1, N):
            mult = Ac[i][j] / piv
            cant_op += 1
            for k in range(j+1, N):
                Ac[i][k] -= mult * Ac[j][k]
                cant_op += 2
            Ac[i][j] = mult
    
    L = np.eye(N,N)
    U = np.eye(N,N)

    for i in range(N):
        for j in range(N):
            if (i <= j): U[i, j] = Ac[i, j]
            else: L[i, j] = Ac[i, j]

    return L, U, cant_op

# Funcion auxiliar para res_tri
def suma(A, X, i, inferior): 
    res = 0
    n = A.shape[0]
    if inferior:
        for j in range(i):
            res += A[i,j]*X[j]
    else:
        for j in range(i+1, n):
            res += A[i,j]*X[j]
    return res

def res_tri(A, b, inferior=True): 
    """
    Resuelve el sistema Lx = b, donde L es triangular. 

    Se puede indicar si es triangular inferior o superior usando el argumento 
    `inferior` (por defecto se asume que es triangular inferior).
    """
    X = np.zeros(b.shape)
    n = X.shape[0]
    if inferior:
        for i in range(n):
            X[i] = (b[i] - suma(A, X, i, inferior=True)) / A[i,i]
    else:
        for i in range(n-1, -1, -1):
            X[i] = (b[i] - suma(A, X, i, inferior=False)) / A[i,i]
    return X

def res_tri_matricial(A, B, inferior=True): 
    """
    Resuelve el sistema L X = B, donde L es triangular. 

    Se puede indicar si es triangular inferior o superior usando el argumento 
    `inferior` (por defecto se asume que es triangular inferior).
    """
    N, M = B.shape
    X = np.zeros((N, M))
    
    # Resolver para cada columna de B independientemente
    for j in range(M):
        X[:, j] = res_tri(A, B[:, j], inferior=inferior)
    
    return X

def inversa(A):
    L, U, _ = calculaLU(A)
    A_inv = np.zeros(A.shape)
    I = np.eye(A.shape[0])

    for i in range(len(A)):
        y = res_tri(L, I[:, i], inferior=True) # Ly = I
        x = res_tri(U, y, inferior=False) # Ux = y
        A_inv[:, i] = x
    return A_inv

def calculaCholesky(A):
    """
    Calcula la factorización de Cholesky A = L * L.T
    usando la factorización LU -> LDU, asumiendo que A es simétrica definida positiva.
    
    Retorna L, D, U (donde U = L.T).
    """

    # Verificar si es SDP

    A = np.array(A, dtype=np.float32)

    # Factorización LU
    lu_res = calculaLU(A)

    L, U, _ = lu_res

    # Armamos la diagonal D
    N = A.shape[0]
    D = np.zeros((N, N))
    for i in range(N):
        D[i, i] = U[i, i]
    
    L_chol = np.zeros(L.shape)
    for i in range(N):
        for j in range(i+1):
            L_chol[i, j] = L[i, j] * np.sqrt(D[j, j])

    return L_chol


# Labo06

def metpot2k(A, tol=1e-15, K=1000):
    N = A.shape[0]

    # Caso matriz nula → todos los autovalores son 0
    if norma(A, 2) < tol:
        v = np.random.rand(N)
        v = v / norma(v, 2)
        v = np.expand_dims(v, axis=0).T
        return v, 0.0, 0

    # vector inicial
    v = np.random.rand(N)
    v = v / norma(v, 2)

    v_monio = aplicTrans(A, v)
    v_monio = v_monio / norma(v_monio, 2)

    e = prodPunto(v_monio, v)
    k = 0
    stuck = 0

    while abs(e - 1) > tol and k < K:

        v = v_monio

        # doble aplicación
        w = aplicTrans(A, v)
        w = w / norma(w, 2)

        v_monio = aplicTrans(A, w)
        v_monio = v_monio / norma(v_monio, 2)

        e = prodPunto(v_monio, v)
        k += 1

        # Si está oscilando por autovalores iguales → reinicio
        if abs(e) < 0.05:
            stuck += 1
            if stuck > 5:
                v = np.random.rand(N)
                v = v / norma(v, 2)
                stuck = 0

    v_monio = np.expand_dims(v_monio, axis=0).T
    lambd = prodMat(prodMat(v_monio.T, A), v_monio)[0][0]

    return v_monio, lambd, e-1


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

    if A.shape[0] != A.shape[1]:
        print("Matriz no cuadrada")
        return None

    N = A.shape[0]

    # 1x1 → trivial
    if N == 1:
        return np.eye(1), A.copy()

    # autovector dominante
    v1, lambda1, _ = metpot2k(A, tol, K)

    # e1
    e1 = np.zeros((N,1))
    e1[0][0] = 1.0

    # vector de Householder
    u = e1 - v1
    nu = norma(np.squeeze(u), 2)

    # si v1 ≈ e1 → no hacer Householder (H=I)
    if nu < 1e-12:
        Hv1 = np.eye(N)
    else:
        factor = 2.0 / (nu * nu)
        Hv1 = np.eye(N) - factor * prodMat(u, u.T)

    # Caso N=2 → ya diagonaliza
    if N == 2:
        S = Hv1
        D = prodMat(prodMat(Hv1, A), Hv1.T)
        return S, D

    # Construir B = H A H^T
    B = prodMat(prodMat(Hv1, A), Hv1.T)

    # Submatriz sin la primera fila/col
    Ahat = B[1:N, 1:N]

    # Recurrencia
    Shat, Dhat = diagRH(Ahat, tol, K)

    # Construyo D grande
    D = np.zeros((N, N))
    D[0][0] = lambda1
    for i in range(1, N):
        for j in range(1, N):
            D[i][j] = Dhat[i-1][j-1]

    # Extiendo Shat a N×N
    Hprod = np.zeros((N, N))
    Hprod[0][0] = 1.0
    for i in range(1, N):
        for j in range(1, N):
            Hprod[i][j] = Shat[i-1][j-1]

    # matriz de autovectores final
    S = prodMat(Hv1, Hprod)

    return S, D


#labo08

def multMat(A, x):
    A = np.array(A)
    x = np.array(x)
    res = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        tot = 0
        for k in range(A.shape[1]):
            tot += A[i, k] * x[k]
        res[i] = tot
    return res


def svd_reducida (A, k ='max' , tol=1e-15):
    n = A.shape[0]
    m = A.shape[1]
    
    if n < m: 
        M = prodMat(A, A.T)
    else:
        M = prodMat(A.T, A)
    S, D = diagRH(M, tol)
    
    aval = []
    avec = []
    for i in range (D.shape[0]):
        if D[i,i] > tol:
            aval.append(np.sqrt(D[i,i]))
            avec.append(S[:, i])
    
    # Aplica k si tiene limite
    if k != 'max':
        aval = aval[:k]
        avec = avec[:k]
    
    
    r = len(aval)
    S = np.array(aval)
    U = np.zeros((n, r))
    V = np.zeros((m,r))
    
    if n < m:
        for i in range(r):
            U[:, i] = avec[i] / norma(avec[i])
            V_moño = multMat(A.T,U[:, i])
            V[:, i] = V_moño / aval[i]
    else:
        for i in range(r):
            V[:, i] = avec[i] / norma(avec[i])
            U_moño = multMat(A,V[:, i])
            U[:,i] = U_moño / aval[i]
    
    
    return U, S, V
 
# NUEVAS 

def fullyConectedCholesky(X, Y):
    N, M = X.shape
    if N == M:
        W = prodMat(Y, inversa(X))
    else:
        if N > M:
            Xh = prodMat(X.T, X)
            B = X.T
        else:
            Xh = prodMat(X, X.T)
            B = X.T
        L = calculaCholesky(Xh)

        # En el caso (a) me queda L L.T U = B -> L V = B -> L.T V = B
        # En el caso (b) me queda L L.T U.T = B -> L V = B -> L.T V.T = B
        # Depende el caso tengo que transponer o no U

        V = res_tri_matricial(L, B)
        U = res_tri_matricial(L.T, V)

        if N > M: W = U
        else: W = U.T

    return W