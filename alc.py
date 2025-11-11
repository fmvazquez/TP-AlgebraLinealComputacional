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
    
    return np.sum(v * w)

# Labo01
def error(x,y):
    """
    Recibe dos numeros x e y, y calcula el error de aproximar x usando y en float64
    """
    return np.abs(np.float64(x) - np.float64(y))

# def matricesIguales(A,B):
#     """    
#     Devuelve True si ambas matrices son iguales y False en otro caso.
#     """
#     A = np.array(A)
#     B = np.array(B)

#     if A.shape != B.shape: return False
#     for i in range(A.shape[0]):
#         for j in range(A.shape[1]):
#             if error(A[i][j], B[i][j]) > np.finfo(np.float64).eps: return False
#     return True

# Labo03
# BIEN
def norma(x, p):
    suma = 0
    if p == "inf": 
        return np.abs(np.array(x)).max()
    xabs = np.abs(x)
    suma = np.sum(xabs**p)
    
    return suma ** (1/p)

def normaliza(X):
    for i in range(X.shape[0]):
        normav = norma(X[i], p=2)
        if (normav == 0): continue
        X[i] /= normav
    return X

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

def QR_con_GS(A):
    Q = np.zeros(A.shape)
    R = np.zeros((A.shape[1],A.shape[1])) #cambio: el tamaño de R es cols A x cols A, no es el de A.
    At_norm = normaliza(A.T)
    for j in range (Q.shape[0]):
        Q[j,0] = At_norm[0][j]

    print("vectores normalizados")
    R[0,0] = norma(A.T[0], p=2)

    for j in range (1, Q.shape[1]): #cambio: acá itero sobre columnas de Q (Q.shape[1]), no sobre filas (Q.shape[0]).
        print(j)
        q = A[:, j]
        for k in range (0, j):
            R[k,j] = prodPunto((Q[:,k].T), q)
            q = q - R[k,j]*Q[:,k]
        R[j,j] = norma(q, p=2)
        Q[:,j] = normaliza(np.expand_dims(q, axis=0))
    print("Q shape:", Q.shape)
    print("R shape:", R.shape)

    return Q, R

def QR_con_HH(A, tol=1e-12):
    A = A.astype(float)
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)

    for k in range(n):
        x = R[k:, k]
        e = np.zeros_like(x)
        e[0] = np.sign(x[0]) * norma(x,p=2) if x[0] != 0 else norma(x)
        u = x - e
        if norma(u,p=2) < tol:
            continue
        u = u / norma(u,p=2)
        H = np.eye(m - k) - 2 * np.outer(u, u)
        H_moño = np.eye(m)
        H_moño[k:, k:] = H
        R = H_moño @ R
        Q = Q @ H_moño

    return Q, R

def calculaCholesky(A):
    """
    Calcula la factorización de Cholesky A = L * L.T
    usando la factorización LU -> LDU, asumiendo que A es simétrica definida positiva.
    
    Retorna L, D, U (donde U = L.T).
    """

    # Verificar si es SDP

    A = np.array(A, dtype=np.float64)

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
            U[:, i] = avec[i] / norma(avec[i], p=2)
            V_moño = multMat(A.T,U[:, i])
            V[:, i] = V_moño / aval[i]
    else:
        for i in range(r):
            V[:, i] = avec[i] / norma(avec[i], p=2)
            U_moño = multMat(A,V[:, i])
            U[:,i] = U_moño / aval[i]
    
    
    return U, S, V
 
# NUEVAS 

def cargarDataset(carpeta):
    train_cats = np.load(f'{carpeta}/train/cats/efficientnet_b3_embeddings.npy')
    train_dogs = np.load(f'{carpeta}/train/dogs/efficientnet_b3_embeddings.npy')

    val_cats = np.load(f'{carpeta}/val/cats/efficientnet_b3_embeddings.npy')
    val_dogs = np.load(f'{carpeta}/val/dogs/efficientnet_b3_embeddings.npy')

    X_train = np.zeros((train_cats.shape[0], train_cats.shape[1] + train_dogs.shape[1]))
    X_train[:, :train_cats.shape[1]] = train_cats
    X_train[:, train_cats.shape[1]:] = train_dogs

    X_val = np.zeros((val_cats.shape[0], val_cats.shape[1] + val_dogs.shape[1]))
    X_val[:, :val_cats.shape[1]] = val_cats
    X_val[:, val_cats.shape[1]:] = val_dogs

    Y_train = np.zeros((2, train_cats.shape[1] + train_dogs.shape[1]))  # ✓ (2, 3000)
    Y_train[0, :train_cats.shape[1]] = 1  # Primera fila = gatos [1,0]
    Y_train[1, train_cats.shape[1]:] = 1  # Segunda fila = perros [0,1]

    Y_val = np.zeros((2, val_cats.shape[1] + val_dogs.shape[1]))  # ✓ (2, 2000)
    Y_val[0, :val_cats.shape[1]] = 1   # Primera fila = gatos
    Y_val[1, val_cats.shape[1]:] = 1   # Segunda fila = perros

    return X_train, Y_train, X_val, Y_val


def matricesIguales(A, B, tol=None):
    """    
    Devuelve True si ambas matrices son iguales y False en otro caso.
    """
    A = np.array(A)
    B = np.array(B)
    
    # Si no se especifica tolerancia, usar eps
    if tol is None:
        tol = np.finfo(np.float64).eps

    if A.shape != B.shape: return False
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if error(A[i][j], B[i][j]) > tol: return False
    return True

def esPseudoInversa(X, pX, tol=1e-8):
    """
    Cheque que se cumplan las las cuatro condiciones de Moore-Penrose
    1. A @ A⁺ @ A = A
    2. A⁺ @ A @ A⁺ = A⁺
    3. (A @ A⁺)* = A @ A⁺
    4. (A⁺ @ A)* = A⁺ @ A
    """

    # # (1)
    # cond1 = matricesIguales(prodMat(X, prodMat(pX, X)), X)
    # if not cond1: return False 
    # # (2)
    # cond2 = matricesIguales(prodMat(pX, prodMat(X, pX)), pX)
    # if not cond2: return False 
    # # (3)
    # cond3 = matricesIguales(prodMat(X, pX).T, prodMat(X, pX))
    # if not cond3: return False 
    # # (4)
    # cond4 =  matricesIguales(prodMat(pX, X).T, prodMat(pX, X))
    # if not cond4: return False 

    # (1)
    cond1 = matricesIguales(X @ pX @ X, X, tol=tol)
    if not cond1: return False 
    
    # (2)
    cond2 = matricesIguales(pX @ X @ pX, pX, tol=tol)
    if not cond2: return False 
    
    # (3)
    cond3 = matricesIguales((X @ pX).T, X @ pX, tol=tol)
    if not cond3: return False 
    
    # (4)
    cond4 = matricesIguales((pX @ X).T, pX @ X, tol=tol)
    if not cond4: return False

    return True


def fullyConectedCholesky(X, Y):
    N, M = X.shape
    if N == M:
        pX = inversa(X)
        W = prodMat(Y, pX)
    else:
        if N > M:
            print("Caso (a) N > M")
            Xh = prodMat(X.T, X) 
            B = X.T
        else:
            print("Caso (b) N < M")
            Xh = prodMat(X, X.T)
            B = X
        print("Calculo de Cholesky")
        L = calculaCholesky(Xh)

        # En el caso (a) me queda L L.T U = B -> L V = B -> L.T V = B
        # En el caso (b) me queda L L.T U.T = B -> L V = B -> L.T V.T = B
        # Depende el caso tengo que transponer o no U
        print("Resolviendo sistemas triangulares .1")
        V = res_tri_matricial(L, B, inferior=True)
        print("Resolviendo sistemas triangulares .2")
        U = res_tri_matricial(L.T, V, inferior=False)

        if N > M: pX = U
        else: pX = U.T

        print("Calculando W")
        W = prodMat(Y, pX)

    return W, pX

def pesosConQR(X, Y, metodo):
  if metodo == "GS":
    print("Calcular QR")
    Q, R = QR_con_GS(X.T)
  elif metodo == "HH":
    Q, R = QR_con_HH(X.T)
  else: 
    print("Metodo incorrecto")
    return

  #Se tiene que VR^T=Q. Por tanto, el núm. de columnas de V es el de filas de R^T. Y su núm. de filas es el de filas de Q.
  V = np.zeros((Q.shape[0],R.T.shape[0]))
  #Además, RV^T = Q^T.
  V = V.T
  print("Resolviendo sistema")
  for j in range(Q.T.shape[1]):
    V[:,j] = res_tri(R, (Q.T)[:, j], inferior=False)
  V = V.T
 
  W = Y@V
  return W, V