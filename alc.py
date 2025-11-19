import numpy as np
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(2000)

def prodMat(A, B):
    """
    Producto matricial entre A y B
    Recibe: A: matriz de tamaño (m x n)
            B: matriz de tamaño (n x p)
    Devuelve: matriz resultado de tamaño (m x p)
    """
    if A.shape[1] != B.shape[0]:   #Condiciones de tamaño para poder hacer el producto matricial
        print("No se puede hacer el producto matricial")
        return
    #Aseguramos consistencia en el tipo de datos interno de A y B:
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    #Inicializamos en 0 la matriz producto:
    res = np.zeros((A.shape[0], B.shape[1]), dtype=np.float64)

    #Cada posición (i,j) de res es el producto escalar entre la fila i de A y la col. j de B:
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i, j] = np.sum(A[i, :] * B[:, j])

    return res

def aplicTrans(A, v):
    """
    Aplica la transformación lineal asociada a la matriz A sobre el vector v.
    Recibe: A: matriz de tamaño (m x n)
            v: vector de tamaño (n)
    Devuelve: vector resultado de tamaño (m)
    """

    if A.shape[1] != v.shape[0]:
        print("No se puede aplicar la matriz al vector")
        return
    res = np.zeros((A.shape[0]))

    #Cada posición i de res es el producto entre las k columnas de la fila i de A y v:
    for i in range(res.shape[0]):
        tot = 0
        for k in range(A.shape[1]):
            tot += A[i, k] * v[k]
        res[i] = tot

    return res

def prodPunto(v, w):
    """
    Producto punto entre dos vectores v y w
    Recibe: v: vector de tamaño (n)
            w: vector de tamaño (n)
    Devuelve: escalar resultado
    """

    if v.shape[0] != w.shape[0]:
        print("No se puede calcular el prodcuto punto")
        return

    return np.sum(v * w)

# Labo01
def error(x,y):
    """
    Calcula el error absoluto entre dos números x e y.
    Recibe: x, y: números reales
    Devuelve: |x - y|
    """
    return np.abs(np.float64(x) - np.float64(y))

# Labo03
def norma(x, p):
    """
    Calcula la norma p del vector x.
    Recibe: x: vector de tamaño (n)
            p: entero positivo o "inf" para norma infinito
    Devuelve: norma p de x
    """
    suma = 0
    #Caso en el que queremos calcular la norma infinito:
    if p == "inf":
        return np.abs(np.array(x)).max()

    #En otro caso, la norma que se quiere está definida por p:
    xabs = np.abs(x)
    suma = np.sum(xabs**p)
    return suma ** (1/p)

def normaliza(X):
    #Para cada fila de X, si la norma 2 de la fila no es 0, se la divide por la misma, es decir, se la normaliza.
    for i in range(X.shape[0]):
        normav = norma(X[i], p=2)
        if (normav == 0): continue
        X[i] /= normav
    return X

# Labo04
def calculaLU(A):
    """
    Calcula la factorización LU de la matriz A y retorna las matrices L y U,
    junto con el número de operaciones realizadas.
    En caso de que la matriz no pueda factorizarse, retorna None.
    Recibe: A: matriz cuadrada de tamaño (N x N)
    Devuelve: L, U, cant_op
    """

    cant_op = 0 # (El cálculo de cant. de operaciones se pide en el labo.)
    M=A.shape[0]
    N=A.shape[1]
    Ac = A.copy().astype(np.float64)

    if M!=N:
        return None

    finfo = np.finfo(Ac.dtype)

    # POr cada columna
    for j in range(N):
        # Toma el pivote
        piv = Ac[j][j]
        # Si es 0 no existe LU
        if abs(piv) < finfo.eps:  # pivote nulo o casi nulo
            return None
        # Elimino hacia abajo
        for i in range(j+1, N):
            # Calculo el multiplicador para eliminar el elemento en la fila i, columna j
            mult = Ac[i][j] / piv
            cant_op += 1
            # Actualizo la fila i restando mult * fila j
            for k in range(j+1, N):
                # Resto el elemento correspondiente de la fila j multiplicado por el multiplicador
                Ac[i][k] -= mult * Ac[j][k]
                cant_op += 2
            # Guardo el multiplicador en la posición correspondiente de la matriz A
            Ac[i][j] = mult

    L = np.eye(N,N)
    U = np.eye(N,N)

    # Separo L y U
    for i in range(N):
        for j in range(N):
            if (i <= j): U[i, j] = Ac[i, j]
            else: L[i, j] = Ac[i, j]

    return L, U, cant_op

def res_tri(A, b, inferior=True):
    """
    Resuelve un sistema de ecuaciones lineales Ax = b, donde A es una matriz triangular.
    Se puede indicar si es triangular inferior o superior usando el argumento
    `inferior` (por defecto se asume que es triangular inferior).
    Recibe: A: matriz triangular de tamaño (n x n)
            b: vector de tamaño (n)
            inferior: booleano que indica si A es triangular inferior (True) o superior (False)
    """
    X = np.zeros(len(b))  
    n = len(b)

    '''
    Si A es inferior, calculo X[i] haciendo sustitución hacia adelante, siendo:
        - A[i, :i] los elementos de A antes de la diagonal en la fila i.
        - X[:i] los valores ya conocidos de la solución
    Si es superior, es análogo pero para los elementos después de la diagonal.
    '''
    if inferior:
        for i in range(n):
            X[i] = (b[i] - (A[i, :i] * X[:i]).sum()) / A[i, i]
    else:
        for i in range(n-1, -1, -1):
            X[i] = (b[i] - (A[i, i+1:] * X[i+1:]).sum()) / A[i, i]

    return X

def res_tri_matricial(A, B, inferior=True):
    """
    Resuelve el sistema L X = B, donde L es triangular.

    Se puede indicar si es triangular inferior o superior usando el argumento
    `inferior` (por defecto se asume que es triangular inferior).

    Recibe: A: matriz triangular de tamaño (n x n)
            B: matriz de tamaño (n x m)
            inferior: booleano que indica si A es triangular inferior (True) o superior (False)
    Devuelve: X: matriz solución de tamaño (n x m)
    """
    N, M = B.shape
    X = np.zeros((N, M))

    #Resolver para cada columna de B independientemente
    for j in range(M):
        X[:, j] = res_tri(A, B[:, j], inferior=inferior)

    return X

def inversa(A):
    """
    Calcula la inversa de la matriz A usando la factorización LU
    Recibe: A: matriz cuadrada de tamaño (n x n)
    Devuelve: A_inv: matriz inversa de tamaño (n x n) o None si no es invertible
    """
    # Calcula la inversa de la matriz A usando la factorización LU
    L, U, _ = calculaLU(A)
    A_inv = np.zeros(A.shape)
    I = np.eye(A.shape[0])

    #Se tiene que A_inv = x <-> Ax = I <-> LUx = I <-> Ux = y and Ly = I
    for i in range(len(A)):
        y = res_tri(L, I[:, i], inferior=True) # Ly = I
        x = res_tri(U, y, inferior=False) # Ux = y
        A_inv[:, i] = x
    return A_inv

def QR_con_GS(A):
    """
    Factorización QR por Gram-Schmidt
    Recibe: A: matriz de tamaño (m x n)
    Devuelve: Q: matriz ortonormal de tamaño (m x n)
              R: matriz triangular superior de tamaño (n x n)
    """
    A = A.astype(np.float64)
    m, n = A.shape  # m filas, n columnas

    Q = np.zeros((m, n), dtype=np.float64)
    R = np.zeros((n, n), dtype=np.float64)

    #Primera columna
    a1 = A[:, 0]
    R[0, 0] = norma(a1, p=2)
    Q[:, 0] = a1 / R[0, 0]


    for j in range(1, n):  #Copiamos la columna j de A para modificarla
        q_tilde = A[:, j].copy()

        #Eliminamos las componentes que ya están en las columnas anteriores de Q
        for k in range(j):
            R[k, j] = prodPunto(Q[:, k], q_tilde)
            q_tilde = q_tilde - R[k, j] * Q[:, k]

        R[j, j] = norma(q_tilde, p=2)

        #Si el vector no es prácticamente cero, lo normalizamos
        if R[j, j] > 1e-10:  #Evitar división por cero
            Q[:, j] = q_tilde / R[j, j]
        else:
            Q[:, j] = q_tilde

    #Devolvemos Q y R, con Q ortonormal y R triangular superior
    return Q, R

def QR_con_HH(A, tol=1e-12):
    """
    Factorización QR por Householder
    Recibe: A: matriz de tamaño (m x n)
    Devuelve: Q: matriz ortogonal de tamaño (m x n)
              R: matriz triangular superior de tamaño (n x n)
    """
    A = A.astype(np.float64)
    m, n = A.shape
    R = A.copy() #R luego quedará triangular superior.
    Q = np.eye(m) #Q irá acumulando las transformaciones  de Householder.

    for k in range(n):
        #Tomamos la parte "inferior" de la col. actual (desde la fila k en adelante)
        x = R[k:, k]

        #Definimos e de modo que al restarlo de x genere la reflexión
        e = np.zeros_like(x)
        e[0] = np.sign(x[0]) * norma(x,p=2) if x[0] != 0 else norma(x)

        #Calculamos el vector de Householder
        u = x - e

        #Verificamos si u es o no casi 0
        if norma(u,p=2) < tol:
            continue

        u = u / norma(u,p=2)
        H = np.eye(m - k) - 2 * prodMat(np.expand_dims(u, axis=1), np.expand_dims(u, axis=0)) #Construimos la matriz de Householder H de tamaño reducido
        H_moño = np.eye(m)
        H_moño[k:, k:] = H #Insertamos H en la submatriz correspondiente

        R = prodMat(H_moño, R) #Anulamos los elementos debajo de la diagonal
        Q = prodMat(Q, H_moño)

    #Tomamos la desc de Q R reducida
    Q_red = Q[:, :n]  # Tomar solo las primeras n columnas de Q
    R_red = R[:n, :]  # Tomar solo las primeras n filas de R

    return Q_red, R_red

def QR_con_HH_NP(A, tol=1e-12):
    """
    Factorización QR por Householder (utilizando numpy para productos matriciales)
    """
    A = A.astype(np.float64)
    m, n = A.shape
    R = A.copy() #R luego quedará triangular superior.
    Q = np.eye(m) #Q irá acumulando las transformaciones  de Householder.

    for k in range(n):
        #Tomamos la parte "inferior" de la col. actual (desde la fila k en adelante)
        x = R[k:, k]

        #Definimos e de modo que al restarlo de x genere la reflexión
        e = np.zeros_like(x)
        e[0] = np.sign(x[0]) * norma(x,p=2) if x[0] != 0 else norma(x)

        #Calculamos el vector de Householder
        u = x - e

        #Verificamos si u es o no casi 0
        if norma(u,p=2) < tol:
            continue

        u = u / norma(u,p=2)
        H = np.eye(m - k) - 2 * (np.expand_dims(u, axis=1) @ np.expand_dims(u, axis=0)) #Construimos la matriz de Householder H de tamaño reducido
        H_moño = np.eye(m)
        H_moño[k:, k:] = H #Insertamos H en la submatriz correspondiente

        R = H_moño @ R #Anulamos los elementos debajo de la diagonal
        Q = Q @ H_moño

    #Tomamos la desc de Q R reducida
    Q_red = Q[:, :n]  # Tomar solo las primeras n columnas de Q
    R_red = R[:n, :]  # Tomar solo las primeras n filas de R

    return Q_red, R_red

def calculaCholesky(A):
    """
    Calcula la factorización de Cholesky A = L * L.T
    Recibe: A: matriz simétrica definida positiva de tamaño (N x N)
    Devuelve: L: matriz triangular inferior de tamaño (N x N)
    """
    # Usa la factorización LU -> LDU, asumiendo que A es simétrica definida positiva.

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


# La k es 150 para controlar el tiempo de ejecución
def metpot2k(A, tol=1e-15, K=150):
    """
    Calcula autovalor y autovector dominante mediante el método de la potencia.

    Retorna v_monio (autovector), lambda (autovalor) y e-1 (error).
    """
    N = A.shape[0]

    # Si la matriz es nula, entonces todos los autovalores son 0
    if norma(A, 2) < tol:
        v = np.random.rand(N)
        v = v / norma(v, 2)
        v = np.expand_dims(v, axis=0).T
        return v, 0, 0

    # Vector inicial
    v = np.random.rand(N)
    v = v / norma(v, 2)

    # Primera transformación lineal
    v_monio = aplicTrans(A, v)
    v_monio = v_monio / norma(v_monio, 2)

    e = prodPunto(v_monio, v)
    k = 0
    
    # Iteración hasta que converja o k llegue al límite
    while abs(e - 1) > tol and k < K:

        v = v_monio

        # Aplico transformación lineal
        w = aplicTrans(A, v)
        w = w / norma(w, 2)

        v_monio = aplicTrans(A, w)
        v_monio = v_monio / norma(v_monio, 2)

        e = prodPunto(v_monio, v)
        k += 1

    # Calculo el autovalor
    v_monio = np.expand_dims(v_monio, axis=0).T
    lambd = prodMat(prodMat(v_monio.T, A), v_monio)[0][0]

    return v_monio, lambd, e-1

def metpot2k_np(A, tol=1e-15, K=150):
    """
    Calcula autovalor y autovector dominante mediante el método de la potencia

    Retorna v_monio (autovector), lambda (autovalor) y e-1 (error)
    Esta función utiliza la libreria NumPy
    """
    N = A.shape[0]

    # Caso matriz nula, entonces todos los autovalores son 0
    if norma(A, 2) < tol:
        v = np.random.rand(N)
        v = v / norma(v, 2)
        v = np.expand_dims(v, axis=0).T
        return v, 0.0, 0

    # vector inicial
    v = np.random.rand(N)
    v = v / norma(v, 2)

    # Primera transformación lineal
    v_monio = A @ v
    v_monio = v_monio / norma(v_monio, 2)

    e = np.sum(v_monio * v)
    k = 0
    
    # Iteración hasta que converja o k llegue al límite
    while abs(e - 1) > tol and k < K:

        v = v_monio

        # Aplico transformación lineal
        w = A @ v
        w = w / norma(w, 2)

        v_monio = A @ w
        v_monio = v_monio / norma(v_monio, 2)

        e = np.sum(v_monio * v)
        k += 1

    # Calculo el autovalor
    v_monio = np.expand_dims(v_monio, axis=0).T
    lambd = (v_monio.T @ (A @ v_monio)) [0,0]

    return v_monio, lambd, e-1


def diagRH(A, tol=1e-15, K=1000):
    """
    Diagonalización recursiva con Householder
    
    Recibe A (matriz a diagonalizar), tol (tolerancia) y K (iteraciones máximas)
    """
    if A.shape[0] != A.shape[1]:
        print("Matriz no cuadrada")
        return None
    
    N = A.shape[0]

    if N == 1:
        return np.eye(1), A.copy()
    
    # Obtengo autovalor y autovector dominante
    v1, lambda1, _ = metpot2k(A, tol, K)
    
    # e1
    e1 = np.zeros((N, 1))
    e1[0,0] = 1.0
    
    # Vector de Householder
    u = e1 - v1
    nu = norma(np.squeeze(u), 2)
    
    if nu < 1e-12:
        Hv1 = np.eye(N)
    else:
        factor = 2 / (nu*nu)
        Hv1 = np.eye(N) - factor * prodMat(u, u.T)
    
    # Caso N=2, ya diagonaliza
    if N == 2:
        S = Hv1
        D = prodMat(prodMat(Hv1, A), Hv1.T)
        return S, D
    
    B = prodMat(prodMat(Hv1, A), Hv1.T)
    Ahat = B[1:N, 1:N]

    # Acá hago la recursión
    Shat, Dhat = diagRH(Ahat, tol, K)
    
    # Construyo D grande
    D = np.zeros((N, N))
    D[0, 0] = lambda1
    D[1:, 1:] = Dhat
    
    # Extiendo Shat a NxN
    Hprod = np.zeros((N, N))
    Hprod[0, 0] = 1.0
    Hprod[1:, 1:] = Shat
    
    # Matriz de autovectores final
    S = prodMat(Hv1, Hprod)
    
    return S, D

def diagRH_np(A, tol=1e-15, K=150):
    """
    Diagonalización recursiva con Householder
    
    Recibe A (matriz a diagonalizar), tol (tolerancia) y K (iteraciones máximas)
    Esta función utiliza la libreria NumPy
    """
    
    N = A.shape[0]

    if N == 1:
        return np.eye(1), A.copy()
    
    # Obtengo autovalor y autovector dominante
    v1, lambda1, _ = metpot2k_np(A, tol, K)
    
    # e1
    e1 = np.zeros((N, 1))
    e1[0,0] = 1.0
    
    # Vector de Householder
    u = e1 - v1
    nu = norma(np.squeeze(u), 2)
        
    if nu < 1e-12:
        Hv1 = np.eye(N)
    else:
        factor = 2.0 / (nu*nu)
        Hv1 = np.eye(N) - factor * (u @ u.T)
    
    # Caso N=2, ya diagonaliza
    if N == 2:
        S = Hv1
        D = Hv1 @ (A @ Hv1.T)
        return S, D
    
    # Modifica A in-place en función de no crear matrices nuevas y tener problemas de memoria
    A[:] = Hv1 @ (A @ Hv1.T)
    Ahat = A[1:, 1:]

    # Acá hago la recursión
    Shat, Dhat = diagRH_np(Ahat, tol, K)
    
    # Construyo D grande
    D = np.zeros((N, N))
    D[0, 0] = lambda1
    D[1:, 1:] = Dhat
    
    # Extiendo Shat a NxN
    Hprod = np.zeros((N, N))
    Hprod[0, 0] = 1.0
    Hprod[1:, 1:] = Shat
    
    # Matriz de autovectores final
    S = Hv1 @ Hprod
    
    return S, D

#labo08

def svd_reducida(A, k='max', tol=1e-15):
    """
    Calcula la descomposición en valores singulares de una matriz.

    Recibe:
    A la matriz de interes (de m x n),
    k el numero de valores singulares (y vectores) a retener,
    tol la tolerancia para considerar un valor singular igual a cero.

    Retorna U (matriz de m x k), S (vector de k valores singulares) y V (matriz de n x k)
    """
    
    n = A.shape[0]
    m = A.shape[1]
    
    if n < m: 
        M = prodMat(A, A.T)
    else:
        M = prodMat(A.T, A)
    
    S, D = diagRH(M, tol)

    # Creo listas con valores singulares y vectores singulares
    vals = []
    vecs = []
    for i in range(D.shape[0]):
        if D[i,i] > tol:
            vals.append(np.sqrt(D[i,i]))
            vecs.append(S[:, i])
    
    # Aplica k si tiene limite
    if k != 'max':
        vals = vals[:k]
        vecs = vecs[:k]
        
    r = len(vals)
    S = np.array(vals)
    U = np.zeros((n, r))
    V = np.zeros((m, r))
    
    # Formo U y V
    if n < m:
        for i in range(r):
            U[:, i] = vecs[i] / norma(vecs[i], 2)
            V_moño = aplicTrans(A.T, U[:, i])
            V[:, i] = V_moño / vals[i]
    else:
        for i in range(r):
            V[:, i] = vecs[i] / norma(vecs[i], 2)
            U_moño = aplicTrans(A, V[:, i])
            U[:, i] = U_moño / vals[i]

    return U, S, V

def svd_reducida_np(A, k='max', tol=1e-15):
    """
    Calcula la descomposición en valores singulares de una matriz.

    Recibe:
    A la matriz de interes (de m x n),
    k el numero de valores singulares (y vectores) a retener,
    tol la tolerancia para considerar un valor singular igual a cero.

    Retorna U (matriz de m x k), S (vector de k valores singulares) y V (matriz de n x k)
    Esta función utiliza funciones de la libreria NumPy
    """   
    n = A.shape[0]
    m = A.shape[1]
    
    if n < m: 
        M = A @ A.T
    else:
        M = A.T @ A
    
    S, D = diagRH_np(M, tol)
    
    # Creo listas con valores singulares y vectores singulares
    vals = []
    vecs = []
    for i in range(D.shape[0]):
        if D[i,i] > tol:
            vals.append(np.sqrt(D[i,i]))
            vecs.append(S[:, i])
            
    sigma = np.array(vals)

    # Se calcula una tolerancia relativa para decidir cuales serán mis autovalores relevantes
    eps = 1e-16
    tol_s = max(A.shape) * eps * np.max(sigma)
    
    # Creamos una lista de booleanos para saber con que autovalores quedarnos como última instancia luego de utilizar la tol del input,
    # En función de reasegurarnos que no se dividirá por un número muy chico a la hora de buscar los vectores singulares.
    utiles = sigma > tol_s
    nuevos_sigma = []
    nuevos_vec = []
    
    for i in range(len(sigma)):
        if utiles[i]:
            nuevos_sigma.append(sigma[i])
            nuevos_vec.append(vecs[i])
    
    sigma = np.array(nuevos_sigma)
    vecs = nuevos_vec
    
    # Aplica k si tiene limite
    if k != 'max':
        vals = vals[:k]
        vecs = vecs[:k]
        
    r = len(vals)
    S = np.array(vals)
    U = np.zeros((n, r))
    V = np.zeros((m, r))
    
    # Formo U y V
    if n < m:
        for i in range(r):
            U[:, i] = vecs[i] / norma(vecs[i],2)
            V_moño = A.T @ U[:, i]
            V[:, i] = V_moño / vals[i]
    else:
        for i in range(r):
            V[:, i] = vecs[i] / norma(vecs[i],2)
            U_moño = A @ V[:, i]
            U[:, i] = U_moño / vals[i]

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
    Ac = np.copy(A).astype(np.float64)
    Bc = np.copy(B).astype(np.float64)

    # Si no se especifica tolerancia, usar eps
    if tol is None:
        tol = np.finfo(np.float64).eps

    if Ac.shape != Bc.shape: return False
    for i in range(Ac.shape[0]):
        for j in range(Ac.shape[1]):
            if error(Ac[i][j], Bc[i][j]) > tol: return False
    return True

def esPseudoInversa(X, pX, tol=1e-8):
    Xc = X.copy().astype(np.float64)
    pXc = pX.copy().astype(np.float64)
    """
    Cheque que se cumplan las las cuatro condiciones de Moore-Penrose
    1. A @ A⁺ @ A = A
    2. A⁺ @ A @ A⁺ = A⁺
    3. (A @ A⁺)* = A @ A⁺
    4. (A⁺ @ A)* = A⁺ @ A
    """

    # (1)
    cond1 = matricesIguales(prodMat(pXc, prodMat(pXc, Xc)), Xc)
    if not cond1: return False 
    # (2)
    cond2 = matricesIguales(prodMat(pXc, prodMat(Xc, pXc)), pXc)
    if not cond2: return False 
    # (3)
    cond3 = matricesIguales(prodMat(Xc, pXc).T, prodMat(Xc, pXc))
    if not cond3: return False 
    # (4)
    cond4 =  matricesIguales(prodMat(pXc, Xc).T, prodMat(pXc, Xc))
    if not cond4: return False 

    return True

def esPseudoInversaNP(X, pX, tol=1e-8):
    Xc = X.copy().astype(np.float64)
    pXc = pX.copy().astype(np.float64)
    """
    Cheque que se cumplan las las cuatro condiciones de Moore-Penrose
    1. A @ A⁺ @ A = A
    2. A⁺ @ A @ A⁺ = A⁺
    3. (A @ A⁺)* = A @ A⁺
    4. (A⁺ @ A)* = A⁺ @ A
    """

    # (1)
    cond1 = matricesIguales(Xc @ pXc @ Xc, Xc, tol=tol)
    if not cond1: return False 
    
    # (2)
    cond2 = matricesIguales(pXc @ Xc @ pXc, pXc, tol=tol)
    if not cond2: return False 
    
    # (3)
    cond3 = matricesIguales((Xc @ pXc).T, Xc @ pXc, tol=tol)
    if not cond3: return False 
    
    # (4)
    cond4 = matricesIguales((pXc @ Xc).T, pXc @ Xc, tol=tol)
    if not cond4: return False

    return True        

def pinvEcuacionesNormales(X, L, Y):
    N, M = X.shape
    Xc = X.copy().astype(np.float64)
    Yc = Y.copy().astype(np.float64)
    if N == M:
        pX = inversa(Xc)
        W = prodMat(Yc, pX)
    else:
        if N > M:
            B = Xc.T
        else:
            B = Xc

    V = res_tri_matricial(L, B, inferior=True)
    U = res_tri_matricial(L.T, V, inferior=False)

    if N > M: pX = U
    else: pX = U.T

    W = prodMat(Yc, pX)

    return W

def pinvHouseHolder(Q, R, Y):
  Yc = Y.copy().astype(np.float64)
 
  #Se tiene que VR^T=Q. Por tanto, el núm. de columnas de V es el de filas de R^T. Y su núm. de filas es el de filas de Q.
  V = np.zeros((Q.shape[0],R.T.shape[0]))
  #Además, RV^T = Q^T.
  V = V.T
  V = res_tri_matricial(R, Q.T, inferior=False)
  V = V.T

  W = prodMat(Yc, V)
  return W

def pinvGramSchmidt(Q, R, Y):
  Yc = Y.copy().astype(np.float64)
 
  #Se tiene que VR^T=Q. Por tanto, el núm. de columnas de V es el de filas de R^T. Y su núm. de filas es el de filas de Q.
  V = np.zeros((Q.shape[0],R.T.shape[0]))
  #Además, RV^T = Q^T.
  V = V.T
  V = res_tri_matricial(R, Q.T, inferior=False)
  V = V.T

  W = prodMat(Yc, V)
  return W

def pinvSVD(U, S, V, Y):
    Yc = Y.copy().astype(np.float64)
    r = V.shape[1]
    
    # En vez de pasar a S+ divido cada Vi por S[i]
    for i in range(r):
        V[:, i] = V[:, i]/S[i]
    
    X_ps = prodMat(V, U.T)
    W = prodMat(Yc, X_ps)
    return W


def matriz_confusion(Y_real, Y_pred):
    """
    Matriz de confusión y accuracy para nuestros valores.
    """

    # Pasar de one-hot a etiquetas enteras
    y_true = np.argmax(Y_real, axis=0)
    y_pred = np.argmax(Y_pred, axis=0)

    # Cantidad de clases
    num_clases = max(y_true.max(), y_pred.max()) + 1

    # Matriz de confusión
    M = np.zeros((num_clases, num_clases), dtype=float)
    for real, pred in zip(y_true, y_pred):
        M[real, pred] += 1
    M /= Y_real.shape[1]

    # Gráfico
    plt.figure()
    im = plt.imshow(M, cmap = 'Blues')
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Clase real")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(im)
    
    # Agregar valores sobre la matriz
    for i in range(num_clases):
        for j in range(num_clases):
            plt.text(j, i, str(M[i, j]),
                     ha='center', va='center')
    plt.show()

    # Calcular accuracy
    aciertos = 0
    total = 0
    for i in range (M.shape[0]):
        for j in range (M.shape[1]):
            if i == j:
                aciertos += M[i,i]
                total += M[i,i]
            else:
                total += M[i,j]

    accuracy = aciertos / total
    return M, accuracy.round(3)