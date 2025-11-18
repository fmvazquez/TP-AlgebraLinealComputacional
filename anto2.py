# import numpy as np
# import sys
# sys.setrecursionlimit(2000)

# def norma(x, p):
#     suma = 0
#     if p == "inf": 
#         return np.abs(np.array(x)).max()
#     xabs = np.abs(x)
#     suma = np.sum(xabs**p)
    
#     return suma ** (1/p)

# def prodMat(A, B):
#     """
#     Se hace slicing multiplicando cada fila de A por cada fila de B_T.
#     """
#     if A.shape[1] != B.shape[0]:
#         print("No se puede hacer el producto matricial")
#         return
    
#     A = A.astype(np.float64)
#     B = B.astype(np.float64)
#     res = np.zeros((A.shape[0], B.shape[1]), dtype=np.float64)

#     B_T = B.T 

#     for i in range(res.shape[0]):
#         res[i, :] = np.sum(A[i,:] * B_T, axis=1)

#     return res

# def prodPunto(v, w):
#     """
#     Asume que el vector es columna
#     """
#     return np.sum(v*w)

# def aplicTrans(A, v):
#     """
#     Asume que el vector es columna
#     """
#     # if A.shape[1] != v.shape[0]:
#     #     print("No se puede aplicar la matriz al vector")
#     #     return
#     # res = np.zeros((A.shape[0]))

#     # for i in range (res.shape[0]):
#     #     res[i] = prodPunto(A[i,:], v)

#     # return res
#     return A@v


# def metpot2k(A, tol=1e-15, K=150):
#     N = A.shape[0]

#     # Caso matriz nula → todos los autovalores son 0
#     if norma(A, 2) < tol:
#         v = np.random.rand(N)
#         v = v / norma(v, 2)
#         v = np.expand_dims(v, axis=0).T
#         return v, 0.0, 0

#     # vector inicial
#     v = np.random.rand(N)
#     v = v / norma(v, 2)

#     v_monio = aplicTrans(A, v)
#     v_monio = v_monio / norma(v_monio, 2)

#     e = prodPunto(v_monio, v)
#     k = 0
    
#     while abs(e - 1) > tol and k < K:

#         v = v_monio

#         # doble aplicación
#         w = aplicTrans(A, v)
#         w = w / norma(w, 2)

#         v_monio = aplicTrans(A, w)
#         v_monio = v_monio / norma(v_monio, 2)

#         e = prodPunto(v_monio, v)
#         k += 1

#     # Calcular A * v_monio (el vector resultante)
#     w_lambda = aplicTrans(A, v_monio) 
#     # Calcular el Cociente de Rayleigh (v^T * w_lambda / v^T * v). Como v está normalizado, v^T * v = 1.
#     lambd = prodPunto(v_monio, w_lambda)
#     # v_monio1 = np.expand_dims(v_monio, axis=0).T
#     # lambd = aplicTrans(prodMat(v_monio1.T, A), v_monio).item()

#     return v_monio, lambd, e-1

# import time

# def diagRH(A, tol=1e-15, K=150, nivel=1):
#     """
#     Diagonalización recursiva con Householder
    
#     Args:
#         A: matriz a diagonalizar    
#         tol: tolerancia
#         K: iteraciones máximas
#         nivel: nivel de recursión (para tracking)
#     """
#     # Marca el inicio de esta iteración
#     inicio = time.time()
    
#     if A.shape[0] != A.shape[1]:
#         print("Matriz no cuadrada")
#         return None
    
#     N = A.shape[0]
    
    
#     # autovector dominante
#     v1, lambda1, _ = metpot2k(A, tol, 150)
    
#     # e1
#     e1 = np.zeros((N, 1))
#     e1[0,0] = 1.0
    
#     # vector de Householder
#     u = e1 - v1
#     nu = norma(np.squeeze(u), 2)
    
#     print(f'Nivel {nivel}: procesando matriz {N}x{N}')
    
#     if nu < 1e-12:
#         Hv1 = np.eye(N)
#     else:
#         factor = 2.0 / (nu*nu)
#         Hv1 = np.eye(N) - factor * (u * u.T)
    
#     # Caso N=2 → ya diagonaliza
#     if N == 2:
#         S = Hv1
#         # D = prodMat(prodMat(Hv1, A), Hv1.T)
#         D = (Hv1 @ A) @ Hv1.T
        
#         fin = time.time()
#         tiempo_transcurrido = fin - inicio
#         minutos = int(tiempo_transcurrido // 60)
#         segundos = tiempo_transcurrido % 60
#         print(f"Nivel {nivel} (N=2, caso base): {minutos}m {segundos:.3f}s")
        
#         return S, D
    
#     # A[:] = prodMat(prodMat(Hv1, A), Hv1.T)
#     A[:] = (Hv1 @ A) @ Hv1.T
#     Ahat = A[1:, 1:]

#     # Recurrencia (pasa el nivel incrementado)
#     Shat, Dhat = diagRH(Ahat, tol, K, nivel + 1)
    
#     # Construyo D grande
#     D = np.zeros((N, N))
#     D[0, 0] = lambda1
#     D[1:, 1:] = Dhat
    
#     # Extiendo Shat a N×N
#     Hprod = np.zeros((N, N))
#     Hprod[0, 0] = 1
#     Hprod[1:, 1:] = Shat
    
#     # matriz de autovectores final
#     # S = prodMat(Hv1, Hprod)
#     S = Hv1 @ Hprod
    
#     return S, D


# def svd_reducida(A, k='max', tol=1e-15):
    
#     n = A.shape[0]
#     m = A.shape[1]
    
#     if n < m: 
#         # M = prodMat(A, A.T)
#         M = A @ A.T
#     else:
#         # M = prodMat(A.T, A)
#         M = A.T @ A
    
#     S, D = diagRH(M, tol)
    
#     aval = []
#     avec = []
#     for i in range(D.shape[0]):
#         if D[i,i] > tol:
#             aval.append(np.sqrt(D[i,i]))
#             avec.append(S[:, i])
            
#     sigma = np.array(aval)

#     # Calculo una tolerancia relativa para decidir cuales serán mis autovalores
#     eps = 1e-16
#     tol_s = max(A.shape) * eps * np.max(sigma)
    
#     # Creamos una lista de booleanos que para saber con que autovalores quedarnos (los que no estén cerca de 0)
#     utiles = sigma > tol_s
#     nuevos_sigma = []
#     nuevos_vec = []
    
#     for i in range(len(sigma)):
#         if utiles[i]:
#             nuevos_sigma.append(sigma[i])
#             nuevos_vec.append(avec[i])
    
#     sigma = np.array(nuevos_sigma)
#     avec = nuevos_vec
    
#     # Aplica k si tiene limite
#     if k != 'max':
#         aval = aval[:k]
#         avec = avec[:k]
    
#     print('3 va por aqui')
    
#     r = len(aval)
#     S = np.array(aval)
#     U = np.zeros((n, r))
#     V = np.zeros((m, r))
    
#     # Formo U y V
#     if n < m:
#         for i in range(r):
#             U[:, i] = avec[i] / norma(avec[i])
#             V_moño = aplicTrans(A.T, U[:, i])
#             V[:, i] = V_moño / aval[i]
#     else:
#         for i in range(r):
#             V[:, i] = avec[i] / norma(avec[i])
#             U_moño = aplicTrans(A, V[:, i])
#             U[:, i] = U_moño / aval[i]

#     return U, S, V

# def fullyConnectedSVD(X,Y):
#     Xc = X.copy().astype(np.float64)
#     Yc = Y.copy().astype(np.float64)
#     U, S, V = svd_reducida(Xc)
    
#     r = V.shape[1]
    
#     # En vez de pasar a S+ divido cada Vi por S[i]
#     for i in range(r):
#         V[:, i] = V[:, i]/S[i]
    
#     # X_ps = prodMat(V, U.T)
#     X_ps = V @ U.T
#     # W = prodMat(Yc, X_ps)
#     W = Yc @ X_ps
    
#     return W, X_ps



import numpy as np
import sys
sys.setrecursionlimit(2000)

def norma(x, p=2):
    x = np.ravel(x)
    if p == 'inf':
        return np.max(np.abs(x))
    return np.sum(np.abs(x)**p)**(1/p)

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

def prodPunto(v, w):
    """
    Asume que el vector es columna
    """

    return np.sum(v*w)


# def aplicTrans(A, v):
#     """
#     Asume que el vector es columna
#     """

#     if A.shape[1] != v.shape[0]:
#         print("No se puede aplicar la matriz al vector")
#         return
#     res = np.zeros((A.shape[0]))

#     for i in range(res.shape[0]):
#         res[i] = np.sum(A[i,:]*v)

#     return res

def aplicTrans(A, v):
    return A @ v

def gram_schmidt(V):
    """Ortogonaliza columnas de V usando Gram-Schmidt modificado"""
    n, m = V.shape
    U = np.zeros_like(V)
    
    for i in range(m):
        U[:, i] = V[:, i].copy()
        # Resta proyecciones sobre vectores anteriores
        for j in range(i):
            proj = np.dot(U[:, j], V[:, i])
            U[:, i] -= proj * U[:, j]
        
        # Normaliza
        norm = norma(U[:, i], 2)
        if norm > 1e-14:
            U[:, i] /= norm
        else:
            # Vector nulo, poner ceros
            U[:, i] = 0
            
    return U

def metpot2k(A, tol=1e-15, K=150):
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


def diagRH(A, tol=1e-15, K=150):
    """
    Diagonalización recursiva con Householder
    
    Args:
        A: matriz a diagonalizar    
        tol: tolerancia
        K: iteraciones máximas
        nivel: nivel de recursión (para tracking)
    """
    
    N = A.shape[0]
    
    # Obtengo autovalor y autovector dominante
    v1, lambda1, _ = metpot2k(A, tol, 150)
    
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
        D = Hv1 @ A @ Hv1.T
        return S, D
    
    A[:] = Hv1 @ A @ Hv1.T
    Ahat = A[1:, 1:]

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
    S = Hv1 @ Hprod
    
    return S, D


def svd_reducida(A, k='max', tol=1e-15):
    
    n = A.shape[0]
    m = A.shape[1]
    
    if n < m: 
        M = A @ A.T
    else:
        M = A.T @ A
    
    S, D = diagRH(M, tol)
    
    # Creo listas con valores singulares y vectores singulares
    vals = []
    vecs = []
    for i in range(D.shape[0]):
        if D[i,i] > tol:
            vals.append(np.sqrt(D[i,i]))
            vecs.append(S[:, i])
            
    sigma = np.array(vals)
    eps = 1e-16
    tol_s = max(A.shape) * eps * np.max(sigma)
    
    # Creamos una lista de booleanos que para saber con que autovalores quedarnos (los que no estén cerca de 0)
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
            U[:, i] = vecs[i] / norma(vecs[i])
            V_moño = A.T @ U[:, i]
            V[:, i] = V_moño / vals[i]
    else:
        for i in range(r):
            V[:, i] = vecs[i] / norma(vecs[i])
            U_moño = A @ V[:, i]
            U[:, i] = U_moño / vals[i]

    return U, S, V

def fullyConnectedSVD(X,Y):
    U, S, V = svd_reducida(X)
    
    VS= np.zeros(V.shape)
    r = VS.shape[1]
    
    # en vez de pasar a S+ divido cada Vi por S[i] que es como multiplicar por 1/S[i] (inversa de sigma)
    for i in range(r):
        VS[:, i] = V[:, i]/S[i]
    
    X_ps = VS@U.T
    W = Y@X_ps
    
    return W, X_ps

import matplotlib.pyplot as plt

def matriz_confusion(Y_real, Y_pred):
    """
    Matriz de confusión y accuracy de nuestros valores.
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
    plt.imshow(M, cmap = 'Blues')
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Clase real")
    plt.xticks([])
    plt.yticks([])
    #plt.colorbar(im)
    # Agregar valores sobre la matriz
    for i in range(num_clases):
        for j in range(num_clases):
            plt.text(j, i, str(M[i, j]),
                     ha='center', va='center')
    plt.show()

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