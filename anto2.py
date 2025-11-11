import numpy as np

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

    if v.shape[0] != w.shape[0]:
        print("No se puede calcular el prodcuto punto")
        return
    
    res = np.sum(v*w)

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
        res[i] = np.sum(A[i,:]*v)

    return res

def metpot2k(A, tol=1e-15, K=50):
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
    #stuck = 0
    
    print('1 va por aca, tdv no entro al while')
    
    while abs(e - 1) > tol and k < K:

        v = v_monio

        # doble aplicación
        w = aplicTrans(A, v)
        w = w / norma(w, 2)

        v_monio = aplicTrans(A, w)
        v_monio = v_monio / norma(v_monio, 2)

        e = prodPunto(v_monio, v)
        k += 1

        # # Si está oscilando por autovalores iguales → reinicio
        # if abs(e) < 0.05:
        #     stuck += 1
        #     if stuck > 5:
        #         v = np.random.rand(N)
        #         v = v / norma(v, 2)
        #         stuck = 0

    v_monio = np.expand_dims(v_monio, axis=0).T
    # lambd = prodMat(prodMat(v_monio.T, A), v_monio)[0][0]
    lambd = (v_monio.T@A@v_monio) [0][0]

    return v_monio, lambd, e-1

import time

def diagRH(A, tol=1e-15, K=50, nivel=1):
    """
    Diagonalización recursiva con Householder
    
    Args:
        A: matriz a diagonalizar
        tol: tolerancia
        K: iteraciones máximas
        nivel: nivel de recursión (para tracking)
    """
    # Marca el inicio de esta iteración
    inicio = time.time()
    
    if A.shape[0] != A.shape[1]:
        print("Matriz no cuadrada")
        return None
    
    N = A.shape[0]
    
    # 1x1 → trivial
    if N == 1:
        fin = time.time()
        tiempo_transcurrido = fin - inicio
        minutos = int(tiempo_transcurrido // 60)
        segundos = tiempo_transcurrido % 60
        print(f"Nivel {nivel} (N=1, caso base): {minutos}m {segundos:.3f}s")
        return np.eye(1), A.copy()
    
    # autovector dominante
    v1, lambda1, _ = metpot2k(A, tol, 50)
    
    # e1
    e1 = np.zeros((N, 1))
    e1[0][0] = 1.0
    
    # vector de Householder
    u = e1 - v1
    nu = norma(np.squeeze(u), 2)
    
    print(f'Nivel {nivel}: procesando matriz {N}x{N}')
    
    # si v1 ≈ e1 → no hacer Householder (H=I)
    if nu < 1e-12:
        Hv1 = np.eye(N)
    else:
        factor = 2.0 / (nu * nu)
        Hv1 = np.eye(N) - factor * u @ u.T
    
    # Caso N=2 → ya diagonaliza
    if N == 2:
        S = Hv1
        D = Hv1 @ A @ Hv1.T
        
        fin = time.time()
        tiempo_transcurrido = fin - inicio
        minutos = int(tiempo_transcurrido // 60)
        segundos = tiempo_transcurrido % 60
        print(f"Nivel {nivel} (N=2, caso base): {minutos}m {segundos:.3f}s")
        
        return S, D
    
    # Construir B = H A H^T
    B = Hv1 @ A @ Hv1.T
    
    # Submatriz sin la primera fila/col
    Ahat = B[1:N, 1:N]
    
    # Recurrencia (pasa el nivel incrementado)
    Shat, Dhat = diagRH(Ahat, tol, K, nivel + 1)
    
    # Construyo D grande
    D = np.zeros((N, N))
    D[0, 0] = lambda1
    D[1:, 1:] = Dhat
    
    # Extiendo Shat a N×N
    Hprod = np.zeros((N, N))
    Hprod[0, 0] = 1.0
    Hprod[1:, 1:] = Shat
    
    # matriz de autovectores final
    S = Hv1 @ Hprod
    
    # Calcula y muestra el tiempo al terminar esta iteración
    fin = time.time()
    tiempo_transcurrido = fin - inicio
    minutos = int(tiempo_transcurrido // 60)
    segundos = tiempo_transcurrido % 60
    print(f"Nivel {nivel} (N={N}): {minutos}m {segundos:.3f}s")
    
    return S, D


def svd_reducida (A, k ='max' , tol=1e-15):
    
    n = A.shape[0]
    m = A.shape[1]
    
    if n < m: 
        # M = prodMat(A, A.T)
        M = A@A.T
    else:
        # M = prodMat(A.T, A)
        M = A.T@A
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
    
    print('3 va por aqui')
    
    r = len(aval)
    S = np.array(aval)
    U = np.zeros((n, r))
    V = np.zeros((m,r))
    
    if n < m:
        for i in range(r):
            U[:, i] = avec[i] / norma(avec[i])
            V_moño = aplicTrans(A.T,U[:, i])
            V[:, i] = V_moño / aval[i]
    else:
        for i in range(r):
            V[:, i] = avec[i] / norma(avec[i])
            U_moño = aplicTrans(A,V[:, i])
            U[:,i] = U_moño / aval[i]
    
    
    return U, S, V

def fullyConnectedSVD(X,Y):
    print('hola')
    U, S, V = svd_reducida(X)
    
    # #convertir S a S+
    # for i in range (S.shape):
    #     S[i] = 1/S[i]
    
    VS= np.zeros(V.shape)
    r = np.zeros[1]
    # en vez de pasar a S+ divido cada Vi por S[i] que es como multiplicar por 1/S[i] (inversa de sigma)
    for i in range(r):
        VS[:, i] = V[:, i]/S[i]
    
    X_ps = VS@U.T
    W = Y@X_ps
    
    return W