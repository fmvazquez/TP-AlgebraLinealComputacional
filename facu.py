from alc import calculaCholesky, prodMat, res_tri_matricial, inversa, matricesIguales
import numpy as np


A = np.array([
    [25,15,-5],
    [15,18,0],
    [-5,0,11]
])

calculaCholesky(A)

def fullyConectedCholesky(X, Y):
    N, M = X.shape
    if N == M:
        pX = inversa(X)
        W = prodMat(Y, inversa(X))
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
        V = res_tri_matricial(L, B)
        print("Resolviendo sistemas triangulares .2")
        U = res_tri_matricial(L.T, V)

        if N > M: pX = U
        else: pX = U.T

        print("Calculando W")
        W = prodMat(Y, pX)

    return W, pX

def esPseudoInversa(X, pX, tol=1e-8):
    """
    Cheque que se cumplan las las cuatro condiciones de Moore-Penrose
    1. A @ A⁺ @ A = A
    2. A⁺ @ A @ A⁺ = A⁺
    3. (A @ A⁺)* = A @ A⁺
    4. (A⁺ @ A)* = A⁺ @ A
    """

    # (1)
    cond1 = matricesIguales(prodMat(X, prodMat(pX, X)), X)
    if not cond1: return False 
    # (2)
    cond2 = matricesIguales(prodMat(pX, prodMat(X, pX)), pX)
    if not cond2: return False 
    # (3)
    cond3 = matricesIguales(prodMat(X, pX).T, prodMat(X, pX))
    if not cond3: return False 
    # (4)
    cond4 =  matricesIguales(prodMat(pX, X).T, prodMat(pX, X))
    if not cond4: return False 


