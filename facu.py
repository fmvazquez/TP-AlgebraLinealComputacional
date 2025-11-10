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


