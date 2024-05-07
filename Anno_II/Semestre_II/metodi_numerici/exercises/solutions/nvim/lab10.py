#!/bin/env python3

import numpy as np
import scipy.linalg as spl
import lib.solve_triangular as st
import matplotlib.pyplot as plt


def eqnorm(B, y):
    G = B.T @ B

    condG = np.linalg.cond(G)
    print('condizionamento G: ', condG)

    f = B.T @ y
    L = spl.cholesky(G, lower=True)
    b, flag = st.Lsolve(L, f)

    if flag == 0:
        a, flag = st.Usolve(L.T, b)

    return a


def qrls(B, y):
    n = min(B.shape[0], B.shape[1])
    Q, R = spl.qr(B)
    R1 = R[0:n, 0:n]
    h = Q.T @ y
    h1 = h[0:n]
    a, flag = st.Usolve(R1, h1)

    condR1 = np.linalg.cond(R1)
    print('condizionamento R1: ', condR1)

    stima_errore = np.linalg.norm(h, ord=2) - np.linalg.norm(h1, ord=2)
    print('stima dell\'errore |h2|:', stima_errore)

    return a


def test_pol_reg(grado=None, fun=eqnorm):
    x = np.array([-3.5,  -3, -2, -1.5, -0.5, 0.5, 1.7, 2.5, 3])
    y = np.array([-3.9, -4.8, -3.3, -2.5, 0.3, 1.8, 4, 6.9, 7.1])
    m = x.shape[0]
    if grado is None:
        n = m-1  # grado del polinomio di regressione
    else:
        n = grado
    n1 = n + 1
    B = np.vander(x, increasing=True, N=n1)
    a = fun(B, y)

    # useful for debugging
    if a is None:
        return

    xv = np.linspace(np.min(x), np.max(x), 200)
    pol_EQN = np.polyval(np.flip(a), xv)
    plt.plot(x, y, '*', xv, pol_EQN)
    plt.show()


test_pol_reg(grado=None, fun=qrls)
