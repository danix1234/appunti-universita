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


def test_pol_reg(grado=None, fun=eqnorm, coords=0):
    if coords == 0:
        x = np.array([-3.5,  -3, -2, -1.5, -0.5, 0.5, 1.7, 2.5, 3])
        y = np.array([-3.9, -4.8, -3.3, -2.5, 0.3, 1.8, 4, 6.9, 7.1])
    elif coords == 1:
        x = np.array([-3.14, -2.4, -1.57, -0.7, -0.3, 0, 0.4, 0.7, 1.57])
        y = np.array([0.02, -1, -0.9, -0.72, -0.2, -0.04, 0.65, 0.67, 1.1])
    elif coords == 2:
        x = np.array([1.001, 1.004, 1.005, 1.0012,
                     1.0013, 1.0014, 1.0015, 1.0016])
        y = np.array([-1.2, -0.95, -0.9, -1.15, -1.1, -1])
    else:
        return
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


test_pol_reg(grado=None, fun=qrls, coords=1)


# build jth lagrange polynomial
def plagr(xnodi, j):
    xzeri = np.zeros_like(xnodi)
    n = xnodi.size

    if j == 0:
        xzeri = xnodi[1:n]
    else:
        xzeri = np.append(xnodi[0:j], xnodi[j+1, n])

    num = np.poly(xzeri)
    den = np.polyval(num, xnodi[j])
    p = num / den

    return p


def interpl(x, y, xv):
    n = x.size
    nv = xv.size
    L = np.zeros((nv, n))

    for j in range(n):
        p = plagr(x, y)
        L[:, j] = np.polyval(p, xv)

    return L@y


def f(x): return 3*x**3 + 2*x**2 + 2*x - 1


n = 3
x = np.linspace(-1, 1, n+1)
y = f(x)
xv = np.linspace(-1, 1, 200)
pol_interpl = interpl(x, y, xv)
plt.plot(xv, pol_interpl, x, y, 'ro')
plt.show()
