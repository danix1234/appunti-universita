#!/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def steepest_descent(A, b, x0, itmax, toll):
    n, m = A.shape
    if n != m:
        print("Matrice non quadrata")
        return None, None, None, None

    x = x0
    r = A@x - b
    p = -r
    it = 0
    nb = np.linalg.norm(b)
    criterio_di_arresto = np.linalg.norm(r) / nb
    vec_sol = [x]
    vec_r = [criterio_di_arresto]

    while criterio_di_arresto >= toll and it < itmax:
        it += 1
        Ap = A@p
        num = -(r.T@p)
        alpha = num / (p.T@Ap)

        x = x + alpha * p
        r = r + alpha * Ap
        vec_sol.append(x)
        criterio_di_arresto = np.linalg.norm(r) / nb
        vec_r.append(criterio_di_arresto)
        p = -r

    return x, vec_r, vec_sol, it


def descent():
    A = np.array([[8, 4], [4, 3]])
    b = np.array([[8], [10]])
    x0 = np.zeros_like(b)
    itmax = 200
    tol = 1e-10
    x, vec_r, vec_sol, it = steepest_descent(A, b, x0, itmax, tol)
    print(x, it)
    plt.semilogy(np.arange(it+1), vec_r, '-r')
    plt.show()


def steepest_descent_CL(A, b, x0, X, Y, Z, f, itmax, toll):
    n, m = A.shape
    if n != m:
        print("Matrice non quadrata")
        return None, None, None, None

    x = x0
    plt.contour(X, Y, Z, levels=f(x, A, b).flatten())
    plt.plot(x[0], x[1], 'ro')
    r = A@x - b
    p = -r
    it = 0
    nb = np.linalg.norm(b)
    criterio_di_arresto = np.linalg.norm(r) / nb
    vec_sol = [x]
    vec_r = [criterio_di_arresto]

    while criterio_di_arresto >= toll and it < itmax:
        it += 1
        Ap = A@p
        num = -(r.T@p)
        alpha = num / (p.T@Ap)

        x = x + alpha * p
        plt.contour(X, Y, Z, levels=f(x, A, b).flatten())
        plt.plot(x[0], x[1], 'ro')
        r = r + alpha * Ap
        vec_sol.append(x)
        criterio_di_arresto = np.linalg.norm(r) / nb
        vec_r.append(criterio_di_arresto)
        p = -r
    return x, vec_r, vec_sol, it


def f(x, A, b):
    Ax = A@x
    xTAx = x.T@Ax
    bx = b.T@x
    return 0.5 * xTAx - bx


def descent_draw():
    A = np.array([[8, 4], [4, 3]])
    b = np.array([[8], [10]])
    x0 = np.zeros_like(b)
    itmax = 200
    tol = 1e-10
    x = np.linspace(-7.0, 3.0, 100)
    y = np.linspace(-5.0, 14.0, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(len(y)):
        for j in range(len(x)):
            x_coor = X[i][j]
            y_coor = Y[i][j]
            Z[i][j] = f(np.array([[x_coor], [y_coor]]), A, b)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap=plt.cm.viridis)
    x, vec_r, vec_sol, it = steepest_descent_CL(
        A, b, x0, X, Y, Z, f, itmax, tol)


def jacobbi(A, b, x0, toll, it_max):
    d = np.diag(A)
    invM = np.diag(1 / d)
    E = np.tril(A, -1)
    F = np.triu(A, 1)
    N = -(E+F)
    T = invM @ N
    eigs = np.linalg.eigvals(T)
    rags = np.max(np.abs(eigs))

    print(rags)

    it = 0
    er_vet = []
    err = np.inf
    q = invM @ b
    while it <= it_max and err >= toll:
        x = T @ x0 + q
        # x = (b + N @ x0)
        err = np.linalg.norm(x-x0)/np.linalg.norm(x)
        er_vet.append(err)
        x0 = x.copy()
        it += 1

    return x, it, er_vet


def test_jacobbi():
    n = 3
    A = np.array([[4, 1, 3], [3, 4, 1], [1, 1, 17]])
    b = np.sum(A, axis=1).reshape(n, 1)
    x0 = np.zeros_like(b)
    itmax = 500
    toll = 1e-8
    x, it, er_vet = jacobbi(A, b, x0, toll, itmax)


# actual execution
# descent()
# descent_draw()
test_jacobbi()
