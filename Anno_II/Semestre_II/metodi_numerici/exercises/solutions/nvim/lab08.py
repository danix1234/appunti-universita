#!/bin/env python3

import numpy as np


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
        alfa = num / (p.T@Ap)

        x = x + alfa*p
        r = r+alfa*Ap
        vec_sol.append(x)
        criterio_di_arresto = np.linalg.norm(r) / nb
        vec_r.append(criterio_di_arresto)
        p = -r

    return x, vec_r, vec_sol, it
