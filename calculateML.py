import numpy as nu
import pandas as pd
from matplotlib import pyplot as pl
from matplotlib import cm
from mpl_toolkits import mplot3d

from scipy.integrate import quadrature
from scipy.special import ellipk, ellipe, ellipkm1


# Constants

mu0 = 4*nu.pi*1e-7
r1 = 12.2e-2/2


# Variables

points = 100
deltas = nu.linspace(1e-1, 1, points)
ns = nu.linspace(0.1, 0.5, points)


# Model

def _f(phi, r1, r2, d):
    return r1 * r2 * nu.cos(phi) / nu.sqrt( r1**2 + r2**2 + d**2 - 2*r1*r2*nu.cos(phi) )

def _g(u, r1, r2, d):
    phi = nu.pi*(u+1)
    delta = r2/r1
    n = d/r1
    return r2 * nu.cos(phi) * nu.pi / nu.sqrt( 1 + (delta)**2 + (n)**2 - 2*delta*nu.cos(phi) )


def MutalInductance(r1, r2, d):
    # return 0.5 * mu0 * quadrature(_f, 0, 2*nu.pi, args=(r1, r2, d), tol=1e-9, maxiter=100000)[0]
    # return 0.5 * mu0 * quadrature(_g, -1, 1, args=(r1, r2, d), tol=1e-12, maxiter=100000)[0]

    k = nu.sqrt(4*r1*r2/((r1+r2)**2+d**2))
    result = mu0 * nu.sqrt(r1*r2) * ( (2/k-k)*ellipk(k**2) - 2/k*ellipe(k**2) )
    if result >= 0:
        return result
    else:
        return 0.5 * mu0 * quadrature(_f, 0, 2*nu.pi, args=(r1, r2, d), tol=1e-6, maxiter=1000)[0]


def Mtotal(r1, l1, N1, r2, l2, N2):
    if N1 >=2:
        position1s = nu.linspace(-l1/2, l1/2, N1)
    else:
        position1s = [0]

    if N2 >=2:
        position2s = nu.linspace(-l2/2, l2/2, N2)
    else:
        position2s = [0]

    Msum = 0
    for position1 in position1s:
        for position2 in position2s:
            d = nu.abs(position1-position2)
            Msum += MutalInductance(r1, r2, d)

    return Msum


def _nagaoka(r, l):
    # k = nu.sqrt(1/( (l/(2*r))**2 + 1 ))
    # result = 4/(3*nu.pi*nu.sqrt(1-k**2)) * ( (1-k**2)/k**2 * ellipk(k**2) - (1-2*k**2)/k**2 * ellipe(k**2) - k )
    # return result
    m = 1/( (l/(2*r))**2 + 1)
    result = 4/(3*nu.pi*nu.sqrt(1-m)) * ( (1-m)/m * ellipk(m) - (1-2*m)/m * ellipe(m) - nu.sqrt(m) )
    return result


def L(r, l, N):
    return _nagaoka(r, l) * mu0 * N**2 * nu.pi * r**2 / l
