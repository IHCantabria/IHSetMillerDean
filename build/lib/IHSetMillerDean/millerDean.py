import numpy as np
from numba import jit
from IHSetUtils import wast

@jit
def millerDean(Hb, depthb, sl, wast, dt, Hberm, Y0, kero, kacr, Yini, flagP=1, Omega=0):
    if flagP == 1:
        kero = np.full_like(Hb, kero)
        kacr = np.full_like(Hb, kacr)
    elif flagP == 2:
        kero *= Hb ** 2
        kacr *= Hb ** 2
    elif flagP == 3:
        kero *= Hb ** 3
        kacr *= Hb ** 3
    elif flagP == 4:
        kero *= Omega
        kacr *= Omega

    yeq = np.zeros_like(Hb)
    Y = np.zeros_like(Hb)
    wl = 0.106 * Hb + sl
    yeq = Y0 - wast * wl / (Hberm + depthb)

    Y[0] = Yini

    for i in range(1, len(Hb)):
        if Y[i] < yeq[i]:
            A = kacr * dt * 0.5
            Y[i] = (Y[i - 1] + A * (yeq[i] + yeq[i - 1] - Y[i - 1])) / (1 + A)
        else:
            A = kero * dt * 0.5
            Y[i] = (Y[i - 1] + A * (yeq[i] + yeq[i - 1] - Y[i - 1])) / (1 + A)

    return Y


    return Y, Seq