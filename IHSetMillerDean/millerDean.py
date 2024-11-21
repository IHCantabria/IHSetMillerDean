import numpy as np
from numba import jit

@jit
def millerDean(Hb, depthb, sl, wast, dt, Hberm, Y0, kero, kacr, Yini, flagP=1, Omega=0):
    if flagP == 1:
        kero_ = np.zeros_like(Hb)+kero
        kacr_ = np.zeros_like(Hb)+kacr
    elif flagP == 2:
        kero_ = Hb ** 2 * kero
        kacr_ = Hb ** 2 * kacr
    elif flagP == 3:
        kero_ = Hb ** 3 * kero
        kacr_ = Hb ** 3 * kacr
    elif flagP == 4:
        kero_ = Omega * kero
        kacr_ = Omega * kacr

    yeq = np.zeros_like(Hb)
    Y = np.zeros_like(Hb)
    wl = 0.106 * Hb + sl
    yeq = Y0 - wast * wl / (Hberm + depthb)

    Y[0] = Yini

    for i in range(1, len(Hb)):
        if Y[i-1] < yeq[i]:
            # A = kacr_[i] * dt * 0.5
            # Y[i] = (Y[i - 1] + A * (yeq[i] + yeq[i - 1] - Y[i - 1])) / (1 + A)
            Y[i] = Y[i-1] + kacr_[i] * dt[i-1] * (yeq[i] - Y[i-1])
        else:
            # A = kero_[i] * dt * 0.5
            # Y[i] = (Y[i - 1] + A * (yeq[i] + yeq[i - 1] - Y[i - 1])) / (1 + A)
            Y[i] = Y[i-1] + kero_[i] * dt[i-1] * (yeq[i] - Y[i-1])

    return Y, yeq