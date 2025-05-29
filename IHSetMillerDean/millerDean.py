import numpy as np
from numba import njit, jit

@jit
def millerDean_old(Hb, depthb, sl, wast, dt, Hberm, Y0, kero, kacr, Yini, flagP=1, Omega=0):
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


@njit(nopython=True, fastmath=True, cache=True)
def millerDean(Hb, depthb, sl, wast, dt, Hberm, Y0, kero, kacr, Yini, flagP, Omega):
    n = Hb.shape[0]

    # Precompute kero_ y kacr_ en O(n)
    kero_arr = np.empty(n)
    kacr_arr = np.empty(n)
    if flagP == 1:
        for i in range(n):
            kero_arr[i] = kero
            kacr_arr[i] = kacr
    elif flagP == 2:
        for i in range(n):
            t2 = Hb[i] * Hb[i]
            kero_arr[i] = t2 * kero
            kacr_arr[i] = t2 * kacr
    elif flagP == 3:
        for i in range(n):
            t3 = Hb[i] * Hb[i] * Hb[i]
            kero_arr[i] = t3 * kero
            kacr_arr[i] = t3 * kacr
    else:  # flagP == 4
        for i in range(n):
            kero_arr[i] = Omega[i] * kero
            kacr_arr[i] = Omega[i] * kacr

    wl = 0.106 * Hb + sl
    denom = Hberm + depthb
    yeq = np.empty(n)
    for i in range(n):
        yeq[i] = Y0 - wast[i] * wl[i] / denom[i]

    Y = np.empty(n)
    Y[0] = Yini
    for i in range(1, n):
        prev = Y[i-1]
        cur_eq = yeq[i]
        delta = cur_eq - prev
        # cond = 1 si acreción, 0 si erosión
        cond = 1.0 if prev < cur_eq else 0.0
        k = cond * kacr_arr[i] + (1.0 - cond) * kero_arr[i]
        Y[i] = prev + k * dt[i-1] * delta

    return Y, yeq


def millerDean_njit(Hb, depthb, sl, wast, dt, Hberm, Y0, kero, kacr, Yini, flagP=1, Omega=0):
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