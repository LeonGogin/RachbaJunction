import logging
import sys

import numpy as np
from matplotlib import ticker

# from scipy import linalg
from tabulate import tabulate


def renormallize_result(res, dy=10):
    #     x_int = x
    #     res_int = res
    for v in res.values():
        for i in v:
            pos = np.where(np.abs(np.diff(i)) >= dy)[0]
            if len(pos) != 0:
                i[pos] = np.nan
            pos = 0


def adjuct_Tick(axs, **kwarg):

    if len(axs.shape) == 1:
        ax = axs.reshape(axs.shape[0], 1)
        for k in kwarg.keys():
            if k != "size":
                kwarg[k] = kwarg[k].reshape(axs.shape[0], 1)
    else:
        ax = axs

    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i, j].tick_params(axis="x", labelsize=kwarg["size"])
            ax[i, j].tick_params(axis="y", labelsize=kwarg["size"])

            ax[i, j].xaxis.set_major_locator(
                ticker.MultipleLocator(kwarg["x_magior"][i, j])
            )
            ax[i, j].xaxis.set_minor_locator(
                ticker.MultipleLocator(kwarg["x_minor"][i, j])
            )

            ax[i, j].xaxis.set_ticks_position("both")

            ax[i, j].yaxis.set_major_locator(
                ticker.MultipleLocator(kwarg["y_magior"][i, j])
            )
            ax[i, j].yaxis.set_minor_locator(
                ticker.MultipleLocator(kwarg["y_minor"][i, j])
            )

            ax[i, j].yaxis.set_ticks_position("both")

            ax[i, j].tick_params(which="major", width=1.00, length=5)
            ax[i, j].tick_params(which="minor", width=0.75, length=2.5, labelsize=15)


def get_loger(name):
    log = logging.getLogger(__name__)  # .addHandler(logging.NullHandler())
    # logger.basicConfig(format="%(levelname)s - %(message)s")
    log.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter("%(levelname)s - %(message)s")

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    log.addHandler(ch)

    return log


def set_zeros(a):
    tol = np.finfo(a.dtype).eps
    a.real[abs(a.real) < tol] = 0.0
    a.imag[abs(a.imag) < tol] = 0.0
    return a


def scater_matrix_iunfo(x):
    # sig = vel_factor*sig

    print("scattering matrix")
    print("\treal part")
    print(tabulate(x.real))
    print("\timmaginary part")
    print(tabulate(x.imag))

    # print("inverse vs complex conjugated")
    # print(np.linalg.inv(x) - x.conj().T, "\n\n")

    # print("complex conjugated")
    # print(sig.conj().T, "\n\n")
    # testt = np.matmul(sig.conj(), sig)
    # print(set_zeros(testt), "\n")
    # print(testt, "\n")

    # print(f"inverse vs complex conjugated: {np.allclose(linalg.inv(x), x.conj())}")
    print(
        f"inverse vs complex conjugated: {np.allclose(np.matmul(x.conj(), x), np.eye(x.shape[0]))}"
    )

    # print(f"inverse vs transpose complex conjugated: {np.allclose(linalg.inv(x), x.T.conj())}")
    print(
        f"inverse vs transpose complex conjugated: {np.allclose(np.matmul(x.T.conj(), x), np.eye(x.shape[0]))}"
    )

    print("\n\n")

    print("trasmission coefficient: (0,1);(1,0)")
    indd = int(x.shape[0] / 2)
    t = x[0:indd, indd:]
    tt = np.linalg.norm(np.trace(np.matmul(t.T.conj(), t))) / indd
    print(tt)

    t = x[indd:, 0:indd]
    tt1 = np.linalg.norm(np.trace(np.matmul(t.T.conj(), t))) / indd
    print(tt1)

    print("\n\n")
    print("reflection coefficient:(0,0);(1,1)")
    indd = int(x.shape[0] / 2)
    r = x[0:indd, 0:indd]
    rr = np.linalg.norm(np.trace(np.matmul(r.T.conj(), r))) / indd
    print(rr)

    indd = int(x.shape[0] / 2)
    r = x[indd:, indd:]
    rr1 = np.linalg.norm(np.trace(np.matmul(r.T.conj(), r))) / indd
    print(rr1)

    print("\n\n")

    print(rr + tt)
    print(rr + tt1)
    print(rr1 + tt)
    print(rr + tt1)


def make_grid(rng, par, funk):
    """
    evaluate function over a domain and grid fo parameters 
        rng: Iterable --> domain of the function
        par: Iterable --> parameters; or a tuple of parameters
        funk: Iterable --> function to efaluate
        
        return Dictionary(par: array(len(funk), len(rng)))
    """
    res = {key: np.zeros((len(funk), len(rng)), dtype=object) for key in par}
    for ap in par:
        for ij, i in enumerate(rng):
            for fj, f in enumerate(funk):
                res[ap][fj, ij] = f(i, ap)
    return res
