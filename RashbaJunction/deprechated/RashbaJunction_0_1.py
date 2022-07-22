import logging

import numpy as np
import scipy.constants as cc

import RashbaJunction.utilities as ut

# from RachbaJunction.ScatteringMatrix.ScatteringMatrix import ScatteringMatrix
from RashbaJunction.ScatteringMatrix import ScatteringMatrix
from RashbaJunction.utilities import set_zeros

logger = ut.get_loger(__name__)

hbar = 6.582119569e-16


class WaveFunction:
    def __init__(self, h):

        self.h_xy = 0.2e-2  # h[0]
        self.phi = h[1]

        self.vel_a = []
        self.vel_b = []

        # should be overwritten
        self.E_so = 0.0
        self.sgn_alpha = -1

        self.mod = []
        self.band = []
        self.l = []
        self.wave_length = []

    @property
    def sgn_k(self):
        # return np.float64(self.sgn_alpha*np.sign(self.wave_length))
        return self.sgn_alpha * np.sign(self.wave_length)

    def k_E(self, e, l):
        return (
            2
            * self.sgn_alpha
            * np.sqrt(self.E_so)
            * np.sqrt(
                e
                + 2 * self.E_so
                + l * np.sqrt(4 * e * self.E_so + 4 * self.E_so ** 2 + 1)
            )
        )

    def q_E(self, e, l):
        return (
            2
            * self.sgn_alpha
            * np.sqrt(self.E_so)
            * np.sqrt(
                -(e + 2 * self.E_so)
                + l * np.sqrt(4 * e * self.E_so + 4 * self.E_so ** 2 + 1)
            )
        )

    def E_0(self, E, l, m):
        res = 0.0
        if m == "k":
            res = (
                E
                + 2 * self.E_so
                + l * np.sqrt(4 * E * self.E_so + 4 * self.E_so ** 2 + 1)
            )
        elif m == "q":
            res = -(E + 2 * self.E_so) + l * np.sqrt(
                4 * E * self.E_so + 4 * self.E_so ** 2 + 1
            )
        else:
            logger.warning("Wrong wave function mode lable (in E_0)")

        return res

    def omega_k(self, x, k, b):
        # res = np.array([np.exp(complex(0,-self.phi))*(k - b*np.sqrt(k**2 + 1)), 1], dtype=np.complex128)
        norm = np.sqrt(1 / (2 * np.sqrt(1 + k ** 2)))
        res = norm * np.array(
            [
                -b
                * np.exp(complex(0, -self.phi))
                * np.sqrt(-b * k + np.sqrt(k ** 2 + 1)),
                1 / np.sqrt(-b * k + np.sqrt(k ** 2 + 1)),
            ],
            dtype=np.complex128,
        )

        # expp = np.exp(complex(0, self.sgn_alpha*k*x))
        # res = 1/np.linalg.norm(res)*res*expp
        return res * np.exp(complex(0, self.sgn_alpha * k * x))

    def omega_q(self, x, q, b):
        norm = np.sqrt(1 / 2)
        res = norm * np.array(
            [
                np.exp(complex(0, -self.phi))
                * (-complex(0, q) - b * np.sqrt(1 - q ** 2)),
                1,
            ],
            dtype=np.complex128,
        )
        # res = 1/np.linalg.norm(res) * res*np.exp(self.sgn_alpha*q*x)
        return res * np.exp(self.sgn_alpha * q * x)

    def flux(self, E, w_func):
        res = []
        for i in range(w_func.shape[0]):
            if self.mod[i] == "k":

                vel_op = self.sgn_k[i] * np.sqrt(
                    self.E_0(E, self.l[i], self.mod[i])
                ) * np.eye(2) - self.sgn_alpha * np.sqrt(self.E_so) * np.array(
                    [[1, 0], [0, -1]]
                )
            elif self.mod[i] == "q":

                vel_op = -(
                    complex(0, self.sgn_k[i])
                    * np.sqrt(self.E_0(E, self.l[i], self.mod[i]))
                    * np.eye(2)
                    + self.sgn_alpha * np.sqrt(self.E_so) * np.array([[1, 0], [0, -1]])
                )
            else:
                logger.warning("Wrong wave function mode lable (in flux)")

            # vel_op = np.sqrt(self.h_xy)*vel_op

            res.append(np.dot(vel_op, w_func[i]))
        return np.array(res, dtype=np.complex128)

    def compile_wave_function(self, x, E, v=False):
        res = []
        if v == True:
            self.calculate_velocity(E)
        for k, m, b in zip(self.wave_length, self.mod, self.band):
            if m == "k":
                # logger.debug(f"{b}, {k}")
                res.append(self.omega_k(x, k, b))
            elif m == "q":
                res.append(self.omega_q(x, k, b))
        return np.array(res, dtype=np.complex128)

    def calculate_velocity(self, E):
        # [rigth lead velocity, left lead velocity]
        if len(self.vel_a) == 0 or len(self.vel_b) == 0:
            # if len(self.vel_a)%2 == 0 or len(self.vel_b)%2 == 0:
            # In rigth lead: injected coefficient(a) is associated with negative k
            #               reflected coefficient(b) is associated with positive k
            logger.debug("\t\trigth lead")
            k_s = -1
        else:
            # In left lead: injected coefficient(a) is associated with positve k
            #               reflected coefficient(b) is associated with negative k
            logger.debug("\t\tleft lead")
            k_s = 1

        vel = 0.0

        for i in range(len(self.wave_length)):
            k = self.wave_length[i]

            if k_s * self.sgn_k[i] > 0 and self.mod[i] == "k":
                # vel = (self.sgn_k[i]*np.sqrt(self.E_0(E, self.l[i], self.mod[i])*(1+k**2)) + self.band[i]*self.sgn_alpha*k*np.sqrt(self.E_so))/np.sqrt(1+k**2)
                vel = (
                    self.sgn_k[i]
                    * (
                        np.sqrt(self.E_0(E, self.l[i], self.mod[i]) * (1 + k ** 2))
                        + self.band[i] * np.abs(k) * np.sqrt(self.E_so)
                    )
                    / np.sqrt(1 + k ** 2)
                )

                logger.debug(
                    f"\t\t{i}: lambda {self.l[i]}, band {self.band[i]}, sign k: {self.sgn_k[i]}"
                )
                logger.debug(f"\t\t-->a - velcity {vel}")

                self.vel_a.append(vel)
            elif k_s * self.sgn_k[i] < 0 and self.mod[i] == "k":
                # vel = (self.sgn_k[i]*np.sqrt(self.E_0(E, self.l[i], self.mod[i])*(1+k**2)) + self.band[i]*self.sgn_alpha*k*np.sqrt(self.E_so))/np.sqrt(1+k**2)
                vel = (
                    self.sgn_k[i]
                    * (
                        np.sqrt(self.E_0(E, self.l[i], self.mod[i]) * (1 + k ** 2))
                        + self.band[i] * np.abs(k) * np.sqrt(self.E_so)
                    )
                    / np.sqrt(1 + k ** 2)
                )

                logger.debug(
                    f"\t\t{i}: lambda {self.l[i]}, band {self.band[i]}, sign k: {self.sgn_k[i]}"
                )
                logger.debug(f"\t\t-->b - velcity {vel}")

                self.vel_b.append(vel)


class RashbaJunction(WaveFunction):
    _m = 0.022 * cc.m_e

    def __init__(self, profile=[[], []], h=[0, 0], logg=False):
        """
        RachbaJunction(self, profile, h, logg = False)
            h ia an array with [h_xy, h_z, phi_xy]
            profile in an array of array [[interface position], alpha profile]
                alpha profile is given in term of E'_so = E_so/H_x
        """
        if logg:
            logger.disabled = False
        else:
            logger.disabled = True
        # if logg:
        #     logger.setLevel(logging.DEBUG)
        # else:
        #     logger.setLevel(logging.WARNING)
        super(RachbaJunction, self).__init__(h)
        # h ia an array with [h_xy, h_z, phi_xy]
        # self.h_xy = h[0]
        # self.phi = h[1]

        # list of tuples (x, alpha)
        self.interface = profile[0]
        self.alpha_profile = profile[1]

        # self._alpha = 0.0
        if len(profile[1]) != 0:
            self.E_so = profile[1][0]

        self.vell = []

        self.scattering_matrix = None

    @property
    def E_so(self):
        return self._E_so

    @E_so.setter
    def E_so(self, e_so):
        self._E_so = np.abs(e_so)
        self._alpha = np.float64(np.sign(e_so))

    @property
    def sgn_alpha(self):
        return self._alpha

    @sgn_alpha.setter
    def sgn_alpha(self, a):
        self._alpha = a

    def __getitem__(self, ind):
        return self.alpha_profile[ind]

    def __setitem__(self, ind, item):
        if ind < len(self.alpha_profile):
            self.alpha_profile[ind] = item
        else:
            raise IndexError()

    def get_transfer_matrix(self, E):
        # depend on alpha
        self.vel_a = []
        self.vel_b = []
        M = np.eye(4, dtype=np.complex128)
        n = len(self.interface)
        for i in range(len(self.interface)):
            # x_0 is the first transition
            # x_{N+1} is an infinite --> matter only lambda_{N+1}
            if n - i - 1 == len(self.interface) - 1:
                # rigth lead velocity
                v = True
            else:
                v = False

            self.E_so = self.alpha_profile[n - i]
            logger.debug(
                f" {n-i-1}: X_i+ boundary matrix\n\tE_so = {self.E_so}\n\tx_i = {self.interface[n-i-1]}\n\talpha sign = {self.sgn_alpha}\n\th_xy = {self.h_xy}"
            )
            W_pls = self.get_boundary_matrix(self.interface[n - i - 1], E, v=v)

            if n - i - 1 == 0:
                # rigth lead velocity
                v = True
            else:
                v = False
            self.E_so = self.alpha_profile[n - i - 1]
            logger.debug(
                f" {n-i-1}: X_i- boundary matrix\n\tE_so = {self.E_so}\n\tx_i = {self.interface[n-i-1]}\n\talpha sign = {self.sgn_alpha}\n\th_xy = {self.h_xy}"
            )
            W_min = self.get_boundary_matrix(self.interface[n - i - 1], E, v=v)

            # W_min = set_zeros(W_min)
            # W_pls = set_zeros(W_pls)

            M = np.matmul(M, np.matmul(np.linalg.inv(W_pls), W_min))
            # M = np.matmul(np.linalg.inv(W_pls), W_min)

        # [left lead velocity, rigth lead velocity]
        self.vel_a = np.array(self.vel_a)
        self.vel_b = np.array(self.vel_b)
        if len(self.vel_a) == 2:
            self.vel_a = np.array(self.vel_a[::-1])
            self.vel_b = np.array(self.vel_b[::-1])
        elif len(self.vel_a) == 4:
            tmp = np.array(self.vel_a[0:2])
            self.vel_a[0:2] = self.vel_a[2:4]
            self.vel_a[2:4] = tmp

            tmp = np.array(self.vel_b[0:2])
            self.vel_b[0:2] = self.vel_b[2:4]
            self.vel_b[2:4] = tmp

        # collumn vector
        v_b = np.lib.scimath.sqrt(self.vel_b.reshape((len(self.vel_b), 1)))
        # row vector
        v_a = 1 / np.lib.scimath.sqrt(self.vel_a.reshape((1, len(self.vel_a))))
        vel_factor_mat = np.dot(v_b, v_a)
        return M, vel_factor_mat

    def get_scattering_matrix(self, E, logg=False):
        M, vell = self.get_transfer_matrix(E)

        res = self.scattering_matrix(M, vell, logg)

        return res

    def get_boundary_matrix(self, x, E, v=False):
        if 1 / 2 < self.E_so:
            logger.info("-->Rashba regime")
            self.prepare_rashba_WF(x, E, v=v)
        elif 1 / 4 < self.E_so and self.E_so < 1 / 2:
            logger.info("-->Week zeeman regime")
            self.prepare_week_zeeman_WF(x, E, v=v)
        elif self.E_so < 1 / 4:
            logger.info("-->Zeeman regime")
            self.prepare_zeeman_WF(x, E, v=v)

        w_func = self.compile_wave_function(x, E, v=v)
        flux = self.flux(E, w_func)
        return np.transpose(np.append(w_func, flux, axis=1))

    def prepare_rashba_WF(self, x, E, v=False):

        if (
            -self.E_so * (1 + (1 / (2 * self.E_so + np.finfo(np.float64).eps)) ** 2)
            < E
            < -1
        ):
            logger.info("\tunder the gap energy range")
            self.scattering_matrix = (
                ScatteringMatrix.above_gap if v and len(self.vel_a) != 0 else None
            )

            # E_0(self, E, l, m)
            self.l = [+1, +1, -1, -1]
            self.wave_length = [
                self.k_E(E, +1),
                -self.k_E(E, +1),
                -self.k_E(E, -1),
                self.k_E(E, -1),
            ]

            self.band = [-1, -1, -1, -1]
            self.mod = 4 * ["k"]

        elif -1 < E < -1 / (4 * self.E_so + np.finfo(np.float64).eps):
            logger.info("\tfirst in the gap energy range")
            self.scattering_matrix = (
                ScatteringMatrix.in_gap if v and len(self.vel_a) != 0 else None
            )

            self.l = [+1, +1, +1, +1]
            self.wave_length = [
                self.k_E(E, +1),
                -self.k_E(E, +1),
                self.q_E(E, +1),
                -self.q_E(E, +1),
            ]
            self.band = [-1, -1, -1, -1]
            self.mod = 2 * ["k"] + 2 * ["q"]

        elif -1 / (4 * self.E_so + np.finfo(np.float64).eps) < E < 1:
            logger.info("\tsecond in the gap energy range")
            self.scattering_matrix = (
                ScatteringMatrix.in_gap if v and len(self.vel_a) != 0 else None
            )

            self.l = [+1, +1, +1, +1]
            self.wave_length = [
                self.k_E(E, +1),
                -self.k_E(E, +1),
                self.q_E(E, +1),
                -self.q_E(E, +1),
            ]
            self.band = [-1, -1, +1, +1]
            self.mod = 2 * ["k"] + 2 * ["q"]

        elif 1 <= E:
            logger.info("\tout of gap energy range")
            self.scattering_matrix = (
                ScatteringMatrix.above_gap if v and len(self.vel_a) != 0 else None
            )

            self.l = [+1, +1, -1, -1]
            # self.l = [+1, +1, -1, -1]
            self.wave_length = [
                self.k_E(E, +1),
                -self.k_E(E, +1),
                self.k_E(E, -1),
                -self.k_E(E, -1),
            ]
            self.band = [-1, -1, +1, +1]
            self.mod = 4 * ["k"]
        else:
            logger.warning("out of range energy")

    def prepare_week_zeeman_WF(self, x, E, v=False):

        if (
            -self.E_so * (1 + (1 / (2 * self.E_so + np.finfo(np.float64).eps)) ** 2)
            < E
            < -1
        ):
            logger.info("\tunder the gap energy range")
            logger.warning(f"\tonly tunelings modes {self.E_so}")

            self.l = [+1, +1, -1, -1]
            self.band = [-1, -1, -1, -1]
            self.wave_length = [
                self.q_E(E, +1),
                -self.q_E(E, +1),
                self.q_E(E, -1),
                -self.q_E(E, -1),
            ]
            self.mod = 4 * ["q"]

        elif -1 < E < -1 / (4 * self.E_so + np.finfo(np.float64).eps):
            logger.info("\tfirst in the gap energy range")
            self.scattering_matrix = (
                ScatteringMatrix.in_gap if v and len(self.vel_a) != 0 else None
            )

            self.l = [+1, +1, +1, +1]
            self.band = [-1, -1, -1, -1]
            self.wave_length = [
                self.k_E(E, +1),
                -self.k_E(E, +1),
                self.q_E(E, +1),
                -self.q_E(E, +1),
            ]
            self.mod = 2 * ["k"] + 2 * ["q"]

        elif -1 / (4 * self.E_so + np.finfo(np.float64).eps) < E < 1:
            logger.info("\tsecond in the gap energy range")
            self.scattering_matrix = (
                ScatteringMatrix.in_gap if v and len(self.vel_a) != 0 else None
            )

            self.l = [+1, +1, +1, +1]
            self.band = [-1, -1, +1, +1]
            self.wave_length = [
                self.k_E(E, +1),
                -self.k_E(E, +1),
                self.q_E(E, +1),
                -self.q_E(E, +1),
            ]
            self.mod = 2 * ["k"] + 2 * ["q"]

        elif 1 <= E:
            logger.info("\tout of gap energy range")
            self.scattering_matrix = (
                ScatteringMatrix.above_gap if v and len(self.vel_a) != 0 else None
            )

            self.l = [+1, +1, -1, -1]
            self.band = [-1, -1, +1, +1]
            self.wave_length = [
                self.k_E(E, +1),
                -self.k_E(E, +1),
                self.k_E(E, -1),
                -self.k_E(E, -1),
            ]
            self.mod = 4 * ["k"]

    def prepare_zeeman_WF(self, x, E, v=False):

        if (
            -self.E_so * (1 + (1 / (2 * self.E_so + np.finfo(np.float64).eps)) ** 2)
            < E
            < -1 / (4 * self.E_so + np.finfo(np.float64).eps)
        ):
            logger.info("\tunder the gap energy range")
            logger.warning(f"\tonly tunelings modes {self.E_so}")

            self.l = [+1, +1, -1, -1]
            self.band = [-1, -1, -1, -1]
            self.wave_length = [
                self.q_E(E, +1),
                -self.q_E(E, +1),
                self.q_E(E, -1),
                -self.q_E(E, -1),
            ]
            self.mod = 4 * ["q"]

        elif -1 / (4 * self.E_so + np.finfo(np.float64).eps) < E < -1:
            logger.info("\tfirst in the gap energy range")
            # self.scattering_matrix = ScatteringMatrix.in_gap if v and len(self.vel_a) != 0 else None

            self.l = [+1, +1, -1, -1]
            self.band = [+1, +1, -1, -1]
            self.wave_length = [
                self.q_E(E, +1),
                -self.q_E(E, +1),
                self.q_E(E, -1),
                -self.q_E(E, -1),
            ]
            self.mod = 4 * ["q"]

        elif -1 < E < 1:
            logger.info(f"\tin the gap energy range {v}, {len(self.vel_a)}")
            self.scattering_matrix = (
                ScatteringMatrix.in_gap if v and len(self.vel_a) != 0 else None
            )

            self.l = [+1, +1, +1, +1]
            self.band = [-1, -1, +1, +1]
            self.wave_length = [
                self.k_E(E, +1),
                -self.k_E(E, +1),
                self.q_E(E, +1),
                -self.q_E(E, +1),
            ]
            self.mod = 2 * ["k"] + 2 * ["q"]

        elif 1 <= E:
            logger.info("\tout of gap energy range")
            self.scattering_matrix = (
                ScatteringMatrix.above_gap if v and len(self.vel_a) != 0 else None
            )

            self.l = [+1, +1, -1, -1]
            self.band = [-1, -1, +1, +1]
            self.wave_length = [
                self.k_E(E, +1),
                -self.k_E(E, +1),
                self.k_E(E, -1),
                -self.k_E(E, -1),
            ]
            self.mod = 4 * ["k"]

