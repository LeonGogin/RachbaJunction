import math

import numpy as np

from RashbaJunction.RashbaJunction_0_4 import RashbaJunction, WaveVector
from RashbaJunction.ScatteringMatrix import ScatteringMatrix


class PureRashba(RashbaJunction):
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)

    @property
    def sgn_k(self):
        if self.wave_vector[1][0] == 0:
            sng1 = math.copysign(1, self.wave_vector[1][0])
            sng2 = math.copysign(1, self.wave_vector[2][0])

            return np.array(
                [
                    np.sign(self.wave_vector[0]),
                    [sng1, sng1],
                    [sng2, sng2],
                    np.sign(self.wave_vector[-1]),
                ]
            )
        else:
            return np.sign(self.wave_vector)

    def k_alpha(self, E, l, m):
        return np.array(
            [
                np.sqrt(self.E_so) * (self.sgn_alpha + l * np.sqrt(1 + E / self.E_so)),
                # k/k_so
                np.sqrt(self.E_so) * (self.sgn_alpha + l * np.sqrt(1 + E / self.E_so)),
            ]
        )

    def E_0(self, E, l, m):
        return self.E_so * (self.sgn_alpha + l * np.sqrt(1 + E / self.E_so)) ** 2

    def omega_k(self, x, k, b):
        if b == -1:
            res = np.array([1, 0], dtype=np.complex256)
        else:
            res = np.array([0, 1], dtype=np.complex256)
        return res * np.exp(complex(0, k[1] * x))

    #     def omega_q(self, x, q, b):
    #         pass\
    def prepare_rashba_WF(self, E, v=False):
        self.scattering_matrix = (
            ScatteringMatrix.above_gap if v and len(self.vel_a) != 0 else None
        )
        #         print(E, self.E_so)
        if E >= -self.E_so:
            self.l = (+1, -1, -1, +1)

            self.band = (-1, -1, +1, +1)
            self.mod = 4 * (WaveVector.k,)

            self.wave_vector = (
                -self.band[0] * self.k_alpha(E, self.l[0], self.mod[0]),
                -self.band[1] * self.k_alpha(E, self.l[1], self.mod[1]),
                -self.band[2] * self.k_alpha(E, self.l[2], self.mod[2]),
                -self.band[3] * self.k_alpha(E, self.l[3], self.mod[3]),
            )
        else:
            raise ValueError

    def prepare_week_zeeman_WF(self, E, v=False):
        self.prepare_rashba_WF(E, v)

    def prepare_zeeman_WF(self, E, v=False):
        self.prepare_rashba_WF(E, v)

    def calculate_velocity(self, E):
        # [rigth lead velocity, left lead velocity]
        if len(self.vel_a) == 0 or len(self.vel_b) == 0:
            # In rigth lead:
            #       injected coefficient(a) is associated with negative k
            #       reflected coefficient(b) is associated with positive k
            #             logger.debug("\t\trigth lead")
            k_s = -1
        else:
            # In left lead:
            #       injected coefficient(a) is associated with positve slope -->0, 2
            #       reflected coefficient(b) is associated with negative slope --> 1, 3
            #             logger.debug("\t\tleft lead")
            k_s = 1
        vel = 0.0

        for i in range(len(self.wave_vector)):
            vel = self.sgn_k[i][1] * np.sqrt(
                self.E_0(E, self.l[i], self.mod[i])
            ) + self.band[i] * self.sgn_alpha * np.sqrt(self.E_so)

            #             vel = np.sqrt(self.E_so) * self.band[i]*(-(self.sgn_alpha + self.l[i]*np.sqrt(1+E/self.E_so)) + self.sgn_alpha)
            #             print(vel)
            if math.copysign(1, k_s * vel) > 0:
                self.vel_a.append(vel)
            else:
                self.vel_b.append(vel)

