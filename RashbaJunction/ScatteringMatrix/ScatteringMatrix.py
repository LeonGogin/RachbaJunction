import numpy as np
import RashbaJunction.utilities as ut
from RashbaJunction.RashbaJunction_0_4 import NDArray
from tabulate import tabulate
from typing_extensions import Self


class ScatteringMatrix:
    """
    Class that handle the calculation of the scattering matrix from the transfer matrix

    A classmethods are used to compute the appropriate scattering matrix for in-gap and out-gap energy ranges
    """

    unity_tol = 1e-9

    def __init__(self, s: NDArray, c: NDArray, logg=False):

        # wave function coefficients
        self.C = c
        # Scattering matrix
        self.S = s
        # check if scattering matrix is unitary
        if not np.allclose(
            np.matmul(s.T.conj(), s), np.eye(s.shape[0]), atol=self.unity_tol
        ):
            self.is_unitary = True
            print("Scattering matrix isn't unitary")
        else:
            self.is_unitary = False

    @property
    def t_coef(self):
        """
        Transmission coefficien
        """
        indd = int(self.S.shape[0] / 2)
        t = self.S[0:indd, indd:]
        # is calculating squares norm
        return np.linalg.norm(np.trace(np.matmul(t.T.conj(), t)))  # /indd

    @property
    def r_coef(self):
        """
        Reflection coefficient
        """
        indd = int(self.S.shape[0] / 2)
        r = self.S[0:indd, 0:indd]
        # is calculating squares norm
        return np.linalg.norm(np.trace(np.matmul(r.T.conj(), r)))  # /indd

    @classmethod
    def in_gap(cls, M_tot: NDArray, vellocity: NDArray, logg=False) -> Self:
        """
        (M_tot: np.ndarray((2, 2), np.complex256), vellocity: np.ndarray((2, 2), np.complex256), logg=False) -> ScatteringMatrix

        compute the scattering matrix inside the gap
        """
        sig_11 = (-M_tot[1, 0] * M_tot[2, 2] + M_tot[1, 2] * M_tot[2, 0]) / (
            M_tot[2, 2] * M_tot[1, 1] - M_tot[1, 2] * M_tot[2, 1]
        )

        sig_12 = M_tot[2, 2] / (M_tot[2, 2] * M_tot[1, 1] - M_tot[1, 2] * M_tot[2, 1])

        sig_21 = (
            M_tot[0, 0]
            + M_tot[0, 1] * sig_11
            - M_tot[0, 2] * (M_tot[2, 0] + M_tot[2, 1] * sig_11) / M_tot[2, 2]
        )

        sig_22 = (M_tot[0, 1] - M_tot[0, 2] * M_tot[2, 1] / M_tot[2, 2]) * sig_12

        sig_31 = -(M_tot[2, 0] + M_tot[2, 1] * sig_11) / M_tot[2, 2]

        sig_32 = -M_tot[2, 1] / M_tot[2, 2] * sig_12

        sig_41 = M_tot[3, 0] + M_tot[3, 1] * sig_11 + M_tot[3, 2] * sig_31

        sig_42 = M_tot[3, 1] * sig_12 + M_tot[3, 2] * sig_32

        c = np.array(
            [[sig_11, sig_12], [sig_21, sig_22], [sig_31, sig_32], [sig_41, sig_42]]
        )

        s = np.array([[sig_11, sig_12], [sig_21, sig_22]]) * vellocity

        return cls(s, c, logg)

    @classmethod
    def out_gap(cls, M_tot, vellocity, logg=False):
        """
        (M_tot: np.ndarray((4, 4), np.complex256), vellocity: np.ndarray((2, 2), np.complex256), logg=False) -> ScatteringMatrix

        compute the scattering matrix out of the gap
        """
        ft = (
            M_tot[1, 1]
            * M_tot[3, 3]
            / (M_tot[1, 1] * M_tot[3, 3] - M_tot[1, 3] * M_tot[3, 1])
        )

        sig_11 = (
            ft
            * (M_tot[1, 3] * M_tot[3, 0] - M_tot[1, 0] * M_tot[3, 3])
            / (M_tot[1, 1] * M_tot[3, 3])
        )

        sig_12 = (
            ft
            * (M_tot[1, 3] * M_tot[3, 2] - M_tot[1, 2] * M_tot[3, 3])
            / (M_tot[1, 1] * M_tot[3, 3])
        )

        sig_13 = ft / M_tot[1, 1]

        sig_14 = -ft * M_tot[1, 3] / (M_tot[1, 1] * M_tot[3, 3])

        sig_21 = -(M_tot[3, 0] + M_tot[3, 1] * sig_11) / M_tot[3, 3]

        sig_22 = -(M_tot[3, 1] * sig_12 + M_tot[3, 2]) / M_tot[3, 3]

        sig_23 = -M_tot[3, 1] * sig_13 / M_tot[3, 3]

        sig_24 = (1 - M_tot[3, 1] * sig_14) / M_tot[3, 3]

        sig_31 = M_tot[0, 0] + M_tot[0, 1] * sig_11 + M_tot[0, 3] * sig_21

        sig_32 = M_tot[0, 2] + M_tot[0, 1] * sig_12 + M_tot[0, 3] * sig_22

        sig_33 = M_tot[0, 1] * sig_13 + M_tot[0, 3] * sig_23

        sig_34 = M_tot[0, 1] * sig_14 + M_tot[0, 3] * sig_24

        sig_41 = M_tot[2, 0] + M_tot[2, 1] * sig_11 + M_tot[2, 3] * sig_21

        sig_42 = M_tot[2, 2] + M_tot[2, 1] * sig_12 + M_tot[2, 3] * sig_22

        sig_43 = M_tot[2, 1] * sig_13 + M_tot[2, 3] * sig_23

        sig_44 = M_tot[2, 1] * sig_14 + M_tot[2, 3] * sig_24

        s = np.array(
            [
                [sig_11, sig_12, sig_13, sig_14],
                [sig_21, sig_22, sig_23, sig_24],
                [sig_31, sig_32, sig_33, sig_34],
                [sig_41, sig_42, sig_43, sig_44],
            ]
        )
        return cls(vellocity * s, s, logg)

    @classmethod
    def insulator(cls, *arg, **karg):
        """
        Place holder method  that is used only if all the modes are evanescent
        In this case the transmission coefficient ca not be defined therefore here is used an arbitrary non unitary matrix
        """
        return cls(np.array([[1, 0], [4, 4]]), np.array([[1, 0], [4, 4]]))

    def __str__(self):
        strr = f"scattering matrix\n\treal part\n{tabulate(self.S.real)}\n\
\timmaginary part\n{tabulate(self.S.imag)}\n\
inverse vs complex conjugated: {np.allclose(np.linalg.inv(self.S), self.S.conj(), atol=1e-4)}\n\
inverse vs transpose complex conjugated: {np.allclose(np.linalg.inv(self.S), self.S.T.conj(), atol=1e-4)}\n\n\
trasmission coefficient: (0,1);(1,0)\n{self.t_coef}\n\n\
reflection coefficient:(0,0);(1,1)\n{self.r_coef}\n\n\
sum: {self.r_coef + self.t_coef}"
        return strr
