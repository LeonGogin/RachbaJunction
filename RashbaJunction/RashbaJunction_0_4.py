import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple

import numpy as np
import scipy.constants as cc
from scipy import linalg

import RashbaJunction.utilities as ut
from RashbaJunction.ScatteringMatrix import ScatteringMatrix

logger = ut.get_loger(__name__)

# Define type hint
# NDArray = npt.NDArray
NDArray = np.ndarray
# Ctype = np.complex256


class WaveVector(Enum):
    """
    Enumerate all wavevector types
        WaveVector.k -> propagating modes
        WaveVector.q -> evanescent modes
    """

    k = 1
    q = 2


class AbstracWaveFunction(ABC):
    """
    The Abstract class for the wave function. 
    SHOULD NEVER BE INITIATED DIRECTLY, must be used as a parent class

    Define interfaces tha must have any wave function
    """

    def __init__(self):
        self.vel_a = []
        self.vel_b = []

        self.scattering_matrix = None

    @property
    @abstractmethod
    def E_so(self):
        return

    @property
    @abstractmethod
    def sgn_alpha(self):
        return

    @abstractmethod
    def prepare_WF(self, E: float, v=False):
        return

    @abstractmethod
    def calculate_velocity(self, E):
        return

    @abstractmethod
    def compile_wave_function(self, x: float) -> NDArray:
        return

    @abstractmethod
    def flux(self, E: float, w_func: NDArray) -> NDArray:
        return


class WaveFunction(AbstracWaveFunction):
    """
    Difine centrall components of wave function:
        Wawe vector -> k_alpha
        'Zero' Energy -> E_0 = hbar^2 k^2/2 m
        Eigenfunctions -> omega_k, omega_q
        Flux operator -> flux
    Also it provide the interfaces to:
        calculate the velocity -> calculate_velocity
        compute the upper part of the transfer matrix -> compile_wave_function
    
    SHOULD NOT BE INSTANTIATED DIRECTLY:
        must be used only as parent class -> most of the properties are overriden
    """

    # may be seen as realizzation of some interface tha is used in the parent class

    def __init__(self):

        # will be set in the parent class
        self.vel_a = []
        self.vel_b = []

        self._alpha = 0.0
        self._E_so = 0.0
        self.E_Z = 0.0

        self.mod = []
        self.band = []
        self.l = []
        self.wave_vector = []

        self.sgn_k = []

        # method to initiate ScatteringMatrix class
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

    def k_alpha(self, E: float, l: int, m: WaveVector) -> NDArray:
        """
        (E: float, l: int, m: WaveVector) -> np.ndarray(2, float)

        Wavevector fopr energy E, band l (+, -) and mode m (WaveVector.k,  WaveVector.q)
            The difference between propagating and evanescent modes is handeled in E_0
        """
        return np.array(
            [
                # \alpha k/E_Z --> is used ONLY in the expression of the spinors and to calculate velocity
                2 * self.sgn_alpha * np.sqrt(self.E_so) * np.sqrt(self.E_0(E, l, m)),
                # k/k_Z
                np.sqrt(self.E_0(E, l, m)),
            ]
        )

    def E_0(self, E: float, l: int, m: WaveVector) -> float:
        """
        (E: float, l: int, m: WaveVector) -> float

        Compute 'Zero' energy E_0 = hbar^2 k_l(E, E_so)^2/2 m
        """
        res = 0.0
        if m is WaveVector.k:
            res = (
                E
                + 2 * self.E_so
                + l * np.sqrt(4 * E * self.E_so + 4 * self.E_so ** 2 + 1)
            )
        elif m is WaveVector.q:
            res = -(E + 2 * self.E_so) + l * np.sqrt(
                4 * E * self.E_so + 4 * self.E_so ** 2 + 1
            )
        else:
            logger.warning("Wrong wave function mode lable (in E_0)")

        return res

    def omega_k(self, x: float, k: NDArray, b: int) -> NDArray:
        """
        (x: float, k: np.ndarray(2, float), b: int) -> np.ndarray(2, np.complex256)

        Compute the propagating eigenstate at x, with wavevector k and labeled by b(+, -)
        """
        norm = np.sqrt(0.5 / (k[0] ** 2 + 1 - b * k[0] * np.sqrt(1 + k[0] ** 2)))
        res = norm * np.array(
            [(k[0] - b * np.sqrt(k[0] ** 2 + 1)), 1], dtype=np.complex256
        )
        return res * np.exp(complex(0, k[1] * x))

    def omega_q(self, x: float, q: NDArray, b: int) -> NDArray:
        """
        (x: float, k: np.ndarray(2, float), b: int) -> np.ndarray(2, np.complex256)


        Compute the evanescent eigenstate at x, with wavevector k and labeled by b(+, -)
        """
        norm = np.sqrt(1 / 2)
        res = norm * np.array(
            [(-complex(0, q[0]) - b * np.sqrt(1 - q[0] ** 2)), 1], dtype=np.complex256
        )
        return res * np.exp(q[1] * x)

    def flux(self, E: float, w_func: NDArray) -> NDArray:
        """
        (E: float, w_func: np.ndarray((4, 2), np.complex256)) -> np.ndarray((4, 2), np.complex256)

        Compute the flux at the energy E of the wavefunction
            cunstruct the lover part of the boundary matrix
        """
        res = []
        for i in range(w_func.shape[0]):
            if self.mod[i] is WaveVector.k:
                vel_op = self.sgn_k[i] * np.sqrt(
                    self.E_0(E, self.l[i], self.mod[i])
                ) * np.eye(2) - self.sgn_alpha * np.sqrt(self.E_so) * np.array(
                    [[1, 0], [0, -1]]
                )
            elif self.mod[i] is WaveVector.q:
                vel_op = -(
                    complex(0, self.sgn_k[i])
                    * np.sqrt(self.E_0(E, self.l[i], self.mod[i]))
                    * np.eye(2)
                    + self.sgn_alpha * np.sqrt(self.E_so) * np.array([[1, 0], [0, -1]])
                )
            else:
                logger.warning("Wrong wave function mode lable (in flux)")

            res.append(np.dot(np.sqrt(self.E_Z) * vel_op, w_func[i]))
        return np.array(res, dtype=np.complex256)

    def compile_wave_function(self, x: float) -> NDArray:
        """
        (x: float) -> np.ndarray((4, 2), np.complex256)

        Compute the upper part of the boundary matrix at x
            All other components have been prepared in get_boundary_matrix()
        """
        res = []

        for k, m, b in zip(self.wave_vector, self.mod, self.band):
            if m is WaveVector.k:
                res.append(self.omega_k(x, k, b))
            elif m is WaveVector.q:
                res.append(self.omega_q(x, k, b))
        return np.array(res, dtype=np.complex256)

    def calculate_velocity(self, E):
        """
        Calculate the velocity of propagating modes at energy E
            is called only if lead are detected
            the function changte in-place properties of the object
            All other components have been prepared in get_boundary_matrix()
        """
        # [rigth lead velocity, left lead velocity]
        if len(self.vel_a) == 0 or len(self.vel_b) == 0:
            # In rigth lead:
            #       injected coefficient(a) is associated with negative k
            #       reflected coefficient(b) is associated with positive k
            logger.debug("\t\trigth lead")
            k_s = -1
        else:
            # In left lead:
            #       injected coefficient(a) is associated with positve k
            #       reflected coefficient(b) is associated with negative k
            logger.debug("\t\tleft lead")
            k_s = 1

        vel = 0.0

        for i in range(len(self.wave_vector)):
            k = self.wave_vector[i][0]

            if k_s * self.sgn_k[i] > 0 and self.mod[i] is WaveVector.k:
                # incoming modes velocity
                vel = (
                    self.sgn_k[i]
                    * (
                        np.sqrt(self.E_0(E, self.l[i], self.mod[i]) * (1 + k ** 2))
                        + self.band[i] * np.abs(k) * np.sqrt(self.E_so)
                    )
                    / np.sqrt(1 + k ** 2)
                )
                self.vel_a.append(vel)
            elif k_s * self.sgn_k[i] < 0 and self.mod[i] is WaveVector.k:
                # outgoing modes velocity
                vel = (
                    self.sgn_k[i]
                    * (
                        np.sqrt(self.E_0(E, self.l[i], self.mod[i]) * (1 + k ** 2))
                        + self.band[i] * np.abs(k) * np.sqrt(self.E_so)
                    )
                    / np.sqrt(1 + k ** 2)
                )
                self.vel_b.append(vel)

    def prepare_WF(self, E: float, v=False):
        """
        choose the appropriate regime
        for each regime will be choosen the corresponding lables and wavevctors for each energy
        """

        if 1 / 2 < self.E_so:
            logger.info("-->Rashba regime")
            self.prepare_rashba_WF(E, v=v)
        elif 1 / 4 < self.E_so and self.E_so < 1 / 2:
            logger.info("-->Week zeeman regime")
            self.prepare_week_zeeman_WF(E, v=v)
        elif self.E_so < 1 / 4:
            logger.info("-->Zeeman regime")
            self.prepare_zeeman_WF(E, v=v)

    def prepare_rashba_WF(self, E, v=False):
        """
        Prepare the lables for Rashba regime

        Beaware of the case in wich energy match exactly the boundary values

        Order: [incoming modes, outgoing modes]
                [propagatingmodes, evanescent modes]
        """

        if (
            -self.E_so * (1 + (1 / (2 * self.E_so + np.finfo(np.float64).eps)) ** 2)
            < E
            < -1
        ):
            # under the gap energy range -> 4 propagating modes
            logger.info("\tunder the gap energy range")
            self.scattering_matrix = (
                ScatteringMatrix.out_gap if v and len(self.vel_a) != 0 else None
            )

            self.l = (+1, +1, -1, -1)
            self.mod = 4 * (WaveVector.k,)
            self.wave_vector = (
                self.k_alpha(E, self.l[0], self.mod[0]),
                -self.k_alpha(E, self.l[1], self.mod[1]),
                -self.k_alpha(E, self.l[2], self.mod[2]),
                self.k_alpha(E, self.l[3], self.mod[3]),
            )
            self.band = (-1, -1, -1, -1)

        elif -1 < E < -1 / (4 * self.E_so + np.finfo(np.float64).eps):
            # inside the gap energy range -> 2 propagating modes
            logger.info("\tfirst in the gap energy range")
            self.scattering_matrix = (
                ScatteringMatrix.in_gap if v and len(self.vel_a) != 0 else None
            )

            self.l = (+1, +1, +1, +1)
            self.band = (-1, -1, -1, -1)
            self.mod = 2 * (WaveVector.k,) + 2 * (WaveVector.q,)

            self.wave_vector = (
                self.k_alpha(E, self.l[0], self.mod[0]),
                -self.k_alpha(E, self.l[1], self.mod[1]),
                self.k_alpha(E, self.l[2], self.mod[2]),
                -self.k_alpha(E, self.l[3], self.mod[3]),
            )

        elif -1 / (4 * self.E_so + np.finfo(np.float64).eps) < E < 1:
            # inside the gap energy range -> 2 propagating modes
            logger.info("\tsecond in the gap energy range")
            self.scattering_matrix = (
                ScatteringMatrix.in_gap if v and len(self.vel_a) != 0 else None
            )

            self.l = (+1, +1, +1, +1)
            self.band = (-1, -1, +1, +1)
            self.mod = 2 * (WaveVector.k,) + 2 * (WaveVector.q,)

            self.wave_vector = (
                self.k_alpha(E, self.l[0], self.mod[0]),
                -self.k_alpha(E, self.l[1], self.mod[1]),
                self.k_alpha(E, self.l[2], self.mod[2]),
                -self.k_alpha(E, self.l[3], self.mod[3]),
            )

        elif 1 <= E:
            # above the gap energy range -> 4 propagating modes
            logger.info("\tout of gap energy range")
            self.scattering_matrix = (
                ScatteringMatrix.out_gap if v and len(self.vel_a) != 0 else None
            )

            self.l = (+1, +1, -1, -1)
            self.band = (-1, -1, +1, +1)
            self.mod = 4 * (WaveVector.k,)

            self.wave_vector = (
                self.k_alpha(E, self.l[0], self.mod[0]),
                -self.k_alpha(E, self.l[1], self.mod[1]),
                self.k_alpha(E, self.l[2], self.mod[2]),
                -self.k_alpha(E, self.l[3], self.mod[3]),
            )
        else:
            logger.warning("out of range energy")
            raise ValueError("out of range energy")
        self.sgn_k = np.sign([i[1] for i in self.wave_vector])

    def prepare_week_zeeman_WF(self, E, v=False):
        """
        Prepare the lables for weak Zeeman regime

        Beaware of the case in wich energy match exactly the boundary values

        Order: [incoming modes, outgoing modes]
                [propagatingmodes, evanescent modes]
        """

        if (
            -self.E_so * (1 + (1 / (2 * self.E_so + np.finfo(np.float64).eps)) ** 2)
            < E
            < -1
        ):
            # under the gap energy range -> 4 evanescent modes
            logger.info("\tunder the gap energy range")
            logger.warning(f"\tonly tunelings modes {self.E_so}")
            self.scattering_matrix = (
                ScatteringMatrix.insulator if v and len(self.vel_a) != 0 else None
            )

            self.l = (+1, +1, -1, -1)
            self.band = (-1, -1, -1, -1)
            self.mod = 4 * (WaveVector.q,)

            self.wave_vector = (
                self.k_alpha(E, self.l[0], self.mod[0]),
                -self.k_alpha(E, self.l[1], self.mod[1]),
                self.k_alpha(E, self.l[2], self.mod[2]),
                -self.k_alpha(E, self.l[3], self.mod[3]),
            )

        elif -1 < E < -1 / (4 * self.E_so + np.finfo(np.float64).eps):
            # inside the gap energy range -> 2 propagating modes
            logger.info("\tfirst in the gap energy range")
            self.scattering_matrix = (
                ScatteringMatrix.in_gap if v and len(self.vel_a) != 0 else None
            )

            self.l = (+1, +1, +1, +1)
            self.band = (-1, -1, -1, -1)
            self.mod = 2 * (WaveVector.k,) + 2 * (WaveVector.q,)

            self.wave_vector = (
                self.k_alpha(E, self.l[0], self.mod[0]),
                -self.k_alpha(E, self.l[1], self.mod[1]),
                self.k_alpha(E, self.l[2], self.mod[2]),
                -self.k_alpha(E, self.l[3], self.mod[3]),
            )

        elif -1 / (4 * self.E_so + np.finfo(np.float64).eps) < E < 1:
            # inside the gap energy range -> 2 propagating modes
            logger.info("\tsecond in the gap energy range")
            self.scattering_matrix = (
                ScatteringMatrix.in_gap if v and len(self.vel_a) != 0 else None
            )

            self.l = (+1, +1, +1, +1)
            self.band = (-1, -1, +1, +1)
            self.mod = 2 * (WaveVector.k,) + 2 * (WaveVector.q,)

            self.wave_vector = (
                self.k_alpha(E, self.l[0], self.mod[0]),
                -self.k_alpha(E, self.l[1], self.mod[1]),
                self.k_alpha(E, self.l[2], self.mod[2]),
                -self.k_alpha(E, self.l[3], self.mod[3]),
            )

        elif 1 <= E:
            #  above the gap energy range -> 4 propagating modes
            logger.info("\tout of gap energy range")
            self.scattering_matrix = (
                ScatteringMatrix.out_gap if v and len(self.vel_a) != 0 else None
            )

            self.l = (+1, +1, -1, -1)
            self.band = (-1, -1, +1, +1)
            self.mod = 4 * (WaveVector.k,)

            self.wave_vector = (
                self.k_alpha(E, self.l[0], self.mod[0]),
                -self.k_alpha(E, self.l[1], self.mod[1]),
                self.k_alpha(E, self.l[2], self.mod[2]),
                -self.k_alpha(E, self.l[3], self.mod[3]),
            )
        else:
            logger.warning("out of range energy")
            raise ValueError("out of range energy")
        self.sgn_k = np.sign([i[1] for i in self.wave_vector])

    def prepare_zeeman_WF(self, E, v=False):
        """
        Prepare the lables for Strong Zeeman regime

        Beaware of the case in wich energy match exactly the boundary values

        Order: [incoming modes, outgoing modes]
                [propagatingmodes, evanescent modes]
        """

        if (
            -self.E_so * (1 + (1 / (2 * self.E_so + np.finfo(np.float64).eps)) ** 2)
            < E
            < -1 / (4 * self.E_so + np.finfo(np.float64).eps)
        ):
            # under the gap energy range -> 4 evanescent modes
            logger.info("\tunder the gap energy range")
            logger.warning(f"\tonly tunelings modes {self.E_so}")
            self.scattering_matrix = (
                ScatteringMatrix.insulator if v and len(self.vel_a) != 0 else None
            )

            self.l = (+1, +1, -1, -1)
            self.band = (-1, -1, -1, -1)
            self.mod = 4 * (WaveVector.q,)

            self.wave_vector = (
                self.k_alpha(E, self.l[0], self.mod[0]),
                -self.k_alpha(E, self.l[1], self.mod[1]),
                self.k_alpha(E, self.l[2], self.mod[2]),
                -self.k_alpha(E, self.l[3], self.mod[3]),
            )

        elif -1 / (4 * self.E_so + np.finfo(np.float64).eps) < E < -1:
            # under the gap energy range -> 4 evanescent modes
            logger.info("\tfirst under the gap energy range")
            self.scattering_matrix = (
                ScatteringMatrix.insulator if v and len(self.vel_a) != 0 else None
            )

            self.l = (+1, +1, -1, -1)
            self.band = (+1, +1, -1, -1)
            self.mod = 4 * (WaveVector.q,)

            self.wave_vector = (
                self.k_alpha(E, self.l[0], self.mod[0]),
                -self.k_alpha(E, self.l[1], self.mod[1]),
                self.k_alpha(E, self.l[2], self.mod[2]),
                -self.k_alpha(E, self.l[3], self.mod[3]),
            )

        elif -1 < E < 1:
            # inside the gap energy range -> 2 propagating modes
            logger.info(f"\tin the gap energy range {v}, {len(self.vel_a)}")
            self.scattering_matrix = (
                ScatteringMatrix.in_gap if v and len(self.vel_a) != 0 else None
            )

            self.l = (+1, +1, +1, +1)
            self.band = (-1, -1, +1, +1)
            self.mod = 2 * (WaveVector.k,) + 2 * (WaveVector.q,)

            self.wave_vector = (
                self.k_alpha(E, self.l[0], self.mod[0]),
                -self.k_alpha(E, self.l[1], self.mod[1]),
                self.k_alpha(E, self.l[2], self.mod[2]),
                -self.k_alpha(E, self.l[3], self.mod[3]),
            )

        elif 1 <= E:
            # above the gap energy range -> 4 propagating modes
            logger.info("\tout of gap energy range")
            self.scattering_matrix = (
                ScatteringMatrix.out_gap if v and len(self.vel_a) != 0 else None
            )

            self.l = (+1, +1, -1, -1)
            self.band = (-1, -1, +1, +1)
            self.mod = 4 * (WaveVector.k,)

            self.wave_vector = (
                self.k_alpha(E, self.l[0], self.mod[0]),
                -self.k_alpha(E, self.l[1], self.mod[1]),
                self.k_alpha(E, self.l[2], self.mod[2]),
                -self.k_alpha(E, self.l[3], self.mod[3]),
            )
        else:
            logger.warning("out of range energy")
            raise ValueError("out of range energy")
        self.sgn_k = np.sign([i[1] for i in self.wave_vector])


class RashbaJunction:
    """
    This class affer the end point to handle the SO profile and compute the Scattering matrix.
    This class should be initialised directly.
    It override some properties and methods of its parrent(WaveFunction) and implement new methods and interfaces that should be used directly in the code.
    
    """

    def __init__(
        self,
        profile=None,
        *,
        wave_function_delegate: AbstracWaveFunction = WaveFunction,
        logg=False,
        verbose=0,
    ):
        """
        RachbaJunction(profile, logg = False, verbose = 0)
            profile in an array of array [interface position, alpha profile, E_Z profile(Optional)]
                alpha profile is given in term of energy E'_so = E_so/E_Z
                the sign before SO energy correspond to the sign of SOC constant
                
                In the case E_Z profile is not present: Magnetic field is assumed to be uniform
                    In this way all the quantities are assumed to be adimensional E' = E/E_Z and x' = k_Z x

                i8n the case E_Z profile are p0resent: the magnetic field is not uniform
                    all the external parameters must be considered dimensional.
                    however they still be replace by adimensional ones internally

        """

        # check if delegate implement all the interfaces
        if issubclass(wave_function_delegate, AbstracWaveFunction):
            self.delegate = wave_function_delegate()
        else:
            raise ValueError("Delegate does not implement required interfaces")

        # set up the logging and verbose level
        if logg and verbose != 0:
            logger.disabled = False
            if verbose == 2:
                logger.setLevel(logging.DEBUG)
            elif verbose == 1:
                logger.setLevel(logging.WARNING)
        else:
            logger.disabled = True

        # list of tuples (x, alpha, E_Z)
        if profile:
            # inteface positions
            self.interface = profile[0]
            # values of the SO energy
            self.alpha_profile = profile[1]
            # initial value
            self.E_so = self.alpha_profile[0]
            # check if the magnetic field is omogeneous
            if len(profile) == 3:
                self.E_z_profile = profile[2]
                logger.warning(
                    "Use dimensional energy [meV] and length scale sqrt(2 m /hbar^2)*x [nm]"
                )
            else:
                # if the magnetic field is omogeneous:
                # set it to 1 -> does not affect other parameters
                self.E_z_profile = np.ones(len(self.alpha_profile))
            self.E_Z = self.E_z_profile[0]
        else:
            logger.warning("Default alpha profile")
            self.interface = (1, 1)
            self.alpha_profile = 0

        self.vell = []

    # Interface to set up a E_so for each region
    def __getitem__(self, ind: float) -> float:
        return self.alpha_profile[ind]

    def __setitem__(self, ind: int, item: float):
        if ind < len(self.alpha_profile):
            self.alpha_profile[ind] = item
        else:
            raise IndexError()

    def get_scattering_matrix(self, E: float, logg=False) -> ScatteringMatrix:
        """
        Higth level interface: typically is called in Notebooks to compute the scatering matrix for a given energy
        """
        M, vell = self.get_transfer_matrix(E)
        if self.delegate.scattering_matrix:
            res = self.delegate.scattering_matrix(M, vell, logg)
        else:
            raise ValueError(
                "Fail to choose the initialisation method for scattering matrix. Possibly: energy out of range"
            )
        # res = self.scattering_matrix(M, vell, logg)
        return res

    def transfer_matrix_at(self, x: int, E: float) -> NDArray:
        """
        (x:int, E: float) -> np.ndarray((4, 4), np.complex256)

        Compute the transfer matrix at x-th interface and energy E
        In contrast with scattering matrix it is not normalized by velocity
        
        Is used to compute the wavefuction with multiple interfaces
        """
        self.E_so = self.alpha_profile[x + 1]
        W_pls = self.get_boundary_matrix(self.interface[x], E, v=False)

        self.E_so = self.alpha_profile[x]
        W_min = self.get_boundary_matrix(self.interface[x], E, v=False)

        return np.matmul(linalg.inv(W_pls), W_min)

    def get_transfer_matrix(self, E: float) -> Tuple[NDArray, NDArray]:
        """
        (E: float) -> np.array((4, 4), np.complex256), np.array((2, 2) or (4, 4), np.complex256)

        Second method in call-chain.

        Compute the total transfer matrix througth all the interfaces.
        Detect leads and compute corresonding velocity

        Return the transfer matrix and the velocity normalizzation matrix. The dimension of later depend on the number of propagating chanels
        """

        self.delegate.vel_a = []
        self.delegate.vel_b = []
        M = np.eye(4, dtype=np.complex256)
        n = len(self.interface)
        # i in (0, N-1) where N is a number of interfaces
        for i in range(n):

            # rigth of the i-th interface
            E_so = self.alpha_profile[n - i]
            E_Z = self.E_z_profile[n - i]

            self.delegate.E_so = E_so
            self.delegate.E_Z = E_Z

            logger.debug(
                f" {n-i-1}: X_i+ boundary matrix\n\tE_so = {E_so}, E_Z = {E_Z}\n\tx_i = {self.interface[n-i-1]}\n\talpha sign = {np.sign(E_so)}\n"
            )

            if n - i - 1 == n - 1:
                # rigth lead detected
                # set flag to compute velocity and choose the method to calculate the scattering matrix
                v = True
            else:
                v = False
            # get rigth boundary matrix
            #   NOTE: here all the parameters are substituted by adimensioal quantity
            #   x-> k_Z x = sqrt(E_Z) x, E -> E/E_Z

            # in the case of homogeneous magnetic field E_Z = 1
            #   is equivalent to the case in which all the parameters always have been adimensional
            W_pls = self.get_boundary_matrix(
                np.sqrt(E_Z) * self.interface[n - i - 1], E / E_Z, v=v
            )

            # left of the i-th interface
            E_so = self.alpha_profile[n - i - 1]
            E_Z = self.E_z_profile[n - i - 1]

            self.delegate.E_so = E_so
            self.delegate.E_Z = E_Z
            logger.debug(
                f" {n-i-1}: X_i- boundary matrix\n\tE_so = {E_so}, E_Z = {E_Z}\n\tx_i = {self.interface[n-i-1]}\n\talpha sign = {np.sign(E_so)}\n"
            )

            if n - i - 1 == 0:
                # left lead detected
                v = True
            else:
                v = False
            # get left boundary matrix
            W_min = self.get_boundary_matrix(
                np.sqrt(E_Z) * self.interface[n - i - 1], E / E_Z, v=v
            )

            # matrix multiplication of the inverse of rigth boundary matyrix and the left one
            M = np.matmul(M, np.matmul(linalg.inv(W_pls), W_min))

        # all the velocities have been computed within get_boundary_matrix()
        # [left lead velocity, rigth lead velocity]
        vel_a = np.array(self.delegate.vel_a)
        vel_b = np.array(self.delegate.vel_b)
        if len(vel_a) == 2:
            vel_a = np.abs(np.array(vel_a[::-1]))
            vel_b = np.abs(np.array(vel_b[::-1]))

        elif len(vel_a) == 4:
            vel_a[0:2], vel_a[2:4] = (
                np.abs(vel_a[2:4]),
                np.abs(np.array(vel_a[0:2])),
            )
            vel_b[0:2], vel_b[2:4] = (
                np.abs(vel_b[2:4]),
                np.abs(np.array(vel_b[0:2])),
            )

        # collumn vector
        v_b = np.lib.scimath.sqrt(vel_b.reshape((len(vel_b), 1)))
        # row vector
        v_a = 1 / np.lib.scimath.sqrt(vel_a.reshape((1, len(vel_a))))
        vel_factor_mat = np.dot(v_b, v_a)

        return M, vel_factor_mat

    def get_boundary_matrix(self, x: float, E: float, v=False) -> NDArray:
        """
        (x:float, E:float, v=False) -> np.ndarray((4, 4), np.complex256)

        Therd method in call-chain
            Energy and length are dimensionless E/E_Z, k_Z*x
            (with E_Z appropriate for each region)
        """
        # change the object property
        #   prepare lables of the wave function
        self.delegate.prepare_WF(E, v=v)

        if v:
            # change the object property
            #   compute velocity in-place
            self.delegate.calculate_velocity(E)
        # get upper part of the boundary matrix
        #   the continuity of wave function
        w_func = self.delegate.compile_wave_function(x)
        # get lower part of the boundary matrix
        #   the continuity of flux
        flux = self.delegate.flux(E, w_func)
        return np.transpose(np.append(w_func, flux, axis=1))

    def get_WF(self, x: float, E: float) -> NDArray:
        """
        (x:float, E:float) -> np.ndarray((2, 4), np.complex256)

        Compute all the eigenstates at x for energy E

        use dimensionless energy and length (E/E_Z) only
        """
        return self.get_boundary_matrix(x, E)[:2, :]

