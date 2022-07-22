import numpy as np
import scipy.constants as cc
# from numba import jit

from scipy import linalg

import logging

from RashbaJunction.utilities import set_zeros
import RashbaJunction.utilities as ut
# from RashbaJunction.ScatteringMatrix.ScatteringMatrix import ScatteringMatrix
from RashbaJunction.ScatteringMatrix import ScatteringMatrix



logger = ut.get_loger(__name__)

hbar = 6.582119569e-16

class WaveFunction():
    # may be seen as realizzation of some interface tha is used in the parent class
    
    def __init__(self):

        # may be only inherited from parent class

        self.vel_a = []
        self.vel_b = []

        self.E_so = 0.0
        self.sgn_alpha = -1
        
        # would be set in the parent class
        self.mod = []
        self.band = []
        self.l = []
        self.wave_length = []

    @property
    def sgn_k(self):
        return np.sign(self.wave_length)
    
    def k_alpha(self, E, l, m):
        return np.array([2*self.sgn_alpha*np.sqrt(self.E_so)* np.sqrt(self.E_0(E, l, m)), np.sqrt(self.E_0(E, l, m))])

    def E_0(self, E, l, m):
        res = 0.0
        if m == "k":
            res = E + 2*self.E_so + l*np.sqrt(4*E*self.E_so+4*self.E_so**2 + 1)
        elif m == "q":
            res = -(E + 2*self.E_so) + l*np.sqrt(4*E*self.E_so+4*self.E_so**2 + 1)
        else:
            logger.warning("Wrong wave function mode lable (in E_0)")

        return res

    def omega_k(self, x, k, b):  
        norm = np.sqrt(0.5/(k[0]**2 +1 - b *k[0]*np.sqrt(1 + k[0]**2)))
        res = norm*np.array([(k[0] - b*np.sqrt(k[0]**2 + 1)), 1], dtype=np.complex256)
        return res*np.exp(complex(0, k[1]*x))

    def omega_q(self, x, q, b):
        norm = np.sqrt(1/2)
        res = norm*np.array([(-complex(0,q[0]) - b*np.sqrt(1 - q[0]**2)), 1], dtype=np.complex256)
        return res*np.exp(q[1]*x)
        
    def flux(self, E, w_func):
        res = []
        for i in range(w_func.shape[0]):
            if self.mod[i] == "k":

                vel_op = (np.sign(self.wave_length[i][1])*np.sqrt(self.E_0(E, self.l[i], self.mod[i])) * np.eye(2) - self.sgn_alpha * np.sqrt(self.E_so) * np.array([[1,0], [0,-1]]))
            elif self.mod[i] == "q":

                vel_op = -(complex(0, np.sign(self.wave_length[i][1])) *np.sqrt(self.E_0(E, self.l[i], self.mod[i])) * np.eye(2) + self.sgn_alpha * np.sqrt(self.E_so) * np.array([[1,0], [0,-1]]))
            else:
                logger.warning("Wrong wave function mode lable (in flux)")

            res.append(np.dot(vel_op, w_func[i]))
        return np.array(res, dtype=np.complex256)

    def compile_wave_function(self, x, E, v = False):
        res = []
        if v == True:
            # change the object property
            self.calculate_velocity(E)
        for k, m, b in zip(self.wave_length, self.mod, self.band):
            if m == "k":
                res.append(self.omega_k(x, k, b))
            elif m == "q":
                res.append(self.omega_q(x, k, b))
        return np.array(res, dtype=np.complex256)

    def calculate_velocity(self, E):
        #[rigth lead velocity, left lead velocity]
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
            k = self.wave_length[i][0]

            if k_s*np.sign(self.wave_length[i][1]) > 0 and self.mod[i] == "k":
                vel = np.sign(self.wave_length[i][1])*(np.sqrt(self.E_0(E, self.l[i], self.mod[i])*(1+k**2)) + self.band[i]*np.abs(k)*np.sqrt(self.E_so))/np.sqrt(1+k**2)

                logger.debug(f"\t\t{i}: lambda {self.l[i]}, band {self.band[i]}, sign k: {np.sign(self.wave_length[i][1])}")
                logger.debug(f"\t\t-->a - velcity {vel}")

                self.vel_a.append(vel)
            elif k_s*np.sign(self.wave_length[i][1]) < 0 and self.mod[i] =="k":
                vel = np.sign(self.wave_length[i][1])*(np.sqrt(self.E_0(E, self.l[i], self.mod[i])*(1+k**2)) + self.band[i]*np.abs(k)*np.sqrt(self.E_so))/np.sqrt(1+k**2)

                logger.debug(f"\t\t{i}: lambda {self.l[i]}, band {self.band[i]}, sign k: {np.sign(self.wave_length[i][1])}")
                logger.debug(f"\t\t-->b - velcity {vel}")

                self.vel_b.append(vel)



class RashbaJunction(WaveFunction):
    _m = 0.022*cc.m_e
    def __init__(self, profile = [[],[]], logg = False):
        """
        RashbaJunction(self, profile, h, logg = False)
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
        super(RashbaJunction, self).__init__()

        
        #list of tuples (x, alpha)
        self.interface = profile[0]
        self.alpha_profile = profile[1]
        
        # self._alpha = 0.0
        if len(profile[1]) != 0:
            self.E_so = profile[1][0]
        
        self.vell = []

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
    @sgn_alpha.setter
    def sgn_alpha(self, a):
        self._alpha = a

    def __getitem__(self, ind):
        return self.alpha_profile[ind]

    def __setitem__(self,ind, item):
        if ind < len(self.alpha_profile):
            self.alpha_profile[ind] = item
        else:
            raise IndexError()

    def transfer_matrix_at(self, x, E):
        # n = len(self.interface)
        self.E_so = self.alpha_profile[x+1]
        W_pls = self.get_boundary_matrix(self.interface[x], E, v = False)

        self.E_so = self.alpha_profile[x]
        W_min = self.get_boundary_matrix(self.interface[x], E, v = False)

        return np.matmul(linalg.inv(W_pls), W_min)

    def get_transfer_matrix(self, E):
        #depend on alpha
        self.vel_a = []
        self.vel_b = []
        M = np.eye(4, dtype=np.complex256)
        n = len(self.interface)
        for i in range(len(self.interface)):
            #x_0 is the first transition
            #x_{N+1} is an infinite --> matter only lambda_{N+1}
            if n-i-1 == len(self.interface)-1:
                #rigth lead velocity
                v = True
            else:
                v = False
            
            self.E_so = self.alpha_profile[n-i]
            logger.debug(f" {n-i-1}: X_i+ boundary matrix\n\tE_so = {self.E_so}\n\tx_i = {self.interface[n-i-1]}\n\talpha sign = {self.sgn_alpha}\n")
            W_pls = self.get_boundary_matrix(self.interface[n-i-1], E, v = v)
            
            if n-i-1 == 0:
                #rigth lead velocity
                v = True
            else:
                v = False
            self.E_so = self.alpha_profile[n-i-1]
            logger.debug(f" {n-i-1}: X_i- boundary matrix\n\tE_so = {self.E_so}\n\tx_i = {self.interface[n-i-1]}\n\talpha sign = {self.sgn_alpha}\n")
            W_min = self.get_boundary_matrix(self.interface[n-i-1], E, v = v)

            M = np.matmul(M, np.matmul(linalg.inv(W_pls), W_min))


        #[left lead velocity, rigth lead velocity]
        self.vel_a = np.array(self.vel_a)
        self.vel_b = np.array(self.vel_b)
        if len(self.vel_a) == 2:
            self.vel_a = np.abs(np.array(self.vel_a[::-1]))
            self.vel_b = np.abs(np.array(self.vel_b[::-1]))

        elif len(self.vel_a) == 4:
            tmp = np.abs(np.array(self.vel_a[0:2]))
            self.vel_a[0:2] = np.abs(self.vel_a[2:4])
            self.vel_a[2:4] = tmp

            tmp = np.abs(np.array(self.vel_b[0:2]))
            self.vel_b[0:2] = np.abs(self.vel_b[2:4])
            self.vel_b[2:4] = tmp


        # collumn vector
        v_b = np.lib.scimath.sqrt(self.vel_b.reshape((len(self.vel_b), 1)))
        # row vector
        v_a = 1/np.lib.scimath.sqrt(self.vel_a.reshape((1, len(self.vel_a))))
        vel_factor_mat = np.dot(v_b, v_a)

        return M, vel_factor_mat
    
    
    def get_scattering_matrix(self, E, logg = False):
        M, vell = self.get_transfer_matrix(E)
        res = self.scattering_matrix(M, vell, logg)
        return res

    def get_boundary_matrix(self, x, E, v = False):
        if 1/2 < self.E_so:
            logger.info("-->Rashba regime")
            self.prepare_rashba_WF(x, E, v = v)
        elif 1/4 < self.E_so and self.E_so < 1/2:
            logger.info("-->Week zeeman regime")
            self.prepare_week_zeeman_WF(x, E, v = v)
        elif self.E_so < 1/4:
            logger.info("-->Zeeman regime")
            self.prepare_zeeman_WF(x, E, v = v)

        w_func = self.compile_wave_function(x, E, v = v)
        flux = self.flux(E, w_func)   
        return np.transpose(np.append(w_func, flux, axis = 1))

    def get_WF(self, x, E):
        return self.get_boundary_matrix(x, E)[:2,:]

    def prepare_rashba_WF(self, x, E, v = False):
        
        if -self.E_so*(1+(1/(2*self.E_so+ np.finfo(np.float64).eps))**2) < E < -1:
            logger.info("\tunder the gap energy range")
            self.scattering_matrix = ScatteringMatrix.above_gap if v and len(self.vel_a) != 0 else None

            self.l = [+1, +1, -1, -1]
            self.mod = 4*["k"]
                # check velocity components and order
            self.wave_length = [self.k_alpha(E, self.l[0], self.mod[0]), 
                                -self.k_alpha(E, self.l[1], self.mod[1]), 
                                -self.k_alpha(E, self.l[2], self.mod[2]), 
                                self.k_alpha(E, self.l[3], self.mod[3])]
            self.band = [-1, -1, -1, -1]

           
        elif -1 < E < -1/(4*self.E_so+ np.finfo(np.float64).eps):
            logger.info("\tfirst in the gap energy range")
            self.scattering_matrix = ScatteringMatrix.in_gap if v and len(self.vel_a) != 0 else None

            self.l = [+1, +1, +1, +1]
            self.band = [-1, -1, -1, -1]
            self.mod = 2*["k"] + 2*["q"]

            self.wave_length = [self.k_alpha(E, self.l[0], self.mod[0]), 
                                -self.k_alpha(E, self.l[1], self.mod[1]), 
                                self.k_alpha(E, self.l[2], self.mod[2]), 
                                -self.k_alpha(E, self.l[3], self.mod[3])]

        elif -1/(4*self.E_so+ np.finfo(np.float64).eps) < E < 1:
            logger.info("\tsecond in the gap energy range")
            self.scattering_matrix = ScatteringMatrix.in_gap if v and len(self.vel_a) != 0 else None

            self.l = [+1, +1, +1, +1]
            self.band = [-1, -1, +1, +1]
            self.mod = 2*["k"] + 2*["q"]

            self.wave_length = [self.k_alpha(E, self.l[0], self.mod[0]), 
                                -self.k_alpha(E, self.l[1], self.mod[1]), 
                                self.k_alpha(E, self.l[2], self.mod[2]), 
                                -self.k_alpha(E, self.l[3], self.mod[3])]
            
        elif 1 <= E:
            logger.info("\tout of gap energy range")
            self.scattering_matrix = ScatteringMatrix.above_gap if v and len(self.vel_a) != 0 else None
            
            self.l = [+1, +1, -1, -1]
            self.band = [-1, -1, +1, +1]
            self.mod = 4*["k"]

            self.wave_length = [self.k_alpha(E, self.l[0], self.mod[0]), 
                                -self.k_alpha(E, self.l[1], self.mod[1]), 
                                self.k_alpha(E, self.l[2], self.mod[2]), 
                                -self.k_alpha(E, self.l[3], self.mod[3])]
        else:
            logger.warning("out of range energy")
        
    def prepare_week_zeeman_WF(self, x, E, v = False):
        
        if -self.E_so*(1+(1/(2*self.E_so + np.finfo(np.float64).eps))**2) < E < -1:
            logger.info("\tunder the gap energy range")
            logger.warning(f"\tonly tunelings modes {self.E_so}")
            self.scattering_matrix = ScatteringMatrix.insulator if v and len(self.vel_a) != 0 else None

            self.l = [+1, +1, -1, -1]
            self.band = [-1, -1, -1, -1]
            self.mod = 4*["q"]

            self.wave_length = [self.k_alpha(E, self.l[0], self.mod[0]), 
                                -self.k_alpha(E, self.l[1], self.mod[1]), 
                                self.k_alpha(E, self.l[2], self.mod[2]), 
                                -self.k_alpha(E, self.l[3], self.mod[3])]
            
        elif -1 < E < -1/(4*self.E_so+  np.finfo(np.float64).eps):
            logger.info("\tfirst in the gap energy range")
            self.scattering_matrix = ScatteringMatrix.in_gap if v and len(self.vel_a) != 0 else None

            self.l = [+1, +1, +1, +1]
            self.band = [-1, -1, -1, -1]
            self.mod = 2*["k"] + 2*["q"]

            self.wave_length = [self.k_alpha(E, self.l[0], self.mod[0]), 
                                -self.k_alpha(E, self.l[1], self.mod[1]), 
                                self.k_alpha(E, self.l[2], self.mod[2]), 
                                -self.k_alpha(E, self.l[3], self.mod[3])]
            
        elif -1/(4*self.E_so + np.finfo(np.float64).eps) < E < 1:
            logger.info("\tsecond in the gap energy range")
            self.scattering_matrix = ScatteringMatrix.in_gap if v and len(self.vel_a) != 0 else None

            self.l = [+1, +1, +1, +1]
            self.band = [-1, -1, +1, +1]
            self.mod = 2*["k"] + 2*["q"]

            self.wave_length = [self.k_alpha(E, self.l[0], self.mod[0]), 
                                -self.k_alpha(E, self.l[1], self.mod[1]), 
                                self.k_alpha(E, self.l[2], self.mod[2]), 
                                -self.k_alpha(E, self.l[3], self.mod[3])]
            
        elif 1 <= E:
            logger.info("\tout of gap energy range")
            self.scattering_matrix = ScatteringMatrix.above_gap if v and len(self.vel_a) != 0 else None

            self.l = [+1, +1, -1, -1]
            self.band = [-1, -1, +1, +1]
            self.mod = 4*["k"]

            self.wave_length = [self.k_alpha(E, self.l[0], self.mod[0]), 
                                -self.k_alpha(E, self.l[1], self.mod[1]), 
                                self.k_alpha(E, self.l[2], self.mod[2]), 
                                -self.k_alpha(E, self.l[3], self.mod[3])]
            
    def prepare_zeeman_WF(self, x, E, v = False):
        if -self.E_so*(1+(1/(2*self.E_so + np.finfo(np.float64).eps))**2) < E < -1/(4*self.E_so + np.finfo(np.float64).eps):
            logger.info("\tunder the gap energy range")
            logger.warning(f"\tonly tunelings modes {self.E_so}")
            self.scattering_matrix = ScatteringMatrix.insulator if v and len(self.vel_a) != 0 else None
            
            self.l = [+1, +1, -1, -1]
            self.band = [-1, -1, -1, -1]
            self.mod = 4*["q"]

            self.wave_length = [self.k_alpha(E, self.l[0], self.mod[0]), 
                                -self.k_alpha(E, self.l[1], self.mod[1]), 
                                self.k_alpha(E, self.l[2], self.mod[2]), 
                                -self.k_alpha(E, self.l[3], self.mod[3])]
 
        elif -1/(4*self.E_so + np.finfo(np.float64).eps) < E < -1:
            logger.info("\tfirst in the gap energy range")
            # self.scattering_matrix = ScatteringMatrix.in_gap if v and len(self.vel_a) != 0 else None
            self.scattering_matrix = ScatteringMatrix.insulator if v and len(self.vel_a) != 0 else None
            
            self.l = [+1, +1, -1, -1]
            self.band = [+1, +1, -1, -1]
            self.mod = 4*["q"]

            self.wave_length = [self.k_alpha(E, self.l[0], self.mod[0]), 
                                -self.k_alpha(E, self.l[1], self.mod[1]), 
                                self.k_alpha(E, self.l[2], self.mod[2]), 
                                -self.k_alpha(E, self.l[3], self.mod[3])]

        elif -1 < E < 1:
            logger.info(f"\tin the gap energy range {v}, {len(self.vel_a)}")
            self.scattering_matrix = ScatteringMatrix.in_gap if v and len(self.vel_a) != 0 else None
            
            self.l = [+1, +1, +1, +1]
            self.band = [-1, -1, +1, +1]
            self.mod = 2*["k"] + 2*["q"]

            self.wave_length = [self.k_alpha(E, self.l[0], self.mod[0]), 
                                -self.k_alpha(E, self.l[1], self.mod[1]), 
                                self.k_alpha(E, self.l[2], self.mod[2]), 
                                -self.k_alpha(E, self.l[3], self.mod[3])]

        elif 1 <= E:
            logger.info("\tout of gap energy range")
            self.scattering_matrix = ScatteringMatrix.above_gap if v and len(self.vel_a) != 0 else None
            
            self.l = [+1, +1, -1, -1]
            self.band = [-1, -1, +1, +1]
            self.mod = 4*["k"]

            self.wave_length = [self.k_alpha(E, self.l[0], self.mod[0]), 
                                -self.k_alpha(E, self.l[1], self.mod[1]), 
                                self.k_alpha(E, self.l[2], self.mod[2]), 
                                -self.k_alpha(E, self.l[3], self.mod[3])]
