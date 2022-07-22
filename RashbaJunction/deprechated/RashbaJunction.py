import numpy as np
import scipy.constants as cc

from RachbaJunction.utilities import set_zeros
import RachbaJunction.utilities as ut

logger = ut.get_loger(__name__)

hbar = 6.582119569e-16

class RachbaJunction:
    _m = 0.022*cc.m_e
    def __init__(self, alpha_profile, h, logg = False):

        if logg:
            logger.disabled = False
        else:
            logger.disabled = True

        
        # h ia an array with [h_xy, h_z, phi_xy]
        self.h_xy = h[0]
        self.h_z = h[1]
        self.phi = h[2]
        
        #list of tuples (x, alpha)
        self.alpha_profile = alpha_profile
        
        # self._alpha = 0.0
        self.current_alpha = alpha_profile[0][1]

        self.vel_a = []
        self.vel_b = []
        self.vell = []
        
    @property    
    def current_alpha(self):
        return self._alpha
    
    @current_alpha.setter
    def current_alpha(self, alpha):
        self.E_so = self._m*alpha**2/(2*hbar**2)
        self._alpha = alpha
        
    def k_E(self, e, l):
        return np.sqrt(2*self._m)/hbar*np.sqrt(e + 2*self.E_so + l*np.sqrt(4*e*self.E_so+4*self.E_so**2 + self.h_xy**2))
    def q_E(self, e, l):
        return np.sqrt(2*self._m)/hbar*np.sqrt(-(e + 2*self.E_so) + l*np.sqrt(4*e*self.E_so+4*self.E_so**2 + self.h_xy**2))
    def E_0(self, k):
        return hbar**2*k**2/(2*self._m)

    def omega_k(self, x, k, b):
        dnom = np.sqrt((self.current_alpha*k + self.h_z)**2 + self.h_xy**2)
        
        res = np.array([np.exp(complex(0,-self.phi))*(self.current_alpha*k - b*dnom)/self.h_xy, 1], dtype=np.complex128)
        res = 1/np.linalg.norm(res) * res*np.exp(complex(0, k*x))
        return res

    def omega_q(self, x, q, b):
        dnom = np.sqrt(-(self.current_alpha*q)**2 + self.h_xy**2)
        
        res = np.array([np.exp(complex(0,-self.phi))*(-complex(0,self.current_alpha*q) - b*dnom)/self.h_xy, 1], dtype=np.complex128)
        res = 1/np.linalg.norm(res) * res*np.exp(complex(0, q*x))
        return res
        
    def flux(self, k, mod, w_func):
        res = []
        for i in range(w_func.shape[0]):
            if mod[i] == "k":
                vel_op = hbar* k[i]/self._m*np.eye(2) - self.current_alpha/hbar*np.array([[1,0], [0,-1]])

            elif mod[i] == "q":
                vel_op = complex(0, -hbar* k[i]/self._m)*np.eye(2) - self.current_alpha/hbar* np.array([[1,0 ], [0,-1]])

            vel_op = np.sqrt(self._m/2)*vel_op
        
            res.append(np.dot(vel_op, w_func[i]))
        return np.array(res, dtype=np.complex128)
        
    def compile_wave_function(self, x, k, mod, band, v = False):
        res = []
        if v == True:
            self.calculate_velocity(k, mod, band)
        for k, m, b in zip(k, mod, band):
            if m == "k":
                # vell = self.group_velocty_k(k, b)
                # res.append(1/np.lib.scimath.sqrt(2*np.pi*hbar*vell)*self.omega_k(x, k, b))

                res.append(self.omega_k(x, k, b))
            elif m == "q":
                # vell = self.group_velocty_q(k, b)
                # res.append(1/np.lib.scimath.sqrt(2*np.pi*hbar*vell)*self.omega_q(x, k, b))

                res.append(self.omega_q(x, k, b))
        return np.array(res, dtype=np.complex128)
    
    def group_velocty_k(self, k, b):
        return hbar*k/self._m + b*self.current_alpha*(self.current_alpha*k + self.h_z)/(hbar*np.sqrt(self.h_xy**2 + (self.current_alpha*k+self.h_z)**2))

    def group_velocty_q(self, k, b):
        return self.group_velocty_k(complex(0, -k), b)
        # return -hbar*k/self._m - b*self.current_alpha**2*k/(hbar*np.sqrt(self.h_xy**2 - (self.current_alpha*k)**2))


    def calculate_velocity(self, k, mod, band):
        #[rigth lead velocity, left lead velocity]
        # if len(self.vel_a) == 0 or len(self.vel_b) == 0:
        if len(self.vel_a)%2 == 0 or len(self.vel_b)%2 == 0:
            # In rigth lead: injected coefficient(a) is associated with negative k
            #               reflected coefficient(b) is associated with positive k
            logger.debug("\t\trigth lead")
            k_s = -1
        else:
            # In left lead: injected coefficient(a) is associated with positve k
            #               reflected coefficient(b) is associated with negative k
            logger.debug("\t\tleft lead")
            k_s = 1
            
        for k, m, b in zip(k, mod, band):
            if k_s*k > 0 and m == "k":
                logger.debug(f"\t\t-->a - velcity {self.group_velocty_k(k, b)}")

                self.vel_a.append(self.group_velocty_k(k, b))
            elif k_s*k < 0 and m =="k":
                logger.debug(f"\t\t-->b - velcity {self.group_velocty_k(k, b)}")

                self.vel_b.append(self.group_velocty_k(k, b))

    def get_transfer_matrix(self, E):
        #depend on alpha
        self.vel_a = []
        self.vel_b = []
        M = np.eye(4, dtype=np.complex128)
        for i in range(len(self.alpha_profile)-1, 0, -1):
            #x_0 is the first transition
            #x_{N+1} is an infinite --> matter only lambda_{N+1}
            if i == len(self.alpha_profile)-1:
                #rigth lead velocity
                v = True
            else:
                v = False
            self.current_alpha = self.alpha_profile[i][1]
            logger.debug(f" {i-1}: X_i+ boundary matrix\n\talpha = {self.current_alpha}\n\tx_i = {self.alpha_profile[i-1][0]}\n\tE_so = {self.E_so}\n\th_xy = {self.h_xy}")
            W_pls = self.get_boundary_matrix(self.alpha_profile[i-1][0], E, v = v)
            
            if i == 1:
                #rigth lead velocity
                v = True
            else:
                v = False
            self.current_alpha = self.alpha_profile[i-1][1]
            logger.debug(f" {i-1}: X_i- boundary matrix\n\talpha = {self.current_alpha}\n\tx_i = {self.alpha_profile[i-1][0]}\n\tE_so = {self.E_so}\n\th_xy = {self.h_xy}")
            W_min = self.get_boundary_matrix(self.alpha_profile[i-1][0], E, v = v)

            W_min = set_zeros(W_min)
            W_pls = set_zeros(W_pls)

            M = np.matmul(M, np.matmul(np.linalg.inv(W_pls), W_min))
            # M = np.matmul(np.linalg.inv(W_pls), W_min)


        #[left lead velocity, rigth lead velocity]
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
        v_a = 1/np.lib.scimath.sqrt(self.vel_a.reshape((1, len(self.vel_a))))
        vel_factor_mat = np.dot(v_b, v_a)
        return M, vel_factor_mat
    
    def get_boundary_matrix(self, x, E, v = False):
        if self.h_xy < 2*self.E_so:
            logger.info("-->Rashba regime")
            res = self.rashba(x, E, v = v)
        elif 2*self.E_so < self.h_xy < 4*self.E_so:
            print("-->Week Zeeman regime")
            res = self.week_zeeman(x, E)
        elif 4*self.E_so < self.h_xy:
            print("-->Zeeman regime")
            res = self.zeeman(x, E)
        return res
    

    def rashba(self, x, E, v = False):
        
        if -self.E_so*(1+(self.h_xy/(2*self.E_so))**2) < E < -self.h_xy:
            logger.info("\tunder the gap energy range")

            k = [self.k_E(E, +1), -self.k_E(E, +1), self.k_E(E, -1), -self.k_E(E, -1)]
            b = [-1, -1, -1, -1]
            mod = 4*["k"]
           
        elif -self.h_xy < E < -self.h_xy**2/(4*self.E_so):
            logger.info("\tfirst in the gap energy range")

            k = [self.k_E(E, +1), -self.k_E(E, +1), self.q_E(E, +1), -self.q_E(E, +1)]
            b = [-1, -1, -1, -1]
            mod = 2*["k"] + 2*["q"]

        elif -self.h_xy**2/(4*self.E_so) < E < self.h_xy:
            logger.info("\tsecond in the gap energy range")

            k = [self.k_E(E, +1), -self.k_E(E, +1), self.q_E(E, +1), -self.q_E(E, +1)]
            b = [-1, -1, +1, +1]
            mod = 2*["k"] + 2*["q"]
            
        elif self.h_xy < E:
            logger.info("\tout of gap energy range")
            
            k = [self.k_E(E, +1), -self.k_E(E, +1), self.k_E(E, -1), -self.k_E(E, -1)]
            b = [-1, -1, +1, +1]
            mod = 4*["k"]

        w_func = self.compile_wave_function(x, k, mod, b, v = v)
        flux = self.flux(k, mod, w_func)   
        return np.transpose(np.append(w_func, flux, axis = 1))
        
    def week_zeeman(self, x, E, v = False):
        
        if -self.E_so*(1+(self.h_xy/(2*self.E_so))**2) < E < -self.h_xy:
            # w_func = np.array([self.omega_q(x, self.q_E(E, +1), -1), self.omega_q(x, -self.q_E(E, +1), -1),
            #                    self.omega_q(x, self.q_E(E, -1), -1), self.omega_q(x, -self.q_E(E, -1), -1)])
            
            b = [-1, -1, -1, -1]
            k = [self.q_E(E, +1), -self.q_E(E, +1), self.q_E(E, -1), -self.q_E(E, -1)]
            mod = 4*["q"]
            
        elif -self.h_xy < E < -self.h_xy**2/(4*self.E_so):
            # w_func = np.array([self.omega_k(x, self.k_E(E, +1), -1), self.omega_k(x, -self.k_E(E, +1), -1),
            #                    self.omega_q(x, self.q_E(E, +1), -1), self.omega_q(x, -self.q_E(E, +1), -1)])
            
            b = [-1, -1, -1, -1]
            k = [self.k_E(E, +1), -self.k_E(E, +1), self.q_E(E, +1), -self.q_E(E, +1)]
            mod = 2*["k"] + 2*["q"]
            
        elif -self.h_xy**2/(4*self.E_so) < E < self.h_xy:
            # w_func = np.array([self.omega_k(x, self.k_E(E, +1), -1), self.omega_k(x, -self.k_E(E, +1), -1),
            #                    self.omega_q(x, self.q_E(E, +1), +1), self.omega_q(x, -self.q_E(E, +1), +1)])
            
            b = [-1, -1, +1, +1]
            k = [self.k_E(E, +1), -self.k_E(E, +1), self.q_E(E, +1), -self.q_E(E, +1)]
            mod = 2*["k"] + 2*["q"]
            
        elif self.h_xy < E:
            # w_func = np.array([self.omega_k(x, self.k_E(E, +1), -1), self.omega_k(x, -self.k_E(E, +1), -1),
            #                    self.omega_k(x, self.k_E(E, -1), +1), self.omega_k(x, -self.k_E(E, -1), +1)])
            
            b = [-1, -1, +1, +1]
            k = [self.k_E(E, +1), -self.k_E(E, +1), self.k_E(E, -1), -self.k_E(E, -1)]
            mod = 4*["k"]
            
        w_func = self.compile_wave_function(x, k, mod, b, v = v)
        flux = self.flux(k, mod, w_func) 
        return np.transpose(np.append(w_func, flux, axis = 1))
            
    def zeeman(self, x, E, v = False):
        
        if -self.E_so*(1+(self.h_xy/(2*self.E_so))**2) < E < -self.h_xy**2/(4*self.E_so):
            # w_func = np.array([self.omega_q(x, self.q_E(E, +1), -1), self.omega_q(x, -self.q_E(E, +1), -1),
            #                    self.omega_q(x, self.q_E(E, -1), -1), self.omega_q(x, -self.q_E(E, -1), -1)])
            
            b = [-1, -1, -1, -1]
            k = [self.q_E(E, +1), -self.q_E(E, +1), self.q_E(E, -1), -self.q_E(E, -1)]
            mod = 4*["q"]
 
        elif -self.h_xy**2/(4*self.E_so) < E < -self.h_xy:
            # w_func = np.array([self.omega_q(x, self.q_E(E, +1), +1), self.omega_q(x, -self.q_E(E, +1), +1),
            #                    self.omega_q(x, self.q_E(E, -1), -1), self.omega_q(x, -self.q_E(E, -1), -1)])
            
            b = [+1, +1, -1, -1]
            k = [self.q_E(E, +1), -self.q_E(E, +1), self.q_E(E, -1), -self.q_E(E, -1)]
            mod = 4*["q"]

        elif -self.h_xy**2/(4*self.E_so) < E < self.h_xy:
            # w_func = np.array([self.omega_k(x, self.k_E(E, +1), -1), self.omega_k(x, -self.k_E(E, +1), -1),
            #                    self.omega_q(x, self.q_E(E, +1), +1), self.omega_q(x, -self.q_E(E, +1), +1)])
            
            b = [-1, -1, +1, +1]
            k = [self.k_E(E, +1), -self.k_E(E, +1), self.q_E(E, +1), -self.q_E(E, +1)]
            mod = 2*["k"] + 2*["q"]

        elif self.h_xy < E:
            # w_func = np.array([self.omega_k(x, self.k_E(E, +1), -1), self.omega_k(x, -self.k_E(E, +1), -1),
            #                    self.omega_k(x, self.k_E(E, -1), +1), self.omega_k(x, -self.k_E(E, -1), +1)])
            
            b = [-1, -1, +1, +1]
            k = [self.k_E(E, +1), -self.k_E(E, +1), self.k_E(E, -1), -self.k_E(E, -1)]
            mod = 4*["k"]
        
        w_func = self.compile_wave_function(x, k, mod, b, v = v)
        flux = self.flux(k, mod, w_func) 
        return np.transpose(np.append(w_func, flux, axis = 1))