import numpy as np
import RashbaJunction.utilities as ut
from tabulate import tabulate

# import logging

# logger = ut.get_loger(__name__)
# logger.setLevel(logging.WARNING)


class ScatteringMatrix():
    unity_tol = 1e-9
    def __init__(self, s, c, logg = False):
        # if logg:
        #     logger.disabled = False
        # else:
        #     logger.disabled = True
        
        self.C = c
        self.S = s
        if not np.allclose(np.matmul(s.T.conj(), s), np.eye(s.shape[0]), atol=self.unity_tol):
            # logger.warning("Scattering matrix isn't unitary")
            print("Scattering matrix isn't unitary")



    @property
    def is_unitary(self):
        if np.allclose(np.matmul(self.S.T.conj(), self.S), np.eye(self.S.shape[0]), atol=self.unity_tol):
            return True
        else:
            return False
    
    @property
    def t_coef(self):
        indd = int(self.S.shape[0]/2)
        t  = self.S[0:indd, indd:]
        # is calculating squares norm
        return np.linalg.norm(np.trace(np.matmul(t.T.conj(), t)))#/indd

    @property
    def r_coef(self):
        indd = int(self.S.shape[0]/2)
        r  = self.S[0:indd, 0:indd]
        # is calculating squares norm
        return np.linalg.norm(np.trace(np.matmul(r.T.conj(), r)))#/indd

    @classmethod
    def in_gap(cls, M_tot, vellocity, logg = False):
        sig_11 = (-M_tot[1,0]*M_tot[2,2] + M_tot[1,2]*M_tot[2,0])/(M_tot[2,2]*M_tot[1,1]- M_tot[1,2]*M_tot[2,1])

        sig_12 = M_tot[2,2]/(M_tot[2,2]*M_tot[1,1]- M_tot[1,2]*M_tot[2,1])

        sig_21 = M_tot[0,0] + M_tot[0,1]* sig_11 - M_tot[0,2]*(M_tot[2,0]+M_tot[2,1]*sig_11)/M_tot[2,2]

        sig_22 = (M_tot[0,1]- M_tot[0,2]*M_tot[2,1]/M_tot[2,2])*sig_12


        sig_31 = -(M_tot[2,0] + M_tot[2,1]*sig_11)/M_tot[2,2]

        sig_32 = -M_tot[2,1]/M_tot[2,2] * sig_12

        sig_41 = M_tot[3,0] + M_tot[3,1]* sig_11 + M_tot[3,2]*sig_31

        sig_42 = M_tot[3,1] * sig_12 + M_tot[3,2] * sig_32

        c = np.array([[sig_11, sig_12], [sig_21, sig_22], 
                        [sig_31, sig_32], [sig_41, sig_42]])

        s = np.array([[sig_11, sig_12], [sig_21, sig_22]])*vellocity

        return cls(s, c, logg)

    @classmethod
    def above_gap(cls, M_tot, vellocity, logg = False):
        ft = M_tot[1,1]*M_tot[3,3]/(M_tot[1,1]*M_tot[3,3] - M_tot[1,3]*M_tot[3,1])

        sig_11 = ft*(M_tot[1,3]*M_tot[3,0] - M_tot[1,0]*M_tot[3,3])/(M_tot[1,1]*M_tot[3,3])

        sig_12 = ft*(M_tot[1,3]*M_tot[3,2] - M_tot[1,2]*M_tot[3,3])/(M_tot[1,1]*M_tot[3,3])

        sig_13 = ft/M_tot[1,1]

        sig_14 = -ft*M_tot[1,3]/(M_tot[1,1]*M_tot[3,3])


        sig_21 = -(M_tot[3,0] + M_tot[3,1]*sig_11)/M_tot[3,3]

        sig_22 = -(M_tot[3,1]*sig_12 + M_tot[3,2])/M_tot[3,3]

        sig_23 = -M_tot[3,1]*sig_13/M_tot[3,3]

        sig_24 = (1 - M_tot[3,1]*sig_14)/M_tot[3,3]


        sig_31 = M_tot[0,0] + M_tot[0,1]*sig_11 + M_tot[0,3]*sig_21 

        sig_32 = M_tot[0,2] + M_tot[0,1]*sig_12 + M_tot[0,3]*sig_22

        sig_33 = M_tot[0,1]*sig_13 + M_tot[0,3]*sig_23

        sig_34 = M_tot[0,1]*sig_14 + M_tot[0,3]*sig_24


        sig_41 = M_tot[2,0] + M_tot[2,1]*sig_11 + M_tot[2,3]*sig_21

        sig_42 = M_tot[2,2] + M_tot[2,1]*sig_12 + M_tot[2,3]*sig_22

        sig_43 = M_tot[2,1]*sig_13 + M_tot[2,3]*sig_23

        sig_44 = M_tot[2,1]*sig_14 + M_tot[2,3]*sig_24


        s = np.array([[sig_11, sig_12, sig_13, sig_14], [sig_21, sig_22, sig_23, sig_24], [sig_31, sig_32, sig_33, sig_34], [sig_41, sig_42, sig_43, sig_44]])
        return cls(vellocity*s, s, logg)


    @classmethod
    def insulator(cls, *arg, **karg):
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
