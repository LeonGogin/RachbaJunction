import numpy as np
import scipy.constants as cc

def set_zeros(a):
    tol = np.finfo(a.dtype).eps
    a.real[abs(a.real) < tol] = 0.0
    a.imag[abs(a.imag) < tol] = 0.0
    return a

from RachbaJunction import RachbaJunction

junction = RachbaJunction([(0, 1), (3, -1)], [0.005, 0, 0])

print(junction.E_so)


M_tot, vel_factor = junction.get_transfer_matrix(0)

M_tot = set_zeros(M_tot)


sig_11 = -(M_tot[1,0]*M_tot[2,2] - M_tot[1,2]*M_tot[2,0])/(M_tot[2,2]*M_tot[1,1]- M_tot[1,2]*M_tot[2,1])

sig_12 = M_tot[2,2]/(M_tot[2,2]*M_tot[1,1]- M_tot[1,2]*M_tot[2,1])

sig_21 = M_tot[0,0] + M_tot[0,1]* sig_11 - M_tot[0,2]*(M_tot[2,0]+M_tot[2,1]*sig_11)/M_tot[2,2]

sig_22 = (M_tot[0,1]- M_tot[0,2]*M_tot[2,1]/M_tot[2,2])*sig_12

sig =  np.array([[sig_11, sig_12], [sig_21, sig_22]])

print("velocity")
print(junction.vel_b)
print(junction.vel_a)

print("determinant")
print(np.linalg.det(M_tot))

print("sigma matrix")
print(vel_factor)

# print(sig, "\n\n") 
# print(np.dot(sig, sig.conj().T), "\n")

sig = vel_factor*sig

print(sig, "\n\n", sig.conj().T) 
print(np.dot(sig.conj().T, sig), "\n")

print(np.norm(sig[0]))