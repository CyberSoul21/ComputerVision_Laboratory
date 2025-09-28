from labSession1.plotData_v15 import ensamble_T
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Load ground truth
R_w_c1 = np.loadtxt('R_w_c1.txt')
R_w_c2 = np.loadtxt('R_w_c2.txt')

t_w_c1 = np.loadtxt('t_w_c1.txt')
t_w_c2 = np.loadtxt('t_w_c2.txt')

#de camara a mundo,
T_w_c1 = ensamble_T(R_w_c1, t_w_c1)
T_w_c2 = ensamble_T(R_w_c2, t_w_c2)
#Queremos lo contrario de mundo a camara
T_c1_w = np.linalg.inv(T_w_c1)
T_c2_w = np.linalg.inv(T_w_c2)

K_c = np.loadtxt('K.txt')

I_3x3 = np.identity(3)          # 3x3 identity
col_zeros = np.zeros((3,1))     # column of zeros (3x1)
I_matrix = np.hstack((I_3x3, col_zeros))

"""
print("I_3x3 ")
print(I_3x3)
print("I: ")
print(I_matrix)

print("T_w_c1: ")
print(T_w_c1)

print("T_w_c2: ")
print(T_w_c2)


print("T_c1_w: ")
print(T_c1_w)

print("T_c2_w: ")
print(T_c2_w)

print("K_c: ")
print(K_c)
"""

X_A =  np.array([3.44, 0.80, 0.82, 1])
X_B =  np.array([4.20, 0.80, 0.82, 1])
X_C =  np.array([4.20, 0.60, 0.82, 1])
X_D =  np.array([3.55, 0.60, 0.82, 1])
X_E =  np.array([-0.01, 2.6, 1.21, 1])

"""
print("X_A: ")
print(X_A)
print("X_B: ")
print(X_B)
print("X_C: ")
print(X_C)
print("X_D: ")
print(X_D)
"""

points = np.array([X_A,X_B,X_C,X_D,X_E]).transpose()
"""
print("Points: ")
print(points)
"""
#Projection Matrices P1 and P2 // P = K[T_w_c]
print("#Projection Matrices P1 and P2 // P = K[T_w_c]")
#P1 = K_c @ T_c1_w[0:3,:]
P1 = K_c @ I_matrix @ T_c1_w
print("P1: ")
print(P1)
#P2 = K_c @ T_c2_w[0:3,:]
P2 = K_c @ I_matrix @ T_c2_w
print("P2: ")
print(P2)


###################################################

# One homogeneous point X_A = np.array([3.44, 0.80, 0.82, 1.0])

# Dot products with rows of P
As_u = np.dot(P1[0], X_A)   # row 1 · point
As_v = np.dot(P1[1], X_A)   # row 2 · point
s   = np.dot(P1[2], X_A)   # row 3 · point
A_2d_im1 = (np.array([As_u , As_v]).transpose())*(1/s)
#print("A_2d_im1: ")
#print(A_2d_im1)


# Dot products with rows of P
Bs_u = np.dot(P1[0], X_B)   # row 1 · point
Bs_v = np.dot(P1[1], X_B)   # row 2 · point
s   = np.dot(P1[2], X_B)   # row 3 · point
B_2d_im1 = (np.array([Bs_u , Bs_v]).transpose())*(1/s)
#print("B_2d_im1: ")
#print(B_2d_im1)

# Dot products with rows of P
Cs_u = np.dot(P1[0], X_C)   # row 1 · point
Cs_v = np.dot(P1[1], X_C)   # row 2 · point
s   = np.dot(P1[2], X_C)   # row 3 · point
C_2d_im1 = (np.array([Cs_u , Cs_v]).transpose())*(1/s)
#print("C_2d_im1: ")
#print(C_2d_im1)

# Dot products with rows of P
Ds_u = np.dot(P1[0], X_D)   # row 1 · point
Ds_v = np.dot(P1[1], X_D)   # row 2 · point
s   = np.dot(P1[2], X_D)   # row 3 · point
D_2d_im1 = (np.array([Ds_u , Ds_v]).transpose())*(1/s)
#print("D_2d_im1: ")
#print(D_2d_im1)


# Dot products with rows of P
Es_u = np.dot(P1[0], X_E)   # row 1 · point
Es_v = np.dot(P1[1], X_E)   # row 2 · point
s   = np.dot(P1[2], X_E)   # row 3 · point
E_2d_im1 = (np.array([Es_u , Es_v]))*(1/s)
#print("E_2d_im1: ")
print(E_2d_im1.transpose())









