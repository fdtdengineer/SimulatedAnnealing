
#%%
import numpy as np

# rand
np.random.seed(0)

n1 = 5 # サイズ横
n2 = 5 # サイズ縦
npr_elem_gh = np.random.randn(n2,n1-1) # 横結合
#npr_elem_gh = np.array([[1,1,1,-1,1,1,1,1,1]])
"""
npr_elem_gh = np.array([
    [1, 1, -1, 1],
    [1, 1, 1, 1]
    ])
"""
#npr_elem_gv = np.array([1,0,0,-1,1]) 
npr_elem_gv = np.random.randn(n1*(n2-1)) # 縦結合

"""
npr_elem_gh = np.array([
        [1,1],
        [1,1],
        [1,-1.01]
        ])#*(-1)
npr_elem_gv = np.array([
        1,2,-1.,#1, 1,-1,
        1,1,0,#1, 1,-1,
        ])
"""

# 非対角成分が npr_g1 である n1 x n1 行列 を n2 個作る
npr_gh_block = np.zeros((n2,n1,n1), dtype=complex)
for i in range(n2):
    npr_gh_block[i] = np.diagflat(npr_elem_gh[i], k=1) + np.diagflat(npr_elem_gh[i], k=-1)

# npr_gh の各要素をブロック対角行列として結合
npr_gh = np.zeros((n2*n1,n2*n1), dtype=complex)
for i in range(n2):
    npr_gh[i*n1:(i+1)*n1, i*n1:(i+1)*n1] = npr_gh_block[i]

npr_gv = np.zeros((n1*n2,n1*n2), dtype=complex)
for i in range(n1*(n2-1)):
    npr_gv[i,i+n1] = npr_elem_gv[i]
    npr_gv[i+n1,i] = npr_elem_gv[i]

# np.round(npr_g,2).real # debug
npr_g = npr_gh + npr_gv

# npr_g の対角成分を、それぞれの行の非対角成分の絶対値の和にする
npr_g_abs = np.abs(npr_g)
npr_elem_gdiag = np.sum(npr_g_abs, axis=1)
npr_gdiag = np.diagflat(npr_elem_gdiag)

H = 1.j*npr_g -1.j*npr_gdiag

#print(np.round(H.imag,5))
#print(np.round(eigenvalues.imag,5))




### eigenvalues and eigenvectors ###
eigenvalues, eigenvectors = np.linalg.eig(H)
print(np.round(eigenvalues.imag, 3))
#print("max imag:"+str(np.max(eigenvalues.imag)))

# get the index which has the highest imag part of eigenvalue
idx = np.argmax(np.imag(eigenvalues))
eig0 = eigenvalues[idx]
vec0 = eigenvectors[:,idx]
#print(eig0)

# plot the eigenvector
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# phase
phase = np.angle(vec0) / np.pi
phase = phase - phase[0]
phase = np.array([x-2 if x>= 1.5 else x for x in phase])
phase = np.array([x+2 if x<=-0.5 else x for x in phase])
phase = phase.reshape(-1,n1)

# mod 2
#phase = np.mod(phase, 2)

# heatmap
fig, ax = plt.subplots()
cax = ax.matshow(phase, cmap=cm.hot, vmin=-1, vmax=1)
fig.colorbar(cax)
plt.show()


plt.figure()
plt.bar(range(len(vec0)), np.abs(vec0))
plt.show()
######################

phi = np.array(phase, dtype=complex)*np.pi
phi_i = phi.reshape(-1,1)
phi_j = phi.reshape(1,-1)
phi_ij = phi_i - phi_j

vtensor = np.cos(phi_ij)
# round
vtensor = np.round(vtensor, 2).real
#print("vtensor:\n", vtensor)

loss = H * vtensor
loss = np.sum(loss)
print("loss:", loss)









#%%

temp = np.array([[ 0.00000000e+00,  2.42874215e-03, -4.88440346e-04],
       [-4.92423227e-01, -4.87526797e-01, -4.89333382e-01],
       [-2.33219423e-02, -2.04629243e-02,  9.80212932e-01]])

# loss
temp *= np.pi
temp_i = temp.reshape(-1,1)
temp_j = temp.reshape(1,-1)
temp_ij = temp_i - temp_j

vtensor = np.cos(temp_ij)
# round
vtensor = np.round(vtensor, 2).real

loss = H * vtensor
loss = np.sum(loss)
print("loss:", loss)