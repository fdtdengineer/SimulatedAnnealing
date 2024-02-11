
#%%
import numpy as np


boundary = "open" # "periodic"
phase = np.pi
#npr_g = np.array([-5,-1,3,4,7,6,2,-8], dtype=complex)
npr_g = np.array([-2,6,-3,4,5,2,-4], dtype=complex)

### build hamiltonian ###
# npr_g の符号をすべて正にした変数を作る
npr_g_abs = np.abs(npr_g)
npr_gdiag = np.concatenate([npr_g_abs,[0]]) + np.concatenate([[0], npr_g_abs])
print(npr_gdiag)
npr_g = -1.j*npr_g
npr_gdiag = -1.j*npr_gdiag
n = npr_g.shape[0] + 1

H = np.zeros((n,n), dtype=complex)
for i in range(n):
    H[i,i] = npr_gdiag[i]

for i in range(n-1):
    H[i,i+1] = npr_g[i]
    H[i+1,i] = npr_g[i]


### eigenvalues and eigenvectors ###
eigenvalues, eigenvectors = np.linalg.eig(H)
#print(np.round(eigenvalues.imag, 2))
print("max imag:"+str(np.max(eigenvalues.imag)))

# get the index which has the highest imag part of eigenvalue
idx = np.argmax(np.imag(eigenvalues))
eig0 = eigenvalues[idx]
vec0 = eigenvectors[:,idx]
print(eig0)

# plot the eigenvector
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.bar(range(n), np.abs(vec0))
plt.ylim(0,1)
plt.show()

# phase
phase = np.angle(vec0) / np.pi
phase = phase - phase[0]
phase = np.array([x-2 if x>= 1.5 else x for x in phase])
phase = np.array([x+2 if x<=-0.5 else x for x in phase])


# mod 2
#phase = np.mod(phase, 2)

# heatmap
fig, ax = plt.subplots()
cax = ax.matshow(phase.reshape(1,-1), cmap=cm.hot)#, vmin=0, vmax=1)
fig.colorbar(cax)
plt.show()

######################

phi = np.array(phase, dtype=complex)*np.pi
phi_i = phi.reshape(-1,1)
phi_j = phi.reshape(1,-1)
phi_ij = phi_i - phi_j

vtensor = np.cos(phi_ij)
# round
vtensor = np.round(vtensor, 2).real
print("vtensor:\n", vtensor)

loss = H * vtensor
loss = np.sum(loss)
print("loss:", loss)


#%%
print(H)


