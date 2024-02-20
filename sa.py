#%%
# https://qiita.com/ShataKurashi/items/c0c6044e97fa9e4a9471
if True:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib import rc
    from abc import *
    import time
    rc('text', usetex=False)
    fs = 18
    plt.rcParams['font.family']= 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams["font.size"] = fs # 全体のフォントサイズが変更されます。
    plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
    plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')



class StatModel (ABC):
    def __init__ (self, N, J, h):
        self.N = N
        self.J = J
        self.h = h

    @abstractmethod
    def hamiltonian (self, state):
        raise NotImplementedError()

    @abstractmethod
    def update (self, state):
        raise NotImplementedError()

    @abstractmethod
    def initializeState (self):
        raise NotImplementedError()

    def prob (self, current, proposal, temperature):
        tmp = np.exp ((current-proposal)/temperature)
        return min ([tmp,1.])

    def scheduleLogarithm (self, step, beta0=1.0):
        return 1/ (beta0 * np.log (2+step))


    def annealing (self, maxStep, Tinit=1e3, C=1-1e-3, seed=24):
        '''
        雑にSAする
        '''
        np.random.seed (seed)
        t = Tinit
        hams = []
        ts = []
        state = self.initializeState ()
        startTime = time.time ()
        times = []

        for step in range (maxStep):
            ham = self.hamiltonian (state)
            hams.append (ham)
            times.append (time.time()-startTime)
            proposal = self.update (state.copy ())
            hamprop = self.hamiltonian (proposal)
            if (np.random.rand () < self.prob (ham,hamprop,t)):
                state = proposal

            t *= C    # 温度を下げる


        hams.append (ham)
        times.append (time.time()-startTime)
        return state, hams, times


class IsingModel (StatModel):

    def __init__ (self, N, J, h):
        super().__init__ (N, J, h.reshape ((N,)));

    # override
    def hamiltonian (self, state):
        tmp1 = -0.5 * state.reshape((1,self.N)) @ self.J @ state.reshape((self.N,1))
        tmp2 = -state.reshape ((1,self.N)) @ self.h
        return (tmp1 + tmp2)[0,0]

    # override
    def update (self, state):
        idx = np.random.randint (low=0,high=self.N)
        state[idx] = - state[idx]    # flip
        return state

    # override
    def initializeState (self):
        return np.sign (np.random.randn (self.N,))


class XYModel (StatModel):

    def __init__ (self, N, J, h):
        '''
        h は2次元列ベクトルの外場
        '''
        super().__init__ (N, J, h.reshape ((2,N)))

    # override
    def hamiltonian (self, state):
        '''
        state は2次元単位列ベクトルがN個
        '''
        tmp1 = -0.5 * self.J * (state.T @ state)
        tmp2 = - self.h * state
        loss = tmp1.sum() + tmp2.sum()
        return loss

    #override
    def update (self, state):
        idx = np.random.randint (low=0, high=self.N)
        tmp = 2*np.pi*np.random.rand ()
        state[0,idx] = np.cos (tmp)
        state[1,idx] = np.sin (tmp)
        #print (np.arctan2 (state[1,idx], state[0,idx]))
        return state

    # override
    def initializeState (self):
        tmp = 2*np.pi*np.random.rand (1,self.N)
        return np.r_[np.cos (tmp), np.sin (tmp)]


import matplotlib.cm as cm


if __name__ == "__main__":    
    import numpy as np

    # rand
    np.random.seed(0)

    n1 = 3 # サイズ横
    n2 = 3 # サイズ縦
    npr_elem_gh = np.random.randn(n2,n1-1) # 横結合
    #npr_elem_gh = np.ones((n2,n1-1)) # 横結合
    #npr_elem_gh = np.array([[1,1,1,-1,1,1,1,1,1]])
    """
    npr_elem_gh = np.array([
        [1, 1, -1, 1],
        [1, 1, 1, 1]
        ])
    """
    #npr_elem_gv = np.array([1,0,0,-1,1]) 
    npr_elem_gv = np.random.randn(n1*(n2-1)) # 縦結合
    #npr_elem_gv = np.ones(n1*(n2-1)) # 縦結合

    """
    npr_elem_gh = np.array([
            [1,1],
            [1,1],
            [1,-1.01]
            ])#*(-1)
    npr_elem_gv = np.array([
            1,1,-1.,#1, 1,-1,
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

    H = H.imag

    lx = n1
    ly = n2
    Nx = lx*ly
    Jx = np.ones ((Nx,Nx))
    Jx = H#.imag
    hx = np.zeros ((2, Nx))
    maxStepx = 10000*3

    modeltype = "XY"
    #modeltype = "Ising"

    if modeltype == "XY":
        hx = np.zeros ((2, Nx))
        modelx = XYModel(Nx, Jx, hx)
    
    elif modeltype == "Ising":
        hx = np.zeros ((1, Nx))
        modelx = IsingModel(Nx, Jx, hx)

    statex, hamsx, timesx = modelx.annealing(maxStepx)

    loss = -0.5*np.sum(modelx.J * (statex.T @ statex))
    print("loss:", loss)
    
    if modeltype == "XY":
        phase = np.arctan2(statex[1,:],statex[0,:]).reshape(ly,lx) / np.pi + 0.5
        phase = phase - phase[0][0]
        phase = phase.flatten()
        phase = np.array([x-2 if x>= 1.5 else x for x in phase])
        phase = np.array([x+2 if x<=-0.5 else x for x in phase])
    elif modeltype == "Ising":
        phase = statex * statex[0]*(-1)
        phase = (phase + 1) / 2
    phase_vec = phase.copy()    
    phase = phase.reshape(ly,lx)

    plt.figure ()
    plt.plot (range(len(hamsx)), hamsx, color="gray")
    plt.xlabel ('Iteration')    
    plt.ylabel ('Energy')
    plt.figure ()
    """
    im = plt.imshow (
                phase,
                cmap=cm.hot,
                vmin=-1, vmax=1)
    plt.colorbar(im)

    ##############
    """

    phi = np.array(phase)*np.pi
    # phi[0][0] を基準にする
    phi = phi - phi[0][0]
    phi *= -1 # 反転対称性
    
    
    # tensor product
    # phi_i - phi_j の値を持つ行列を作る
    phi_i = phi.reshape(-1,1)
    phi_j = phi.reshape(1,-1)
    phi_ij = phi_i - phi_j

    vtensor = np.cos(phi_ij)

    loss = H * vtensor
    loss = np.sum(loss)
    print("loss_replicated:", -loss/2)

    # quiver
    coef = 0.5
    X = np.arange(0, lx, 1)
    Y = np.arange(0, ly, 1)
    X, Y = np.meshgrid(X, Y)
    U = coef*np.cos(phi - np.pi/2)
    V = coef*np.sin(phi - np.pi/2)
    
    plt.figure(figsize=(4,4))
    plt.quiver(X-U/2, Y-V/2, U, V, angles='xy', scale_units='xy', scale=1)
    plt.xlim(-0.5, lx-0.5)
    plt.ylim(ly-0.5, -0.5)
    #plt.xlabel('location x')
    #plt.ylabel('location y')
    
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.tick_params(bottom=False, left=False, right=False, top=False)
    #plt.box(False)
    plt.tight_layout()


    plt.show()




# %%
