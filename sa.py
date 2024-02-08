#%%
# https://qiita.com/ShataKurashi/items/c0c6044e97fa9e4a9471
import numpy as np
import matplotlib.pyplot as plt
from abc import *
import time

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


    def annealing (self, maxStep, Tinit=1e3, C=0.995, seed=24):
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
        return tmp1.sum() + tmp2.sum()

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
    lx = 20
    Nx = lx*lx
    Jx = np.ones ((Nx,Nx))
    hx = np.zeros ((2, Nx))
    maxStepx = 100000

    modelx = XYModel (Nx, Jx, hx)
    statex, hamsx, timesx = modelx.annealing (maxStepx)

    plt.figure ()
    plt.plot (timesx, hamsx)
    plt.xlabel ('times [sec]')
    plt.ylabel ('energy')
    plt.figure ()
    im = plt.imshow (np.arctan2 (statex[1,:],statex[0,:]).reshape (lx,lx), cmap=cm.hsv, vmin=0, vmax=2*np.pi)
    plt.colorbar(im)


