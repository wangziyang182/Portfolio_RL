import numpy as np 
import pickle as pkl



class Env():

    def __init__(self,training_period,horizon,cs,cp):
        with open('./Data/input_tensor.pkl','rb') as f:
            #4 dimensional tensor
            #[batch,]
            data = pkl.load(f)

        self.full_state = np.transpose(data,(1,2,0))[None,...]
        self.full_horizon = self.full_state.shape[2]
        self.action_space = self.full_state.shape[1]
        self.start = 0
        self.training_period = training_period
        self.horizon = horizon
        self.cs =cs 
        self.cp = cp

    def test_reset(self,test_start):
        # self.test_start = 12000
        return self.full_state[:,:,test_start:test_start+self.horizon,:]
    def reset(self):
        self.start = np.random.randint(0,self.full_horizon - 12000)
        return self.full_state[:,:,self.start:self.start+self.horizon,:]

    def step(self,action,action_prev,):
        '''
        take action

        input: action

        return: next state
                reward
                boolean done
        '''
        state = self.full_state[:,:,self.start:self.start+self.horizon,:]
        vt = state[0,:,-1,3]
        vt_1 = state[0,:,-2,3]

        wt = action
        wt_1 = action_prev

        mu = self.get_mu(vt, vt_1, wt_1, wt)
        # print(mu)
        r = self.get_reward(vt, vt_1,mu,wt_1)

        self.start += 1
        
        return self.full_state[:,:,self.start:self.start+self.horizon,:],r

    # def get_mu(self,vt, vt_1, wt_1, wt):
    #     """
    #     cs = 0.01
    #     cp = 0.03
    #     vt =  np.array([10,20,30])
    #     vt_1 = np.array([9,18,32])
    #     wt_1 = np.array([0.4,0.4,0.2])
    #     wt = np.array([0.2,0.2,0.4])
    #     """
    #
    #     cs = self.cs
    #     cp = self.cp
    #     yt = vt_1/vt
    #     yt[0] = 1
    #     wt_1 = wt_1.flatten()
    #     wt = wt.flatten()
    #     wt_prime = (yt*wt_1)/(yt@wt_1)
    #     mu = 0.8
    #     right = 1/(1-cp*wt[0])*(1-cp*wt_prime[0]-(cs+cp-cs*cp)*sum(np.maximum((wt_prime-mu*wt)[1:],0,(wt_prime-mu*wt)[1:])))
    #     while(abs(mu-right)>0.001):
    #         mu = right
    #         right = 1/(1-cp*wt[0])*(1-cp*wt_prime[0]-(cs+cp-cs*cp)*sum(np.maximum((wt_prime-mu*wt)[1:],0,(wt_prime-mu*wt)[1:])))
    #     return mu


    def get_reward(self,vt,vt_1,mu,wt_1):
        wt_1 = wt_1.flatten()
        yt = vt_1/vt
        yt[0] = 1
        r = mu*yt@wt_1
        return r

    def get_mu(self,vt, vt_1, wt_1, wt):
        """
        cs = 0.01
        cp = 0.03
        vt =  np.array([10,20,30])
        vt_1 = np.array([9,18,32])
        wt_1 = np.array([0.4,0.4,0.2])
        wt = np.array([0.2,0.2,0.4])
        """

        cs = self.cs
        cp = self.cp
        yt = vt_1/vt
        yt[-1] = 1
        wt_1 = wt_1.flatten()
        wt = wt.flatten()
        wt_prime = (yt*wt_1)/(yt@wt_1)
        mu = 0.8
        right = 1/(1-cp*wt[-1])*(1-cp*wt_prime[-1]-(cs+cp-cs*cp)*sum(np.maximum((wt_prime-mu*wt)[:-1].copy(),0,(wt_prime-mu*wt)[:-1].copy())))
        while(abs(mu-right)>0.0001):
            mu = right
            right =1/(1-cp*wt[-1])*(1-cp*wt_prime[-1]-(cs+cp-cs*cp)*sum(np.maximum((wt_prime-mu*wt)[:-1].copy(),0,(wt_prime-mu*wt)[:-1].copy())))
        # if mu<0:
        #
        # print('\n')
        # print('yt',yt)
        # print('wt_1', wt_1,sum(wt_1))
        # print('wt', wt,sum(wt))
        #     print('wt_prime',wt_prime)
        #     print('nominator',(yt*wt_1))
        #     print('denominator',(yt@wt_1))
        #     print()
        assert  mu>0,'mu has to be larger than 0, otherwise it means we have to much risk'
        return mu


