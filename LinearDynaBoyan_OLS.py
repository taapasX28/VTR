import numpy as np
from tqdm import tqdm
from envs import Boyan
import matplotlib.pyplot as plt


def policy(state):
    """Uniform random policy for Boyan chain env"""
    if state<=2:
        action = 0
    else:
        action = np.random.choice([0,1], p=[0.5, 0.5])
    return action

class LinearDyna(object):
    def __init__(self, env, K, steps, alpha_l, alpha_p, tau,gamma, feature_size, B):
        self.env = env
        self.K = K
        self.steps = steps
        self.alpha_l = alpha_l
        self.alpha_p = alpha_p
        self.tau = tau
        self.gamma = gamma
        self.B = B
        self.eps = 0.0
        self.num_actions = 2
        self.feature_size = feature_size
        self.theta = np.zeros(self.feature_size)
        self.F = np.zeros((self.feature_size, self.feature_size))
        self.f = np.zeros((self.feature_size))
        self.Dinv = 100*np.identity(self.feature_size)
        self.I = np.identity(self.feature_size)
        self.buffer = []

    def get_phi(self, state):
        '''
        Computes the boyan chain representation the size is feature vector size
        '''
        #Gets the coded features
        feature_encoder = Boyan.BoyanRep()
        phi = feature_encoder.encode(state)
        return phi

    def Sherman_Morrison(self,vec,Mat):
        Matvec = np.dot(Mat,vec)
        vec_vec_out = np.outer(Matvec,vec)
        top = np.matmul(vec_vec_out,Mat)
        vec_mat = np.dot(vec,Mat)
        bottom = 1 + np.inner(vec_mat,vec)
        
        return Mat - top/bottom
    
    def update_F(self):
        top = self.phi_ - np.dot(self.F,self.phi)
        bottom = 1 + np.inner(self.x,self.phi)
        
        frac = top / bottom
        
        mat_update = np.outer(frac,self.x)
        
        self.F = self.F + mat_update
    
    def update_f(self,r):
        
        top = r - np.inner(self.phi,self.f)
        bottom = 1 + np.inner(self.phi,self.x)
        
        frac = top / bottom
        
        vec_update = frac * self.x
        
        self.f = self.f + vec_update

    def act(self,s):
        
        a = policy(s)
        return a
    
    #Commented code below is for computing the exact Least Squares Update. Currently does not work.
   
    def update(self,s,a,r,s_,done):
        '''
        Updates the values of theta, our estimated transition model F, and our estimated reward model f.
        '''
        
        
        #Computes the coded feature for current state s
        self.phi = self.get_phi(s)
        
        #Computes the tile coded feature for next state s_
        self.phi_ = self.get_phi(s_)
        
        #Updates our theta values using gradient descent
        self.theta = self.theta + self.alpha_l*(r + self.gamma * np.inner(self.phi_,self.theta) \
                                             - np.inner(self.phi,self.theta))*self.phi
        self.Dinv = self.Sherman_Morrison(self.phi,self.Dinv)
        self.x = np.dot(self.Dinv, self.phi)
        self.update_F()
        self.update_f(r)
        #Runs our planning step.
        self.plan()
    
    def plan(self):
        '''
        Using Dyna-style planning to update our theta estimate with simulated experience on a learnt model.
        '''
        
        #initializes the theta using in planning to be the current theta estimate
        theta_tilde = self.theta
        
        #we do the planning portion p many times
        for p in range(self.tau):
            
            #Below are different ways to sample a state s for planning
            
            #Here we sample s uniformly from the space of all states
            
            #position = np.random.uniform(-1.2,0.6)
            #velocity = np.random.uniform(-0.07,0.07)
            #active_tiles_tilde = self.tc.get_tiles(position,velocity)
            #phi_tilde = self.get_phi(active_tiles_tilde)
            
            #Here we sample s from a buffer the stores all observed states
            
            row = np.random.randint(len(self.buffer))
            sample_state = self.buffer[row]
            phi_tilde = self.get_phi(sample_state)
            
            #Here we sample s from the support. Meaning we sample a unit vector as the state
            
            # row = np.random.randint(self.feature_size)
            # phi_tilde = self.I[row]
            
            #Compute the featurized next state given a featurized state and non featurized action
            phi_tilde_ = np.dot(self.F, phi_tilde)
            #compute the reward given a featurized state and a non featurized action
            r_tilde = np.inner(phi_tilde, self.f)
            #Update theta using the simulated experience 
            theta_tilde = theta_tilde + self.alpha_p * (r_tilde + self.gamma * np.inner(theta_tilde, phi_tilde_) \
                                                     - np.inner(theta_tilde,phi_tilde))*phi_tilde
        #Update the current estimate of theta to be the estimate from the simulation
        self.theta = theta_tilde
     
    def update_state_buffer(self, s):
        '''
        Updates the buffer with the curretn state s
        '''
        self.buffer.append(s)

    def get_val(self):
        """
        Gets true analytical value of states by solving
        Bellman equation directly
        """

        P, R = Boyan.getPR()
        I = np.identity(98)
        value_states = np.linalg.inv(( I- self.gamma * P)) @ R
        return value_states

    def run(self):
        '''
        Runs the rl algorithm and returns the loss.
        '''
        print("Linear-Dyna")
        true_value_states = self.get_val()
        feature_encoder = Boyan.BoyanRep()
        map = feature_encoder.getmap()
        loss = []
        for k in tqdm(range(1,self.K+1)):
            s = self.env.reset()
            done = None
            step = 0
            while not done:
                step += 1
                if  s==0 :
                    s = self.env.reset()
                self.update_state_buffer(s)
                a = self.act(s)
                r, s_, done = self.env.step(a)
                self.update(s,a,r,s_,done)
                s = s_
            L = np.linalg.norm(true_value_states - np.dot(map, self.theta)) / 10
            loss.append(L)
            print(L)
        print(true_value_states)
        print(np.dot(map, self.theta))
        print(self.theta)
        return loss

#number of episodes
K = 50
#num of runs
runs = 1
#the environment
env = Boyan.Boyan()

feature_size= 25
#max number of interactions with an environment before a reset, chosen according to hengshaui's work
steps = 98
#learning rate for theta
alpha_l = 0.001
#learning rate for theta tilde, should somehow scale with tau, the number of planning steps
alpha_p = 0.05
#number of planning steps
tau = 5
#The discounting factor, chosen according to hengshaui's work
gamma = 1.0
#The learning rate for updating the learnt models F and f, chosen according to hengshaui's work
B = 0.01
#A matrix the stores the step for each episode within a run.
loss = np.zeros((runs,K))
for i in tqdm(range(runs)):
    agent = LinearDyna(env,K, steps, alpha_l, alpha_p, tau, gamma, feature_size, B)
    loss[i,:] = agent.run()

#averages the result for each episode by the steps per run.
results = np.mean(loss, axis=0)
plt.plot(results)
plt.xlabel("Number of Episodes")
plt.ylabel("RMSE(Between analytical and predicted vals)")
plt.show()