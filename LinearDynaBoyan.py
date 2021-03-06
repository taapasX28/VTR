import numpy as np
from tqdm import tqdm
from envs import Boyan
import matplotlib.pyplot as plt


def policy(state):
    """Uniform random policy for Boyan chain env"""
    if state<=2:
        action = 0
    else:
        action = np.random.choice([0,1])
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
        self.Dinv = np.zeros((self.feature_size, self.feature_size))
        self.Dinv = 1/500*np.identity(self.feature_size)
        self.I = np.identity(self.feature_size)
        self.buffer = []
        
    def boyan_reward(self, state):
        '''
        The true reward function for Boyan Chain.
        '''
        P, R = Boyan.getPR()
        return R[state]

    def get_phi(self, state):
        '''
        Computes the boyan chain representation the size is feature vector size
        '''
        #Gets the coded features
        feature_encoder = Boyan.BoyanRep()
        phi = feature_encoder.encode(state)
        return phi
    
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
        
        #Update our transition model using gradient descent
        self.F = self.F + self.B * np.outer((self.phi_ - np.dot(self.F,self.phi)),self.phi)
        #Update our reward model using gradient descent
        self.f = self.f + self.B * (r - np.inner(self.f, self.phi)) * self.phi
        
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
            
            #row = np.random.randint(self.iht_size)
            #phi_tilde = self.I[row]
            
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
        value_states = R @ np.linalg.inv(( I- self.gamma * P))
        return value_states

    def run(self):
        '''
        Runs the rl algorithm and returns the loss.
        '''
        print("Linear-Dyna")
        true_value_states = self.get_val()
        feature_encoder = Boyan.BoyanRep()
        map = feature_encoder.getmap()
        R = 0
        N_0 = 100.0
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
            self.alpha_l = self.alpha_l * ((N_0 +1)/(N_0 + (k)**1.1))
            L = np.linalg.norm(true_value_states - np.dot(self.theta ,map.T))
            loss.append(L)
            print(L)
        return loss

#number of episodes
K = 30
#num of runs
runs = 30
#the environment
env = Boyan.Boyan()

feature_size= 25
#max number of interactions with an environment before a reset, chosen according to hengshaui's work
steps = 98
#learning rate for theta
alpha_l = 0.1
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