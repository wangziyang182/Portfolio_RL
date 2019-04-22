from Policy_Net import Policy_Net
from Value_Net import Value_Net
from utils import add_arguments,discounted_rewards
import tensorflow as tf 
import numpy as np
import pickle as pkl
import argparse
import os
from Env import Env


parser = argparse.ArgumentParser()
add_arguments(parser)
args = parser.parse_args()


if not os.path.exists("Params"):
    os.mkdir("Params")
    
with open("./Params/args.pickle","wb") as f:
    pkl.dump(args,f)




#network parameter
sess = tf.Session()
feature_depth = args.num_signal_feature
num_asset = args.num_asset
horizon = args.horizon
policy_lr = args.learning_rate_policy_net
value_lr = args.learning_rate_value_net
gamma = args.discount_rate
tc = args.transcation_cost
Simga = args.sigma
sigma_decay = args.variance_decay
numtrajs = args.num_traj
iteration = args.iteration
depth1 = 2
depth2 = 1
max_train = args.max_train
training_period = args.max_train
cs = args.cs  
cp = args.cp


# initialize environment
env = Env(training_period,horizon,cs,cp)
# obssize = env.observation_space.low.size
actsize = env.action_space

# sess
sess = tf.Session()

# initialize networks
optimizer_policy = tf.train.AdamOptimizer(policy_lr)
optimizer_value = tf.train.AdamOptimizer(policy_lr)
sigma = [0.8] * num_asset

Sigma = np.diag(sigma)

policy = Policy_Net(sess,feature_depth,num_asset,horizon,optimizer_policy,tc,depth1,depth2,sigma)
value = Value_Net(sess,optimizer_value,feature_depth,num_asset,horizon,depth1 = 6)

# initialize tensorflow graphs
sess.run(tf.global_variables_initializer())

# main iteration
for ite in range(1):    
# for ite in range(1):    

    # trajs records for batch update
    OBS = []  # observations
    ACTS = []  # actions
    ADS = []  # advantages (to update policy)
    VAL = []  # value functions (to update baseline)

    for num in range(numtrajs):
    # for num in range():

        # record for each episode
        obss = []  # observations
        acts = []   # actions
        rews = []  # instant rewards

        obs = env.reset()
        w = []
        w.append(np.zeros((1,10)))
        w[0][:,-1] = 1
        for i in range(10):
            mu = policy.get_mu(obs,w[i]).flatten()
            action = np.random.multivariate_normal(mu,Sigma)

            # action = np.random.choice(actsize, p=prob.flatten(), size=1)
            newobs, reward = env.step(action,w[i])
            w.append(action[None,...])

            # record
            obss.append(obs)
            acts.append(action[0])
            rews.append(reward)

            # update
            obs = newobs
       
        # compute returns from instant rewards
        returns = discounted_rewards(rews, gamma)

        # record for batch update
        VAL += returns
        OBS += obss
        ACTS += acts
    
    # update baseline
    VAL = np.array(VAL)
    OBS = np.array(OBS)
    ACTS = np.array(ACTS)
    
    OBS = OBS[:,0,:]
    value.train(OBS, VAL)  # update only one step
    
    # update policy
    BAS = baseline.compute_values(OBS)  # compute baseline for variance reduction
    ADS = VAL - np.squeeze(BAS,1)

    actor.train(OBS, ACTS, ADS)  # update only one step

