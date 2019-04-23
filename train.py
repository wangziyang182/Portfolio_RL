from Policy_Net import Policy_Net
from Value_Net import Value_Net
from utils import add_arguments,discounted_rewards
import tensorflow as tf 
import numpy as np
import pickle as pkl
import argparse
import os
from Env import Env
import os
print(os.getcwd())
# import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
for ite in range(iteration):
    if ite%100==0:
        print("ite",ite)

    # trajs records for batch update
    Prev_Acts = []
    OBS = []  # observations
    ACTS = []  # actions
    ADS = []  # advantages (to update policy)
    VAL = []  # value functions (to update baseline)

    Sigma = np.exp(-ite * sigma_decay) * Sigma
    for num in range(numtrajs):
    # for num in range(10):

        # record for each episode
        obss = []  # observations
        acts = []   # actions
        prev_acts = []
        rews = []  # instant rewards

        prev_acts.append(np.zeros((1,10)))
        prev_acts[0][:,-1] = 1

        obs = env.reset()

        for i in range(max_train):
            mu = policy.get_mu(obs,prev_acts[i]).flatten()
            mu = mu/10
            action = np.random.multivariate_normal(mu,Sigma)
            action[-1] = 1-sum(action[:-1])

            # action = np.random.choice(actsize, p=prob.flatten(), size=1)
            newobs, reward = env.step(action,prev_acts[i])

            # record
            if i != (max_train - 1):
                prev_acts.append(action[None,...])
            obss.append(obs)
            acts.append(action[None,...])
            rews.append(reward)

            # update
            obs = newobs
        # compute returns from instant rewards
        returns = discounted_rewards(rews, gamma)

        # record for batch update
        Prev_Acts += prev_acts
        VAL += returns
        OBS += obss
        ACTS += acts
    # print(reward)

    # update baseline
    Prev_Acts = np.array(Prev_Acts)
    # print(Prev_Acts)
    VAL = np.array(VAL)
    OBS = np.array(OBS)
    ACTS = np.array(ACTS)
    
    OBS = OBS[:,0,:]
    ACTS = ACTS[:,0,:]
    Prev_Acts = Prev_Acts[:,0,:]
    # print('Prev_Acts',Prev_Acts.shape)
    # print('OBS',OBS.shape)
    # print('Acts',ACTS.shape)


    value.train(OBS, VAL)  # update only one step
    
    # update policy
    BAS = value.get_state_value(OBS)  # compute baseline for variance reduction
    ADS = VAL - np.squeeze(BAS,1)

    print(OBS.shape)
    print(Prev_Acts.shape)
    print(ADS.shape)
    print(ACTS.shape)

    policy.train_net(OBS,Prev_Acts,ADS,ACTS)  # update only one step
policy.save()
value.save()

#### IMMEDATE TEST


# network parameter
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
env = Env(training_period, horizon, cs, cp)
# obssize = env.observation_space.low.size
actsize = env.action_space

# sess
sess = tf.Session()

# # initialize networks
# optimizer_policy = tf.train.AdamOptimizer(policy_lr)
# optimizer_value = tf.train.AdamOptimizer(policy_lr)
sigma = [0.8] * num_asset

Sigma = np.diag(sigma)



# main iteration
test_start = 12000
test_end = 13000
periods = (test_end-test_start)//50

    # trajs records for batch update

Prev_Acts = []
OBS = []  # observations
ACTS = []  # actions
ADS = []  # advantages (to update policy)
VAL = []  # value functions (to update baseline)


obss = []  # observations
acts = []  # actions
prev_acts = []
rews = []  # instant rewards
#
prev_acts.append(np.zeros((1, 10)))
prev_acts[0][:, -1] = 1
#
obs = env.test_reset(test_start)

for i in range(test_end-test_start):
    mu = policy.get_mu(obs, prev_acts[i]).flatten()
    action =mu
    newobs, reward = env.step(action, prev_acts[i])
    if i != test_end-test_start-1:
        prev_acts.append(action[None, ...])
    obss.append(obs)
    acts.append(action[None, ...])
    rews.append(reward)

    # updatex
    obs = newobs
    # compute returns from instant rewards
    returns = discounted_rewards(rews, gamma)

step_reward = [np.prod(rews[:i+1]) for i in range(len(rews))]
plot_x_range = np.arange(len(step_reward))
# plt.plot(plot_x_range,step_reward)
# plt.show()

total_reward = np.prod(rews)
rews = np.array(rews)
os.chdir(os.getcwd())
if not os.path.exists("Testing_Reward"):
    os.mkdir("Testing_Reward")

np.save('./Testing_Reward/test_results', rews)
print(rews)
print(total_reward)

