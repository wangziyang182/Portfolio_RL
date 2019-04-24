from Policy_Net import Policy_Net
from Value_Net import Value_Net
from utils import add_arguments, discounted_rewards
import tensorflow as tf
import numpy as np
import pickle as pkl
import argparse
import os
from Env import Env
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser()
add_arguments(parser)
args = parser.parse_args()

if not os.path.exists("Params"):
    os.mkdir("Params")

with open("./Params/args.pickle", "wb") as f:
    pkl.dump(args, f)

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
Sigma = np.identity(10)*0.01
sigma_decay = args.variance_decay
numtrajs = args.num_traj
iteration = args.iteration
depth1 = 2
depth2 = 1
max_train = args.max_train
training_period = args.max_train
step_size = args.step_size
cs = 1e-3
# args.cs
cp = 1e-3

# initialize environment
env = Env(training_period, horizon, cs, cp,step_size)
# obssize = env.observation_space.low.size
actsize = env.action_space

# sess
sess = tf.Session()

# initialize networks
optimizer_policy = tf.train.AdamOptimizer(policy_lr)
optimizer_value = tf.train.AdamOptimizer(policy_lr)
sigma = [0.8] * num_asset

Sigma = np.diag(sigma)

policy = Policy_Net(sess, feature_depth, num_asset, horizon, optimizer_policy, tc, depth1, depth2, sigma)
value = Value_Net(sess, optimizer_value, feature_depth, num_asset, horizon, depth1=6)
# policy.restore()
# value.restore()
# initialize tensorflow graphs
sess.run(tf.global_variables_initializer())

# main iteration
test_start = 12050
test_end = 12100
total_rews = []

Prev_Acts = []
OBS = []  # observations
ACTS = []  # actions
ADS = []  # advantages (to update policy)
VAL = []  # value functions (to update baseline)

# for num in range(numtrajs):
#     # for num in range(10):
#
#     # record for each episode

obss = []  # observations
acts = []  # actions
prev_acts = []
rews = []  # instant rewards
#
prev_acts.append(np.zeros((1, num_asset)))
prev_acts[0][:, -1] = 1
#
obs = env.test_reset(test_start)

        # print('ddddd',prev_acts)
for i in range(test_end-test_start):
    # print('obs',obs[0,2,:,3])
    # print('fl',sess.run(policy.fl,feed_dict = {policy.state_tensor:obs,policy.w_t_prev:prev_acts[i]}))
    # print('nornamlized_obs',sess.run(policy.normalize_state_tensor,feed_dict = {policy.state_tensor:obs,policy.w_t_prev:prev_acts[i]})[0,0,:,3])
    mu = policy.get_mu(obs, prev_acts[i]).flatten()


    mu = mu
    # mu[-1] = 1 - sum(mu[:-1])
    action = mu
    newobs, reward = env.step(action, prev_acts[i])
    if i != test_end-test_start-1:
        prev_acts.append(action[None, ...])
    print(action[None,...])
    obss.append(obs)
    acts.append(action[None, ...])
    rews.append(reward)

    print(obs[0,0,:,3])
    # update
    obs = newobs
    # compute returns from instant rewards
    returns = discounted_rewards(rews, gamma)
    total_rews.extend(rews)

total_reward = np.prod(rews)
print(total_reward,rews)