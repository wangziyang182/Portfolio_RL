from Policy_Net import Policy_Net
from Value_Net import Value_Net
from utils import add_arguments
import tensorflow as tf 
import numpy as np
import pickle as pkl
import argparse
import os


parser = argparse.ArgumentParser()
add_arguments(parser)
args = parser.parse_args()

print(args.num_asset)

if not os.path.exists("Params"):
    os.mkdir("Params")
    

with open("./Params/args.pickle","wb") as f:
    pkl.dump(args,f)


# sess = tf.Session()
# feature_depth = 6
# num_asset = 10
# horizon = 50
# optimizer = tf.train.AdamOptimizer(0.01)
# tc = 0.02
# depth1 = 2
# depth2 = 1

# policy = Policy_Net(sess,feature_depth,num_asset,horizon,optimizer,tc,depth1,depth2)
# sess.run(tf.global_variables_initializer())

# with open('/Users/william/Google Drive/STUDY/Columbia 2019 Spring/RL8100/Project/Finance/Portfolio_RL/Data/input_tensor.pkl','rb') as f:
#     data = pkl.load(f)

# print(data.shape)
# w = np.random.randn(1,10)
# x = data[:,:,0:50:1] + 1e-10
# x = np.transpose(x, (1, 2, 0))[None,...]


# # state_tensor = sess.run(policy.normalize_state_tensosr,feed_dict = {policy.state_tensor: x, policy.w_t:w})
# # print('state_tensor',state_tensor.shape)
# conv1 = sess.run(policy.conv1,feed_dict = {policy.state_tensor: x, policy.w_t_prev:w})
# print('conv1',conv1.shape)
# conv2 = sess.run(policy.conv2,feed_dict = {policy.state_tensor: x, policy.w_t_prev:w})
# print('conv2',conv2.shape)
# conv3 = sess.run(policy.conv3_input,feed_dict = {policy.state_tensor: x, policy.w_t_prev:w})
# print('conv3_input',conv3.shape)
# conv3 = sess.run(policy.conv3,feed_dict = {policy.state_tensor: x, policy.w_t_prev:w})
# print('conv3',conv3.shape)
# conv3_input = sess.run(policy.conv3_input,feed_dict = {policy.state_tensor: x, policy.w_t_prev:w})
# print('conv3_input',conv3_input.shape)
# fl = sess.run(policy.fl,feed_dict = {policy.state_tensor: x, policy.w_t_prev:w})
# print('fl',fl.shape)
# w_t = sess.run(policy.w_t,feed_dict = {policy.state_tensor: x, policy.w_t_prev:w})
# print('w_t',w_t.shape)
# print('w_t',w_t)
# # w_t = sess.run(policy.w_t_ut,feed_dict = {policy.state_tensor: x, policy.w_t_prev:w})

# # print('w_t_ut',w_t)

# # conv1,conv2,w_t,_ = policy.test(x,w)
# # print(conv1.shape)
# # print(conv2.shape)
# # print(w_t.shape)







# # def weights_variable(shape):
# #     initial = tf.random.truncated_normal(shape,stddev = 0.001)

# #     return tf.variable(initial)

# # def bias_variable(shape):
# #     initial = tf.random.truncated_nomr




