'''
utility file provide helper function
'''
import numpy as np
import tensorflow as tf

def discounted_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_sum = 0
    for i in reversed(range(0,len(r))):
        discounted_r[i] = running_sum * gamma + r[i]
        running_sum = discounted_r[i]
    return list(discounted_r)


def add_FC_layer(inputs,in_size,out_size, activation_function = None):
    Weights = tf.Variable(tf.random.normal([in_size,out_size]),tf.float32)
    Bias = tf.Variable(tf.zeros([1,out_size]),tf.float32)
    Wx_plus_b = tf.matmul(inputs,Weights) + Bias
    if activation_function == None:
        output = Wx_plus_b
    else:
        output = activation_function(Wx_plus_b)

    return output

def add_arguments(parser):
    parser.add_argument("--num_asset", type=int, default=10, help="asset number.")
    parser.add_argument("--num_horizon", type=int, default=50, help="time step.")
    parser.add_argument("--num_signal_feature",type=int, default = 6, help = "number of signal axiuliary feature")
    parser.add_argument("--learning_rate_policy_net", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--learning_rate_value_net", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--Optimizer_policy_net", type = float, default = 1e-3, help = "hpyerparam for policy net optimier")
    parser.add_argument("--Optimizer_value_net", type = float, default = 1e-3, help = "hpyerparam for value net optimier")
    parser.add_argument("--discount rate", type = float, default = 1, help = 'dicounted rate for reward')
    parser.add_argument("--with_model", action="store_true", help="Continue from previously saved model")


