import tensorflow as tf
import numpy as np
import pickle as pkl
from utils import add_FC_layer, get_normal
import os


#haven't add cash bias into it yet

class Policy_Net():
    '''
    Actor Critic
    '''

    def __init__(self,sess,feature_depth,num_asset,horizon,optimizer,trading_cost,depth1,depth2,sigma):


        self.num_asset = num_asset
        self.horizon = horizon
        self.feature_depth = feature_depth
        self.optimizer = optimizer
        self.sess = sess
        self.tc = trading_cost
        self.depth1 = depth1
        self.depth2 = depth2
        self.sigma = sigma


        with tf.variable_scope('input'):
            self.state_tensor = tf.placeholder(tf.float32,[None,self.num_asset,self.horizon,self.feature_depth])
            self.normalize_state_tensor = (tf.div(self.state_tensor,self.state_tensor[:,:,-1:,:])) * 100
            self.w_t_prev = tf.placeholder(tf.float32,[None,self.num_asset]) #+ 1])

        with tf.variable_scope('policy_net'):
            # shape_state_t = tf.shape(self.state_tensor)[1:]

            with tf.variable_scope('conv1'):
                self.conv1 = tf.layers.conv2d(
                    self.normalize_state_tensor,
                    filters = depth1,
                    kernel_size = [1,3],
                    strides=(1, 1),
                    padding='valid',
                    data_format='channels_last',
                    activation=tf.nn.tanh,
                    kernel_initializer=tf.random_normal_initializer(stddev=0.1, dtype=tf.float32),
                    bias_initializer=tf.zeros_initializer(),
                )
            self.shape = tf.shape(self.conv1)
            with tf.variable_scope('conv2'):
                self.conv2 = tf.layers.conv2d(
                    self.conv1,
                    filters = depth2,
                    kernel_size = [1,self.conv1.get_shape()[2]],
                    strides=(1, 1),
                    padding='valid',
                    data_format='channels_last',
                    activation=tf.nn.tanh,
                    kernel_initializer=tf.random_normal_initializer(stddev=0.1, dtype=tf.float32),
                    bias_initializer=tf.zeros_initializer(),
                )

            w_t_prev = tf.expand_dims(self.w_t_prev,-1)
            w_t_prev = tf.expand_dims(w_t_prev,-1)
            # w_t_prev = tf.ones(tf.shape((self.conv2))) * w_t_prev
            self.conv3_input = tf.concat([self.conv2,w_t_prev],axis =3)

            with tf.variable_scope('conv3'):
                self.conv3 = tf.layers.conv2d(
                    self.conv3_input,
                    filters = 1,
                    kernel_size = [1,1],
                    strides=(1, 1),
                    padding='valid',
                    data_format='channels_last',
                    activation=tf.nn.tanh,
                    kernel_initializer=tf.random_normal_initializer(stddev=0.1, dtype=tf.float32),
                    use_bias=True,
                    bias_initializer=tf.zeros_initializer(),
                )

            in_size = int(self.conv3.get_shape()[1])
            out_size = in_size
            self.FC_input = tf.squeeze(self.conv3,[-1,-2])


            with tf.variable_scope('fully_connected'):
                self.fl = add_FC_layer(self.FC_input,in_size,out_size, activation_function = tf.nn.tanh)
            
            # w_t = self.w_t
            Sigma = tf.diag(sigma)
            Sigma_inv = tf.linalg.inv(Sigma)

            self.w_t = tf.transpose(self.fl,perm = [1,0])
            self.top = self.w_t[:-1,:]
            self.bot = tf.expand_dims((1 - tf.reduce_sum(self.w_t[:-1,:],axis = 0)),0)
            self.w_t = tf.concat([self.top,self.bot],axis = 0)

            w_t = tf.transpose(self.w_t)

            with tf.variable_scope('training'):
                self.advantage = tf.placeholder(tf.float32,[None])
                self.actions = tf.placeholder(tf.float32,[None,num_asset])
                ele = (-1/2 * tf.matmul(tf.matmul((self.actions - w_t),Sigma_inv),tf.transpose((self.actions - w_t))))
                self.surrogate_loss = -tf.reduce_mean(tf.diag_part(ele) * self.advantage)
                self.train_op = optimizer.minimize(self.surrogate_loss)

            self.saver = tf.train.Saver()

    def get_mu(self,X,w):
        return self.sess.run(self.w_t,feed_dict = {self.state_tensor:X,self.w_t_prev:w})

    def train_net(self,X,w,advantage,actions):
        _,loss = self.sess.run([self.train_op,self.surrogate_loss], feed_dict = {self.state_tensor:X,self.w_t_prev:w,self.advantage:advantage,self.actions:actions})
        print('policy loss',loss)

    def save(self):
        if not os.path.exists("Model_Params"):
            os.mkdir("Model_Params")
        self.saver.save(self.sess,"./Model_Params/policy_net.cpkt")

    def restore(self):
        self.saver.restore(self.sess, "./Model_Params/policy_net.ckpt")


if __name__ == '__main__':

    
    sess = tf.Session()
    feature_depth = 6
    num_asset = 10
    horizon = 50
    optimizer = tf.train.AdamOptimizer(0.01)
    tc = 0.02
    depth1 = 2
    depth2 = 1
    sigma = [0.8] * num_asset

    policy = Policy_Net(sess,feature_depth,num_asset,horizon,optimizer,tc,depth1,depth2,sigma)
    sess.run(tf.global_variables_initializer())

    with open('./Data/input_tensor.pkl','rb') as f:
        data = pkl.load(f)

    # print(data.shape)
    w = np.zeros((2,10))
    w[:,-1] = 1
    # print(w)
    x = data[:,:,0:50:1]
    x = np.transpose(x, (1, 2, 0))[None,...]

    x = np.random.randn(2,10,50,6)
    # normalize_state_tensosr = sess.run(policy.normalize_state_tensosr,feed_dict = {policy.state_tensor: x, policy.w_t_prev:w})
    # print(normalize_state_tensosr)
    # conv1 = sess.run(policy.conv1,feed_dict = {policy.state_tensor: x, policy.w_t_prev:w})
    # print('conv1',conv1)
    conv2 = sess.run(policy.conv2,feed_dict = {policy.state_tensor: x, policy.w_t_prev:w})
    print('conv2',conv2.shape)
    conv3_input = sess.run(policy.conv3_input,feed_dict = {policy.state_tensor: x, policy.w_t_prev:w})
    print('conv3_input',conv3_input)
    # conv3 = sess.run(policy.conv3,feed_dict = {policy.state_tensor: x, policy.w_t_prev:w})
    # print('conv3',conv3)
    # conv3_input = sess.run(policy.conv3_input,feed_dict = {policy.state_tensor: x, policy.w_t_prev:w})
    # print('conv3_input',conv3_input)
    # fl = sess.run(policy.fl,feed_dict = {policy.state_tensor: x, policy.w_t_prev:w})
    # print('fl',fl)
    w_t = sess.run(policy.w_t,feed_dict = {policy.state_tensor: x, policy.w_t_prev:w})
    print('w_t',w_t.shape)
    print('w_t',w_t)
    top = sess.run(policy.top,feed_dict = {policy.state_tensor: x, policy.w_t_prev:w})
    print('top',top)
    bot = sess.run(policy.bot,feed_dict = {policy.state_tensor: x, policy.w_t_prev:w})
    print('bot',bot)

    actions = np.random.randn(2,10)
    advantage = np.ones(2)

    # print('hello madafaka',sess.run(policy.ele, feed_dict = {policy.state_tensor:x,policy.w_t_prev:w,policy.advantage:advantage,policy.actions:actions}))
    sess.run(policy.train_op, feed_dict = {policy.state_tensor:x,policy.w_t_prev:w,policy.advantage:advantage,policy.actions:actions})