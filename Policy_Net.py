import tensorflow as tf
import numpy as np
import pickle as pkl
from utils import add_FC_layer

#haven't add cash bias into it yet

class Policy_Net():
    '''
    Actor Critic
    '''

    def __init__(self,sess,feature_depth,num_asset,horizon,optimizer,trading_cost,depth1,depth2):


        self.num_asset = num_asset
        self.horizon = horizon
        self.feature_depth = feature_depth
        self.optimizer = optimizer
        self.sess = sess
        self.tc = tc
        self.depth1 = depth1
        self.depth2 = depth2


        with tf.variable_scope('input'):
            self.state_tensor = tf.placeholder(tf.float32,[None,self.num_asset,self.horizon,self.feature_depth])
            self.normalize_state_tensosr = tf.div(self.state_tensor,self.state_tensor[:,:,-1:,:])
            # self.state_tensor = self.state_tensor / self.state_tensor[:,:,-1:,:]

            self.w_t_prev = tf.placeholder(tf.float32,[None,self.num_asset]) #+ 1])

        with tf.variable_scope('policy_net'):
            # shape_state_t = tf.shape(self.state_tensor)[1:]

            with tf.variable_scope('conv1'):
                self.conv1 = tf.layers.conv2d(
                    self.state_tensor,
                    filters = depth1,
                    kernel_size = [1,3],
                    strides=(1, 1),
                    padding='valid',
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    bias_initializer=tf.zeros_initializer(),
                )
            self.shape = tf.shape(self.conv1)
            print(self.shape)
            with tf.variable_scope('conv2'):
                self.conv2 = tf.layers.conv2d(
                    self.conv1,
                    filters = depth2,
                    kernel_size = [1,self.conv1.get_shape()[2]],
                    strides=(1, 1),
                    padding='valid',
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    bias_initializer=tf.zeros_initializer(),
                )

            w_t_prev = tf.expand_dims(self.w_t_prev,-1)
            w_t_prev = tf.expand_dims(w_t_prev,-1)
            self.conv3_input = tf.concat([self.conv2,w_t_prev],axis =3)

            with tf.variable_scope('conv3'):
                self.conv3 = tf.layers.conv2d(
                    self.conv3_input,
                    filters = 1,
                    kernel_size = [1,1],
                    strides=(1, 1),
                    padding='valid',
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    use_bias=True,
                    bias_initializer=tf.zeros_initializer(),
                )

            in_size = int(self.conv3.get_shape()[1])
            out_size = in_size
            print(in_size)
            print(out_size)
            self.FC_input = tf.squeeze(self.conv3,[-1,-2])


            with tf.variable_scope('fully_connected'):
                # self.fl = self.add_layer(self.conv3_input,in_size,out_size, activation_function = tf.nn.tanh)
                self.fl = add_FC_layer(self.FC_input,in_size,out_size, activation_function = tf.nn.tanh)
            # assert tf.shape(self.fl)[0] == 1, "check fully connected layer 1st dimension"
            
            self.w_t = self.fl/tf.reduce_sum(tf.abs(self.fl),axis = 1)
            self.w_t = tf.transpose(self.w_t,perm = [1,0])


            # w_t = self.fl/tf.reduce_sum(tf.abs(self.fl_ut),axis = 1)
            # self.w_t_ut = tf.transpose(w_t,perm = [1,0])

    def add_layer(self,inputs,in_size,out_size, activation_function = None):
        Weights = tf.Variable(tf.random.normal([in_size,out_size]),tf.float32)
        Bias = tf.Variable(tf.zeros([1,out_size]),tf.float32)
        Wx_plus_b = tf.matmul(inputs,Weights) + Bias
        if activation_function == None:
            output = Wx_plus_b
        else:
            output = activation_function(Wx_plus_b)

        return output

    def get_action(self,X,W):
        return self.sess.run(self.w_t,feed_dict = {self.state_tensor:X,self.w_t:W})

    # def trian_net(self,X,W):




if __name__ == '__main__':

    
    sess = tf.Session()
    feature_depth = 6
    num_asset = 10
    horizon = 50
    optimizer = tf.train.AdamOptimizer(0.01)
    tc = 0.02
    depth1 = 2
    depth2 = 1

    policy = Policy_Net(sess,feature_depth,num_asset,horizon,optimizer,tc,depth1,depth2)
    sess.run(tf.global_variables_initializer())

    with open('/Users/william/Google Drive/STUDY/Columbia 2019 Spring/RL8100/Project/Finance/Portfolio_RL/Data/input_tensor.pkl','rb') as f:
        data = pkl.load(f)

    print(data.shape)
    w = np.random.randn(1,10)
    x = data[:,:,0:50:1] + 1e-10
    x = np.transpose(x, (1, 2, 0))[None,...]


    # state_tensor = sess.run(policy.normalize_state_tensosr,feed_dict = {policy.state_tensor: x, policy.w_t:w})
    # print('state_tensor',state_tensor.shape)
    conv1 = sess.run(policy.conv1,feed_dict = {policy.state_tensor: x, policy.w_t_prev:w})
    print('conv1',conv1.shape)
    conv2 = sess.run(policy.conv2,feed_dict = {policy.state_tensor: x, policy.w_t_prev:w})
    print('conv2',conv2.shape)
    conv3 = sess.run(policy.conv3_input,feed_dict = {policy.state_tensor: x, policy.w_t_prev:w})
    print('conv3_input',conv3.shape)
    conv3 = sess.run(policy.conv3,feed_dict = {policy.state_tensor: x, policy.w_t_prev:w})
    print('conv3',conv3.shape)
    conv3_input = sess.run(policy.conv3_input,feed_dict = {policy.state_tensor: x, policy.w_t_prev:w})
    print('conv3_input',conv3_input.shape)
    fl = sess.run(policy.fl,feed_dict = {policy.state_tensor: x, policy.w_t_prev:w})
    print('fl',fl.shape)
    w_t = sess.run(policy.w_t,feed_dict = {policy.state_tensor: x, policy.w_t_prev:w})
    print('w_t',w_t.shape)
    print('w_t',w_t)
    # w_t = sess.run(policy.w_t_ut,feed_dict = {policy.state_tensor: x, policy.w_t_prev:w})

    # print('w_t_ut',w_t)

    # conv1,conv2,w_t,_ = policy.test(x,w)
    # print(conv1.shape)
    # print(conv2.shape)
    # print(w_t.shape)




    


    # def weights_variable(shape):
    #     initial = tf.random.truncated_normal(shape,stddev = 0.001)

    #     return tf.variable(initial)

    # def bias_variable(shape):
    #     initial = tf.random.truncated_nomr




