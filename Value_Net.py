
import pickle as pkl
import numpy as np
from utils import add_FC_layer
import tensorflow as tf
import os


class Value_Net():
    '''
    Compute the Value of each state
    '''

    def __init__(self,sess,optimizer,feature_depth,num_asset,horizon,depth1):
        self.sess = sess
        self.optimizer = optimizer
        self.num_asset = num_asset
        self.horizon = horizon
        self.feature_depth = feature_depth

        with tf.variable_scope('input'):
            self.state = tf.placeholder(tf.float32, [None,num_asset,horizon, feature_depth])

        with tf.variable_scope('Value_Net'): #reuse=tf.AUTO_REUSE):
            with tf.variable_scope('conv1'):
                self.conv1 = tf.layers.conv2d(
                    self.state,
                    filters= depth1,
                    kernel_size = [1,self.horizon],
                    strides = (1,1),
                    padding = 'valid',
                    data_format = 'channels_last',
                    activation=tf.nn.relu,
                    bias_initializer=tf.zeros_initializer(),
                )

            with tf.variable_scope('conv2d'):
                self.conv2 = tf.layers.conv2d(
                    self.conv1,
                    filters = 1,
                    kernel_size = [1,self.conv1.get_shape()[2]],
                    strides = (1,1),
                    padding = 'valid',
                    data_format = 'channels_last',
                    activation = tf.nn.relu,
                    bias_initializer = tf.zeros_initializer()
                    )
            
            self.FC_input = tf.squeeze(self.conv2,[-1,-2])
            with tf.variable_scope('FC_layers'):
                self.fl = add_FC_layer(self.FC_input,num_asset,1)


            self.target = tf.placeholder(tf.float32,[None])
            self.state_loss = tf.reduce_mean(tf.square(self.fl - self.target))
            self.train_op = optimizer.minimize(self.state_loss)

            self.saver = tf.train.Saver()


    def get_state_value(self,state):
        return self.sess.run(self.fl,feed_dict = {self.state: state})


    def train(self,state,target):
        _,loss = self.sess.run([self.train_op,self.state_loss], feed_dict = {self.state:state,self.target:target})
        print('state loss',loss)

    def save(self):
        if not os.path.exists("Model_Params"):
            os.mkdir("Model_Params")
        self.saver.save(self.sess,"./Model_Params/value_net.cpkt")

    






# if __name__ == '__main__':
#     sess = tf.Session()
#     num_asset = 10
#     feature_depth = 6
#     horizon = 50
#     optimizer = tf.train.AdamOptimizer(0.01)
#     tc = 0.02
#     depth1 = 2
#     depth2 = 1

#     value = Value_Net(sess,optimizer,feature_depth,num_asset,horizon,depth1 = 6)
#     sess.run(tf.global_variables_initializer())

#     with open('/Users/william/Google Drive/STUDY/Columbia 2019 Spring/RL8100/Project/Finance/Portfolio_RL/Data/input_tensor.pkl','rb') as f:
#         data = pkl.load(f)

#     x = data[:,:,0:50:1] + 1e-10
#     x = np.transpose(x,(1,2,0))[None,...]

#     state = sess.run(value.state,feed_dict = {value.state:x})
#     print(state.shape)

#     conv1 = sess.run(value.conv1,feed_dict = {value.state:x})
#     print(conv1.shape)

#     conv2 = sess.run(value.conv2,feed_dict = {value.state:x})
#     print(conv2.shape)

#     FC_input = sess.run(value.FC_input,feed_dict = {value.state:x})
#     print(FC_input.shape)

#     fl = sess.run(value.fl,feed_dict = {value.state:x})

#     print(value.get_state_value(x))


