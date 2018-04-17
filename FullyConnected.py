#!/usr/bin/env python
#__author__ = 'neevsamuel'
import tensorflow as tf
import numpy as np
import time as tm



"""
Data generation for train and test phases
"""
def generate_data(B,K,N,snr_low,snr_high,H_org):
    #H_=np.random.randn(B,N,K)
    W_=np.zeros([B,K,K])
    x_=np.sign(np.random.rand(B,K)-0.5)
    y_=np.zeros([B,N])
    w=np.random.randn(B,N)
    Hy_=x_*0
    H_ = np.zeros([B,N,K])
    HH_= np.zeros([B,K,K])
    SNR_= np.zeros([B])
    for i in range(B):
	#print i
	SNR = np.random.uniform(low=snr_low,high=snr_high)
        H=H_org
        tmp_snr=(H.T.dot(H)).trace()/K
        H=H/np.sqrt(tmp_snr)*np.sqrt(SNR)
        H_[i,:,:]=H
        y_[i,:]=(H.dot(x_[i,:])+w[i,:])
        Hy_[i,:]=H.T.dot(y_[i,:])
        HH_[i,:,:]=H.T.dot( H_[i,:,:])
	SNR_[i] = SNR
    return y_,H_,Hy_,HH_,x_,SNR_


def affine_layer(x,input_size,output_size,Layer_num):
    W = tf.Variable(tf.random_normal([input_size, output_size], stddev=0.01))
    w = tf.Variable(tf.random_normal([1, output_size], stddev=0.01))
    y = tf.matmul(x, W)+w
    return y

def relu_layer(x,input_size,output_size,Layer_num):
    y = tf.nn.relu(affine_layer(x,input_size,output_size,Layer_num))
    return y



"""
simulation - parameters:
B - batch size
train_iter - num of train iterations
test_iter -  num of test iteration
low_snr_db_train - the low range of snr in db during train phase
high_snr_db_train - the high range of snr in db during train phase
low_snr_db_test - the low range of snr in db during test phase
high_snr_db_test - the high range of snr in db during test phase
num_snr -  number of SNRs to be tested, equally spaced between low_snr_db_test and high_snr_db_test
fc_size  -  the size of each hidden layer
startingLearningRate - the initial step size of the gradient descent algorithm
decay_factor & decay_step_size - each decay_step_size steps the learning rate decay by decay_factor
H - the channel to detect over
"""
K = 20
N = 30
B = 1000
train_iter = 1000000
test_iter = 1000
low_snr_db_train = 7.0
high_snr_db_train = 14.0
low_snr_db_test = 8.0
high_snr_db_test = 13.0
num_snr = 6
fc_size = 200
num_of_hidden_layers = 4
startingLearningRate = 0.0003
decay_factor = 0.97
decay_step_size = 1000
H=np.genfromtxt('Top06_30_20.csv', dtype=None, delimiter=',')
#end of parameters

low_snr_train = 10.0 ** (low_snr_db_train/10.0)
high_snr_train = 10.0 ** (high_snr_db_train/10.0)
low_snr_test = 10.0 ** (low_snr_db_test/10.0)
high_snr_test = 10.0 ** (high_snr_db_test/10.0)
bers = np.zeros((1,num_snr))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

sess = tf.InteractiveSession()


NNinput = tf.placeholder(tf.float32, shape=[None, N], name='input')
org_siganl = tf.placeholder(tf.float32, shape=[None, K], name='org_siganl')
batchSize = tf.placeholder(tf.int32)

#The network
h_fc = []
W_fc_input = weight_variable([N, fc_size])
b_fc_input = bias_variable([fc_size])
h_fc.append(tf.nn.relu(tf.matmul(NNinput, W_fc_input) + b_fc_input))


for i in range(num_of_hidden_layers):
    h_fc.append(relu_layer(h_fc[i-1],fc_size, fc_size,'relu'+str(i)))


W_fc_final = weight_variable([fc_size, K])
b_final= bias_variable([K])
h_final= tf.matmul(h_fc[i], W_fc_final) + b_final

ssd = tf.reduce_sum(tf.square(org_siganl - h_final))

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(startingLearningRate, global_step, decay_step_size, decay_factor, staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(ssd)

val = tf.reshape(org_siganl, tf.stack([K * batchSize]))
final = tf.reshape(h_final, tf.stack([K * batchSize]))
rounded = tf.sign(final)
eq = tf.equal(rounded, val)
eq2 = tf.reduce_sum(tf.cast(eq, tf.int32))


accuracy = ssd

sess.run(tf.global_variables_initializer())
"""
training phase og the network
"""
for i in range(train_iter):
    batch_Y, batch_H, batch_HY, batch_HH, batch_X , SNR1= generate_data(B , K , N , low_snr_train , high_snr_train , H)
    if i % 100 == 0: 
        correct_bits = eq2.eval(feed_dict={NNinput: batch_Y, org_siganl: batch_X, batchSize: B})
        train_accuracy = accuracy.eval(feed_dict={NNinput: batch_Y, org_siganl: batch_X, batchSize: B})
        print("step %d, loss is %g, number of correct bits %d" % (i, train_accuracy,correct_bits))
    train_step.run(feed_dict={NNinput: batch_Y, org_siganl: batch_X, batchSize: B})

"""
start testing our net
"""
tmp_bers = np.zeros((1,test_iter))
tmp_times = np.zeros((1,test_iter))
times = np.zeros((1,1))
testHitCount = 0

snr_list_db = np.linspace(low_snr_db_test,high_snr_db_test,num_snr)
snr_list = 10.0**(snr_list_db/10.0)
for i_snr in range (num_snr):
    Cur_SNR = snr_list[i_snr]
    print 'cur snr'
    print Cur_SNR
    for i in range(test_iter):

	batch_Y, batch_H, batch_HY, batch_HH, batch_X , SNR1= generate_data(B , K , N , Cur_SNR , Cur_SNR , H)
	tic = tm.time()
        tmp_bers[0][i] =   eq2.eval(feed_dict={NNinput: batch_Y, org_siganl: batch_X, batchSize: B}	)
	toc = tm.time()
	tmp_times[0][i] = toc - tic
      	if i % 100 == 0:  
           	eq2.eval(feed_dict={NNinput: batch_Y, org_siganl: batch_X, batchSize: B})
            	train_accuracy = accuracy.eval(feed_dict={NNinput: batch_Y, org_siganl: batch_X, batchSize: B})
            	print("test accuracy %g" % eq2.eval(feed_dict={NNinput: batch_Y, org_siganl: batch_X, batchSize: B}))
    bers[0][i_snr] = np.mean(tmp_bers[0])
    
times[0][0] = np.mean(tmp_times[0])/B
print ('Average time to detect a single K bit signal is:')
print times
bers =  bers/(K*B)
bers = 1- bers
snrdb_list = np.linspace(low_snr_db_test,high_snr_db_test,num_snr)
print('snrdb_list')
print(snrdb_list)
print ('Bit error rates are is:')
print bers
