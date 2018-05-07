import tensorflow as tf
import utils
from tensorflow.python.training import moving_averages
import sys
from time import time
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.python.ops.rnn import dynamic_rnn

from md_lstm import *



FLAGS = utils.FLAGS
num_classes = utils.num_classes

class LSTMOCR(object):
    def __init__(self,mode):
        self.mode = mode
        # image
        #self.inputs = tf.placeholder(tf.float32, [None, FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
        # SparseTensor for ctc_loss op
        self.labels = tf.sparse_placeholder(tf.int32)
        # id array, size = [batch_size] <- Indicate the seq len of img a in batch d.
        self.seq_len = tf.placeholder(tf.int32, [None])
        # l2
        #self._extra_train_ops = []
        # Other stuffs
        self.learning_rate = FLAGS.initial_learning_rate
        self.decay_steps = FLAGS.decay_steps
        self.decay_rate = FLAGS.decay_rate
        self.batch_size = FLAGS.batch_size
        self.beta1 = FLAGS.beta1
        self.beta2 = FLAGS.beta2
        self.h = FLAGS.image_height
        self.w = FLAGS.image_width
        self.channels = FLAGS.image_channel
        self.hidden_size = FLAGS.num_hidden
        self.inputs = tf.placeholder(tf.float32, [None,self.h,self.w,self.channels])
        self.is_training = True

    def build_graph(self):
        self._build_model()
        self._build_train_op()

        self.merged_summay = tf.summary.merge_all()
        

    def _build_model(self):

        batch_norm_params = {'is_training': self.is_training, 'decay': 0.9, 'updates_collections': None}
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params):
                x = self.inputs #tf.reshape(inputs, [-1, self.height, self.width, 1])
                x = tf.reshape(x, [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
                # For slim.conv2d, default argument values are like
                # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
                # padding='SAME', activation_fn=nn.relu,
                # weights_initializer = initializers.xavier_initializer(),
                # biases_initializer = init_ops.zeros_initializer,
                net = slim.conv2d(x, 16, [5, 5], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                #net  = lstm2d.separable_lstm( net, 32, kernel_size=None, scope='lstm2d-1')
                net,_ = multi_dimensional_rnn_while_loop(rnn_size = 32, input_data = net, sh = [1,1], dims = None, scope_n = 'mdlstm1')
                net = slim.conv2d(net, 64, [5, 5], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                #net  = lstm2d.separable_lstm( net, 124, kernel_size=None, scope='lstm2d-2')
                net,_ = multi_dimensional_rnn_while_loop(rnn_size = 124, input_data = net, sh = [1,1], dims = None, scope_n = 'mdlstm2')

        ss = net.get_shape().as_list()
        shape = tf.shape(net)
        batch_size = shape[0]
        #bat, h, w , chanels
        outputs =  tf.transpose(net, [2,0,1,3])
        outputs =  tf.reshape(outputs, [-1, shape[1]*shape[3]])

        with tf.name_scope('Train'):
            with tf.variable_scope("ctc_loss-1") as scope:
                myInitializer = tf.truncated_normal_initializer(mean=0., stddev=.075, seed=None, dtype=tf.float32)
            
                W = tf.get_variable('w',[ss[1]*ss[3],200],initializer=myInitializer)
                # Zero initialization
                b = tf.get_variable('b', shape=[200],initializer=myInitializer)
                
                W1 = tf.get_variable('w1',[200,num_classes],initializer=myInitializer)
                # Zero initialization
                b1 = tf.get_variable('b1', shape=[num_classes],initializer=myInitializer)

            tf.summary.histogram('histogram-b-ctc', b)
            tf.summary.histogram('histogram-w-ctc', W)

        logits = tf.matmul(outputs, W) +  b 
        logits = slim.dropout(logits, is_training=self.is_training, scope='dropout4')
        logits = tf.matmul(logits, W1) +  b1


        logits = tf.reshape(logits, [batch_size, -1, num_classes])
        self.logits = tf.transpose(logits, (1, 0, 2))
        #self.logits = tf.reshape(logits, [ -1,batch_size, num_classes]) 

    def _build_model1(self):
        
        # maybe no y, we use self.labels ?? y = tf.placeholder(tf.float32, [batch_size, 

        # layer0 4 direction scanning mdlstm
        hidden0_0,_ = multi_dimensional_rnn_while_loop(rnn_size = 2, input_data = self.inputs, sh = [3,4], dims = None, scope_n = 'hidden0_0')
        hidden0_1,_ = multi_dimensional_rnn_while_loop(rnn_size = 2, input_data = self.inputs, sh = [3,4], dims = [1], scope_n = 'hidden0_1')
        hidden0_2,_ = multi_dimensional_rnn_while_loop(rnn_size = 2, input_data = self.inputs, sh = [3,4], dims = [2], scope_n = 'hidden0_2')
        hidden0_3,_ = multi_dimensional_rnn_while_loop(rnn_size = 2, input_data = self.inputs, sh = [3,4], dims = [1,2], scope_n = 'hedden0_3')

        
        hidden0_out = tf.concat([hidden0_0, hidden0_1, hidden0_2, hidden0_3], 3)
        hidden0_out = tf.reduce_mean(hidden0_out, axis = 3)
        hidden0_out = tf.expand_dims(hidden0_out, axis = 3)
        hidden0_out = tf.nn.tanh(hidden0_out)
        layer0_out = slim.fully_connected(inputs = hidden0_out, num_outputs = 6, activation_fn = tf.tanh) #num_outputs = num_classes?       
        # Debug shape


        # layer1 4 directions
        hidden1_0,_ = multi_dimensional_rnn_while_loop(rnn_size = 5, input_data = layer0_out, sh = [2,4], dims = None, scope_n = 'hidden1_0')
        hidden1_1,_ = multi_dimensional_rnn_while_loop(rnn_size = 5, input_data = layer0_out, sh = [2,4], dims = [1], scope_n = 'hidden1_1')
        hidden1_2,_ = multi_dimensional_rnn_while_loop(rnn_size = 5, input_data = layer0_out, sh = [2,4], dims = [2], scope_n = 'hidden1_2')
        hidden1_3,_ = multi_dimensional_rnn_while_loop(rnn_size = 5, input_data = layer0_out, sh = [2,4], dims = [1,2], scope_n = 'hedden1_3')

        hidden1_out = tf.concat([hidden1_0, hidden1_1, hidden1_2, hidden1_3], 3)
        hidden1_out = tf.reduce_mean(hidden1_out, axis = 3)
        hidden1_out = tf.expand_dims(hidden1_out, axis = 3)
        hidden1_out = tf.nn.tanh(hidden1_out)
        layer1_out = slim.fully_connected(inputs = hidden1_out, num_outputs = 20, activation_fn = tf.tanh) #num_outputs = num_classes? 

        # Debug shape
        #layer1_out = tf.Print(layer1_out, [tf.shape(layer1_out)],message = 'layer1_out.shape = ')

        # layer2 4 directions
        hidden2_0,_ = multi_dimensional_rnn_while_loop(rnn_size = 50, input_data = layer1_out, sh = [1,1], dims = None, scope_n = 'hidden2_0')
        hidden2_1,_ = multi_dimensional_rnn_while_loop(rnn_size = 50, input_data = layer1_out, sh = [1,1], dims = [1], scope_n = 'hidden2_1')
        hidden2_2,_ = multi_dimensional_rnn_while_loop(rnn_size = 50, input_data = layer1_out, sh = [1,1], dims = [2], scope_n = 'hidden2_2')
        hidden2_3,_ = multi_dimensional_rnn_while_loop(rnn_size = 50, input_data = layer1_out, sh = [1,1], dims = [1,2], scope_n = 'hedden2_3')

        hidden2_out = tf.concat([hidden2_0, hidden2_1, hidden2_2, hidden2_3], 3)
        layer2_out = slim.fully_connected(inputs = hidden2_out, num_outputs = num_classes, activation_fn = tf.tanh) #num_outputs = num_classes? 

        # Debug shape
        # layer2_out = tf.Print(layer2_out, [tf.shape(layer2_out)],message = 'layer2_out.shape = ')

        # Reshape for ctc
        shape = tf.shape(self.inputs)
        batch_s= shape[0]
        logits = tf.reshape(layer2_out, [batch_s, -1, num_classes])
        self.logits = tf.transpose(logits, (1, 0, 2))
        
    


    def _build_train_op(self):
        # loss from cnn_lstm_ctc
        self.global_step = tf.Variable(0,trainable = False)
        self.loss = tf.nn.ctc_loss(labels = self.labels, inputs = self.logits, sequence_length = self.seq_len)
        self.cost = tf.reduce_mean(self.loss)
        tf.summary.scalar('cost',self.cost)

        self.lrn_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate, self.global_step, FLAGS.decay_steps, FLAGS.decay_rate, staircase = True)
        #self.optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.initial_learning_rate, beta1= FLAGS.beta1, beta2 = FLAGS.beta2).minimize(self.loss, global_step = self.global_step)
        self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lrn_rate,momentum=FLAGS.momentum,use_nesterov=True).minimize(self.cost,global_step=self.global_step)
        train_ops = [self.optimizer]  # no '+ self._extra_train_ops' here since we don't have cnn
        self.train_op = tf.group(*train_ops)
        self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(self.logits, self.seq_len, merge_repeated = False)
        self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value =-1)    

        









































