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
        self._extra_train_ops = []
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

    def build_graph(self):
        
        # maybe no y, we use self.labels ?? y = tf.placeholder(tf.float32, [batch_size, 

        # 4 direction scanning mdlstm
        hidden0_0,_ = multi_dimensional_rnn_while_loop(rnn_size = self.hidden_size, input_data = self.inputs, sh = [3,4], dims = None, scope_n = 'hidden0_0')
        hidden0_1,_ = multi_dimensional_rnn_while_loop(rnn_size = self.hidden_size, input_data = self.inputs, sh = [3,4], dims = [1], scope_n = 'hidden0_1')
        hidden0_2,_ = multi_dimensional_rnn_while_loop(rnn_size = self.hidden_size, input_data = self.inputs, sh = [3,4], dims = [2], scope_n = 'hidden0_2')
        hidden0_3,_ = multi_dimensional_rnn_while_loop(rnn_size = self.hidden_size, input_data = self.inputs, sh = [3,4], dims = [1,2], scope_n = 'hedden0_3')

        hidden0_0 = tf.nn.dropout(hidden0_0,0.5)
        hidden0_1 = tf.nn.dropout(hidden0_1,0.5)
        hidden0_2 = tf.nn.dropout(hidden0_2,0.5)
        hidden0_3 = tf.nn.dropout(hidden0_3,0.5)
        
        hidden_out = tf.concat([hidden0_0, hidden0_1, hidden0_2, hidden0_3], 3)

        hidden_out = tf.Print(hidden_out, [tf.shape(hidden_out)], message = 'hidden_out.shape = ')

        
        # Suppose only 1 direction
        #print('hello hello')
        #print('rnn_out.shape = {0}'.format(str(rnn_out.get_shape().as_list())))
        #rnn_out_shape = tf.shape(rnn_out)
        #rnn_out = tf.reshape(rnn_out, [rnn_out_shape[0], -1, rnn_out_shape[3]])
        model_out = slim.fully_connected(inputs = hidden_out, num_outputs = num_classes, activation_fn = tf.tanh)
        model_out = tf.Print(model_out, [tf.shape(model_out)],message = 'model_out.shape = ')
        # Reshape for ctc
        shape = tf.shape(self.inputs)
        batch_s= shape[0]
        logits = tf.reshape(model_out, [batch_s, -1, num_classes])
        self.logits = tf.transpose(logits, (1, 0, 2))
        
        ''' /* loss from ctc_example */
        # Compute loss
        self.loss = tf.nn.ctc_loss(labels = self.labels, inputs = self.logits, sequence_length = self.seq_len)
        self.cost = tf.reduce_mean(self.loss)

        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate,0.9).minimize(self.cost)

        # Option 2: tf.nn.ctc_beam_search_decoder
        # (it's slower but you'll get better results)
        self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(self.logits, self.seq_len)

        # Inaccuracy: label error rate
        self.ler = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.labels))
        '''
        
        # loss from cnn_lstm_ctc
        self.global_step = tf.Variable(0,trainable = False)
        self.loss = tf.nn.ctc_loss(labels = self.labels, inputs = self.logits, sequence_length = self.seq_len)
        self.cost = tf.reduce_mean(self.loss)
        tf.summary.scalar('cost',self.cost)

        self.lrn_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase = True)
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate, beta1= self.beta1, beta2 = self.beta2).minimize(self.loss, global_step = self.global_step)

        train_ops = [self.optimizer]  # no '+ self._extra_train_ops' here since we don't have cnn
        self.train_op = tf.group(*train_ops)
        self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(self.logits, self.seq_len, merge_repeated = False)
        self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value =-1)    

        self.merged_summay = tf.summary.merge_all()









































