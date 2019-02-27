# CSIRO Open Source Software License Agreement (GPLv3)
# Commonwealth Scientific and Industrial Research Organisation (CSIRO) ABN 41 687 119 230.
# All rights reserved. CSIRO is willing to grant you a license to MDSeqVAE on the terms of the GNU General Public License version 3 as published by the Free Software Foundation (http://www.gnu.org/licenses/gpl.html), except where otherwise indicated for third party material.
# The following additional terms apply under clause 7 of that license:
# EXCEPT AS EXPRESSLY STATED IN THIS AGREEMENT AND TO THE FULL EXTENT PERMITTED BY APPLICABLE LAW, THE SOFTWARE IS PROVIDED "AS-IS". CSIRO MAKES NO REPRESENTATIONS, WARRANTIES OR CONDITIONS OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY REPRESENTATIONS, WARRANTIES OR CONDITIONS REGARDING THE CONTENTS OR ACCURACY OF THE SOFTWARE, OR OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT, THE ABSENCE OF LATENT OR OTHER DEFECTS, OR THE PRESENCE OR ABSENCE OF ERRORS, WHETHER OR NOT DISCOVERABLE.
# TO THE FULL EXTENT PERMITTED BY APPLICABLE LAW, IN NO EVENT SHALL CSIRO BE LIABLE ON ANY LEGAL THEORY (INCLUDING, WITHOUT LIMITATION, IN AN ACTION FOR BREACH OF CONTRACT, NEGLIGENCE OR OTHERWISE) FOR ANY CLAIM, LOSS, DAMAGES OR OTHER LIABILITY HOWSOEVER INCURRED.  WITHOUT LIMITING THE SCOPE OF THE PREVIOUS SENTENCE THE EXCLUSION OF LIABILITY SHALL INCLUDE: LOSS OF PRODUCTION OR OPERATION TIME, LOSS, DAMAGE OR CORRUPTION OF DATA OR RECORDS; OR LOSS OF ANTICIPATED SAVINGS, OPPORTUNITY, REVENUE, PROFIT OR GOODWILL, OR OTHER ECONOMIC LOSS; OR ANY SPECIAL, INCIDENTAL, INDIRECT, CONSEQUENTIAL, PUNITIVE OR EXEMPLARY DAMAGES, ARISING OUT OF OR IN CONNECTION WITH THIS AGREEMENT, ACCESS OF THE SOFTWARE OR ANY OTHER DEALINGS WITH THE SOFTWARE, EVEN IF CSIRO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH CLAIM, LOSS, DAMAGES OR OTHER LIABILITY.
# APPLICABLE LEGISLATION SUCH AS THE AUSTRALIAN CONSUMER LAW MAY APPLY REPRESENTATIONS, WARRANTIES, OR CONDITIONS, OR IMPOSES OBLIGATIONS OR LIABILITY ON CSIRO THAT CANNOT BE EXCLUDED, RESTRICTED OR MODIFIED TO THE FULL EXTENT SET OUT IN THE EXPRESS TERMS OF THIS CLAUSE ABOVE "CONSUMER GUARANTEES".  TO THE EXTENT THAT SUCH CONSUMER GUARANTEES CONTINUE TO APPLY, THEN TO THE FULL EXTENT PERMITTED BY THE APPLICABLE LEGISLATION, THE LIABILITY OF CSIRO UNDER THE RELEVANT CONSUMER GUARANTEE IS LIMITED (WHERE PERMITTED AT CSIRO’S OPTION) TO ONE OF FOLLOWING REMEDIES OR SUBSTANTIALLY EQUIVALENT REMEDIES:
# (a)               THE REPLACEMENT OF THE SOFTWARE, THE SUPPLY OF EQUIVALENT SOFTWARE, OR SUPPLYING RELEVANT SERVICES AGAIN;
# (b)               THE REPAIR OF THE SOFTWARE;
# (c)               THE PAYMENT OF THE COST OF REPLACING THE SOFTWARE, OF ACQUIRING EQUIVALENT SOFTWARE, HAVING THE RELEVANT SERVICES SUPPLIED AGAIN, OR HAVING THE SOFTWARE REPAIRED.
# IN THIS CLAUSE, CSIRO INCLUDES ANY THIRD PARTY AUTHOR OR OWNER OF ANY PART OF THE SOFTWARE OR MATERIAL DISTRIBUTED WITH IT.  CSIRO MAY ENFORCE ANY RIGHTS ON BEHALF OF THE RELEVANT THIRD PARTY.
# Third Party Components
# The following third party components are distributed with the Software.  You agree to comply with the license terms for these components as part of accessing the Software.  Other third party software may also be identified in separate files distributed with the Software.
# ___________________________________________________________________
# MDSeqVAE (https://github.com/dascimal-org/MDSeqVAE)
# This software is licensed under CSIRO GPLv3 License v2.0
# ___________________________________________________________________

import os
import tensorflow as tf
import time
import utils
import numpy as np
from collections import OrderedDict
from sklearn import metrics as mt
import sys
from tensorflow.python.util import nest
from tensorflow.python.framework import tensor_shape

def _state_size_with_prefix(state_size, prefix=None):
  """Helper function that enables int or TensorShape shape specification.

  This function takes a size specification, which can be an integer or a
  TensorShape, and converts it into a list of integers. One may specify any
  additional dimensions that precede the final state size specification.

  Args:
    state_size: TensorShape or int that specifies the size of a tensor.
    prefix: optional additional list of dimensions to prepend.

  Returns:
    result_state_size: list of dimensions the resulting tensor size.
  """
  result_state_size = tensor_shape.as_shape(state_size).as_list()
  if prefix is not None:
    if not isinstance(prefix, list):
      raise TypeError("prefix of _state_size_with_prefix should be a list.")
    result_state_size = prefix + result_state_size
  return result_state_size

def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)

def get_initial_cell_state(cell, initializer, batch_size, dtype):
    """Return state tensor(s), initialized with initializer.
    Args:
      cell: RNNCell.
      batch_size: int, float, or unit Tensor representing the batch size.
      initializer: function with two arguments, shape and dtype, that
          determines how the state is initialized.
      dtype: the data type to use for the state.
    Returns:
      If `state_size` is an int or TensorShape, then the return value is a
      `N-D` tensor of shape `[batch_size x state_size]` initialized
      according to the initializer.
      If `state_size` is a nested list or tuple, then the return value is
      a nested list or tuple (of the same structure) of `2-D` tensors with
    the shapes `[batch_size x s]` for each s in `state_size`.
    """
    state_size = cell.state_size
    if nest.is_sequence(state_size):
        state_size_flat = nest.flatten(state_size)
        init_state_flat = [
            initializer(_state_size_with_prefix(s), batch_size, dtype, i)
                for i, s in enumerate(state_size_flat)]
        init_state = nest.pack_sequence_as(structure=state_size,
                                    flat_sequence=init_state_flat)
    else:
        init_state_size = _state_size_with_prefix(state_size)
        init_state = initializer(init_state_size, batch_size, dtype, None)

    return init_state

def make_variable_state_initializer(**kwargs):
    def variable_state_initializer(shape, batch_size, dtype, index):
        args = kwargs.copy()

        if args.get('name'):
            args['name'] = args['name'] + '_' + str(index)
        else:
            args['name'] = 'init_state_' + str(index)

        args['shape'] = shape
        args['dtype'] = dtype

        var = tf.get_variable(**args)
        var = tf.expand_dims(var, 0)
        var = tf.tile(var, tf.stack([batch_size] + [1] * len(shape)))
        var.set_shape(_state_size_with_prefix(shape, prefix=[None]))
        return var

    return variable_state_initializer

def make_gaussian_state_initializer(initializer, deterministic_tensor=None, stddev=0.3):
    def gaussian_state_initializer(shape, batch_size, dtype, index):
        init_state = initializer(shape, batch_size, dtype, index)
        if deterministic_tensor is not None:
            return tf.cond(deterministic_tensor,
                lambda: init_state,
                lambda: init_state + tf.random_normal(tf.shape(init_state), stddev=stddev))
        else:
            return init_state + tf.random_normal(tf.shape(init_state), stddev=stddev)
    return gaussian_state_initializer

class RNNClassifier:
    def __init__(self, iPara):
        # -------------------- important parameters -------------------- #

        # self.data_size = 'tiny_dataset'
        self.data_size = 'full_dataset'
        self.running_mode = 1  # 0: test_and_compute_score, 1: train, 2: visualization
        self.important_notes = 'Bi-rnn 2 layer(s) (64,128) GRUCell + MM-OCSVM + tune gamma=0.25'
        self.datetime = utils._DATETIME
        # self.datetime = '2018-24-8-1-56-1'

        # -------------------- important parameters -------------------- #

        tf.set_random_seed(utils._RANDOM_SEED)
        self.logging_path = utils._LOG

        # For writing gradient and value into a easy readable file
        self.list_str_variables = OrderedDict()
        self.gradient_and_value = []

        # Training settings
        self.batch_size = 64
        self.num_train_steps = 100
        self.trade_off = self.batch_size
        self.display_step = 1
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.max_gradient_norm = 5.0
        self.learning_rate = 1e-3
        self.optimizer = 'adam'

        # Embedding
        self.vocab_assembly_size = 256

        # RNN hyper-parameter
        self.num_hidden = 256

        # VAE
        self.n_hidden_recog_1 = 500
        self.n_hidden_recog_2 = 500
        self.n_hidden_gener_1 = 500
        self.n_hidden_gener_2 = 500
        self.n_z = 2
        self.OutName = 'txtLog/256_lr1e3_RNN' + str(iPara) + '.txt'
        self.transfer_fct = tf.nn.softplus

        # Classifier hyper-parameter: e.g: SVM, Neural Nets
        self.num_random_features = 1024
        self.gamma_init = 0.25
        self.train_gamma = False
        self.lamda_l2 = 0.01
        self.num_classes = 2  # 0: non_vul_function, 1: vulnerable_function
        self.num_input = 50

        utils.save_all_params(self)

    def _create_placeholders(self):
        with tf.name_scope("input"):
            self.X_opcode = tf.placeholder(tf.float32, [None, self.time_steps, 1, self.vocab_opcode_size], name='x_opcode_input')
            self.X_assembly = tf.placeholder(tf.float32, [None, self.time_steps, 1, self.vocab_assembly_size], name='x_assemply_input')
            self.Y = tf.placeholder(tf.int32, [None], name='true_label')
            self.sequence_length = tf.placeholder(tf.int32, [None], name='seq_length')

    def _create_embedding(self):
        with tf.name_scope("embedding"):
            self.w_opcode = tf.Variable(tf.truncated_normal([self.vocab_opcode_size, self.num_input], stddev=0.05),
                                        name='w_opcode')

            self.w_assembly = tf.Variable(tf.truncated_normal([self.vocab_assembly_size, self.num_input], stddev=0.05),
                                          name='w_assembly')

            # (batch_size, time_steps, 1, vocab_size) * (vocab_size, num_input) = (batch_size, time_steps, 1, num_input)
            self.embed_opcode = tf.tensordot(self.X_opcode, self.w_opcode, axes=((3,), (0,)))
            self.embed_assembly = tf.tensordot(self.X_assembly, self.w_assembly, axes=((3,), (0,)))

            embed_opcode_assembly = tf.concat([self.embed_opcode, self.embed_assembly], axis=3)
            self.rnn_input = tf.reshape(embed_opcode_assembly, [self.batch_size, self.time_steps, 2*self.num_input])

    def _create_bi_rnn(self):
        with tf.name_scope("bi-rnn"):
            deterministic = tf.constant(False)
            #self.initial_stateRNN = tf.Variable(xavier_init(self.batch_size, self.num_hidden))
            
            cell = tf.nn.rnn_cell.GRUCell(self.num_hidden)
            initializer = make_gaussian_state_initializer(make_variable_state_initializer(), deterministic)
            self.init_state = get_initial_cell_state(cell, initializer, self.batch_size, tf.float32)
            self.outputs, self.states = tf.nn.dynamic_rnn(cell, self.rnn_input, dtype=tf.float32,
                                                                        sequence_length=self.sequence_length, initial_state=self.init_state)
            self.g_h_concatination = self.states
            #self.opAssignRNNInit = tf.assign(self.initial_stateRNN, self.states)

    def _create_vae(self):
        with tf.name_scope("vae"):
            network_weights = self._initialize_weights()
            self.z_mean, self.z_log_sigma_sq = self._recognition_network(network_weights["weights_recog"], network_weights["biases_recog"])
            eps = tf.random_normal((self.batch_size, self.n_z), 0, 1, 
                               dtype=tf.float32)
            # z = mu + sigma*epsilon
            self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

            self.h0zSoftmax = self._generator_network_0(network_weights["weights_gener"], network_weights["biases_gener"], self.init_state) #64 x 64
            self.h1NzSoftmax = self._generator_network(network_weights["weights_gener"], network_weights["biases_gener"], self.outputs) # 64 x 158 x 64
            self.h0NzSoftmax = tf.concat([tf.reshape(self.h0zSoftmax, (self.batch_size, 1, self.vocab_opcode_size)), self.h1NzSoftmax[:,:-1,:]], axis=1)

    def _initialize_weights(self):
        self.prior_0_mean = tf.Variable(-tf.ones([self.n_z], dtype=tf.float32))
        self.prior_1_mean = tf.Variable(tf.ones([self.n_z], dtype=tf.float32))
        self.prior_0_log_sigma_sq = tf.Variable(tf.zeros([self.n_z], dtype=tf.float32))
        self.prior_1_log_sigma_sq = tf.Variable(tf.zeros([self.n_z], dtype=tf.float32))
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_init(self.g_h_concatination.get_shape().as_list()[1], self.n_hidden_recog_1)),
            'h2': tf.Variable(xavier_init(self.n_hidden_recog_1, self.n_hidden_recog_2)),
            'out_mean': tf.Variable(xavier_init(self.n_hidden_recog_2, self.n_z)),
            'out_log_sigma': tf.Variable(xavier_init(self.n_hidden_recog_2, self.n_z))}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([self.n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([self.n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([self.n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([self.n_z], dtype=tf.float32))}
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(self.n_z + self.num_hidden, self.n_hidden_gener_1)),
            'h2': tf.Variable(xavier_init(self.n_hidden_gener_1, self.n_hidden_gener_2)),
            'out_mean': tf.Variable(xavier_init(self.n_hidden_gener_2, self.vocab_opcode_size)),
            'out_log_sigma': tf.Variable(xavier_init(self.n_hidden_gener_2, self.vocab_opcode_size))}
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([self.n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([self.n_hidden_gener_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([self.vocab_opcode_size], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([self.vocab_opcode_size], dtype=tf.float32))}
        return all_weights

    def _recognition_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.g_h_concatination, weights['h1']), 
                                           biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 
        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                        biases['out_mean'])
        z_log_sigma_sq = tf.add(tf.matmul(layer_2, weights['out_log_sigma']), 
                   biases['out_log_sigma'])
        return (z_mean, z_log_sigma_sq)

    def _generator_network_0(self, weights, biases, state):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        hz = tf.concat([state, self.z], axis=1)
        layer_1 = self.transfer_fct(tf.add(tf.matmul(hz, weights['h1']), 
                                           biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 
        hz_softmax = tf.nn.softmax(tf.add(tf.matmul(layer_2, weights['out_mean']), 
                                 biases['out_mean']), 1)
        return hz_softmax

    def _generator_network(self, weights, biases, state):
        hz = tf.concat([state, tf.tile(tf.reshape(self.z, (self.batch_size,1,self.n_z)), [1, self.time_steps, 1])], axis=2)
		# In = tf.constant([[[10.0,  11.0], [12.0 , 13.0], [14.0, 15.0]], [[16.0,  17.0], [18.0 , 19.0], [20.0, 21.0]]])
		# A = tf.constant([[1.0,  2.0], [ 3.0 , 4.0]])
		# B = 
		# C = tf.concat([In, B], axis=2)

        layer_1 = self.transfer_fct(tf.add(tf.tensordot(hz, weights['h1'], axes=((2,), (0,))), biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.tensordot(layer_1, weights['h2'], axes=((2,), (0,))), biases['b2'])) 
        hz_softmax = tf.nn.softmax(tf.add(tf.tensordot(layer_2, weights['out_mean'], axes=((2,), (0,))), biases['out_mean']), 2)
        return hz_softmax

    def _create_oc_svm(self):
        with tf.name_scope("oc-svm"):
            # self.input_of_rf = self.g_h_concatination
            # log_gamma = tf.get_variable(name='log_gamma', shape=[1],
            #                             initializer=tf.constant_initializer(
            #                                 np.log(self.gamma_init)),
            #                             trainable=self.train_gamma)  # (1,)
            # e = tf.get_variable(name="unit_noise", shape=[self.input_of_rf.get_shape().as_list()[1], self.num_random_features],
            #                     initializer=tf.random_normal_initializer(), trainable=False)
            # omega = tf.multiply(tf.exp(log_gamma), e, name='omega')  # (2*128, n_random_features) = e.shape
            # omega_x = tf.matmul(self.input_of_rf,
            #                     omega)  # (batch_size, 2*128) x (2*128, n_random_features) = (batch_size, n_random_features)
            # phi_x_tilde = tf.concat([tf.cos(omega_x), tf.sin(omega_x)], axis=1,
            #                         name='phi_x_tilde')  # (batch_size, 2*n_random_features)

            #phi_x_tilde = self.z
            phi_x_tilde = self.g_h_concatination

            self.w_rf = tf.Variable(tf.truncated_normal((phi_x_tilde.get_shape().as_list()[1], 1), stddev=0.05),
                                    name='w_rf')  # (2*n_random_features, 1)

            # self.predict = tf.nn.tanh(tf.matmul(phi_x_tilde, self.w_rf))  # (batch_size, 1)
            self.predict = tf.matmul(phi_x_tilde, self.w_rf)  # (batch_size, 1)
            self.l2_regularization = self.lamda_l2 * tf.reduce_sum(tf.square(self.w_rf))

    def _create_loss(self):

        with tf.name_scope("loss_vae"):
            self.loss_prior_0_1 =   - tf.reduce_sum(tf.square(self.prior_0_mean - self.prior_1_mean)) \
                                    - tf.reduce_sum(tf.square(tf.sqrt(tf.exp(self.prior_0_log_sigma_sq)) - tf.sqrt(tf.exp(self.prior_1_log_sigma_sq))))

            self.loss_z_0 =   tf.reduce_mean(tf.reduce_sum(tf.square(self.z_mean[:(self.batch_size//2),:] - self.prior_0_mean), 1) + \
                                    tf.reduce_sum(tf.square(tf.sqrt(tf.exp(self.z_log_sigma_sq[:(self.batch_size//2),:])) - \
                                        tf.sqrt(tf.exp(self.prior_0_log_sigma_sq))), 1))

            self.loss_z_1 =   tf.reduce_mean(tf.reduce_sum(tf.square(self.z_mean[-(self.batch_size//2):,:] - self.prior_1_mean), 1) + \
                                    tf.reduce_sum(tf.square(tf.sqrt(tf.exp(self.z_log_sigma_sq[-(self.batch_size//2):,:])) - \
                                        tf.sqrt(tf.exp(self.prior_1_log_sigma_sq))), 1))

            self.reconstr_loss = tf.reduce_sum(tf.multiply(-tf.log(self.h0NzSoftmax + 1e-10), tf.reshape(self.X_opcode, (self.batch_size, self.time_steps, self.vocab_opcode_size)))) / (self.batch_size * 1.0)
        #     self.cost = tf.reduce_mean(reconstr_loss + latent_loss)   # average over batch
        #     # Use ADAM optimizer
        #     self.optimizer =             tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        with tf.name_scope("loss_svm"):
            self.Y_svm = self.Y # self.Y í belong to 1 (vul), 0 (non-vul)
            self.Y_svm = tf.subtract(tf.cast(self.Y_svm, tf.float32), 0.5)
            self.Y_svm = tf.reshape(tf.sign(self.Y_svm), shape=(self.batch_size, 1)) # self.Y_svm is belong to 1 (vul), -1 (non-vul)

            self.loss_svm = tf.reduce_mean(
                tf.maximum(0.0, (1.0 + self.Y_svm)/2 - self.predict * self.Y_svm)) + self.l2_regularization
            # self.loss = self.loss_rnn + self.trade_off * self.loss_svm

        #self.loss = 0.6 * self.loss_svm + 0.15 * self.loss_prior_0_1 + 0.1 * self.loss_z_0 + 0.1 * self.loss_z_1 + 0.05 * self.reconstr_loss
        self.loss = self.loss_svm

    def _create_optimizer(self):
        with tf.name_scope("train"):
            parameters = tf.trainable_variables()  # get all trainable variables
            gradients = tf.gradients(self.loss, parameters)  # compute gradient on parameters
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)  # clip gradients

            if self.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            elif self.optimizer == 'grad':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            else:
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)

            # self.grads will have None gradient w_rf if set self.loss = self.loss_rnn and raise error when running feed_dict,
            # therefore, we need to create self.gradient_and_value to make sure not element in this list is None
            self.grads = optimizer.compute_gradients(self.loss)
            #self.training_op = optimizer.apply_gradients(zip(clipped_gradients, parameters), global_step=self.global_step)  # backpropagation
            self.training_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def _create_predict_rnn(self):
        self.correct = tf.nn.in_top_k(self.logits, self.Y, 1)
        self.accuracy_bi_rnn = tf.reduce_mean(tf.cast(self.correct, tf.float32))
        self.y_pred_rnn = tf.arg_max(self.logits, 1)

    def _create_predict_svm(self):
        self.y_pred_svm = tf.sign(self.predict)  # self.y_pred_svm (batch_size, 1) is belong to {-1,1}
        self.y_tmp = tf.add(tf.cast(self.y_pred_svm, tf.float32), 1.0)
        self.y_pred_svm = tf.reshape(tf.sign(self.y_tmp), shape=(self.batch_size,))

    def _create_summaries(self):
        with tf.name_scope("summaries"):
            # tf.summary.scalar("loss_rnn", self.loss_rnn)
            tf.summary.scalar("loss_svm", self.loss_svm)
            tf.summary.scalar("loss", self.loss)

            # tf.summary.histogram("histogram loss_rnn", self.loss_rnn)
            # tf.summary.histogram("histogram loss_svm", self.loss_svm)
            # tf.summary.histogram("histogram loss", self.loss)

        with tf.name_scope("visualize"):
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name + '/values', var)
            for grad, var in self.grads:
                if grad is not None:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
                    # self.gradient_and_value is run after self.training_op, therefore, var in this case
                    # is the value of variables which was updated (different from self.grads, its 'var' is not updated)
                    # self.gradient_and_value += [(grad, var)]
                    # self.list_str_variables[var.op.name] = grad.get_shape().as_list()

        self.summary_op = tf.summary.merge_all()

    def _create_logging_files(self):
        self.graph_path = os.path.join('graphs', self.data_size, self.datetime)
        self.checkpoint_path = os.path.join('saved-model', self.data_size, self.datetime)
        # self.save_results = 'results'
        utils.make_dir(self.checkpoint_path)
        # utils.make_dir(self.save_results)

    def build_model(self):
        self._create_placeholders()
        self._create_embedding()
        self._create_bi_rnn()
        self._create_vae()
        self._create_oc_svm()
        self._create_loss()
        self._create_optimizer()
        # self._create_predict_rnn()
        self._create_predict_svm()
        self._create_summaries()
        self._create_logging_files()

    def train(self, x_train_opcode, x_train_assembly, x_train_seq_len, y_train, x_valid_opcode, x_valid_assembly,
                x_valid_seq_len, y_valid, x_test_opcode, x_test_assembly, x_test_seq_len, y_test):
        outFile = open(self.OutName, 'w')    
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

        with tf.Session(config=utils.get_default_config()) as sess:
            writer = tf.summary.FileWriter(self.graph_path, sess.graph)

            check_point = tf.train.get_checkpoint_state(self.checkpoint_path)
            if check_point and tf.train.checkpoint_exists(check_point.model_checkpoint_path):
                message = "Load model parameters from %s\n" % check_point.model_checkpoint_path
                utils.print_and_write_logging_file(self.logging_path, message, self.running_mode)
                saver.restore(sess, check_point.model_checkpoint_path)
            else:
                message = "Create the model with fresh parameters\n"
                utils.print_and_write_logging_file(self.logging_path, message, self.running_mode)
                sess.run(tf.global_variables_initializer())

            ####Seperate dataset
            x_train_opcode_0 = []
            x_train_opcode_1 = []
            x_train_assembly_0 = []
            x_train_assembly_1 = []
            y_train_0 = []
            y_train_1 = []
            x_train_seq_len_0 = []
            x_train_seq_len_1 = []

            for index, aLabel in enumerate(y_train):
                 if (aLabel == 0.0):
                    x_train_opcode_0.append(x_train_opcode[index,:,:,:])
                    x_train_assembly_0.append(x_train_assembly[index,:,:,:])
                    y_train_0.append(y_train[index])
                    x_train_seq_len_0.append(x_train_seq_len[index])
                 else:
                    x_train_opcode_1.append(x_train_opcode[index,:,:,:])
                    x_train_assembly_1.append(x_train_assembly[index,:,:,:])
                    y_train_1.append(y_train[index])
                    x_train_seq_len_1.append(x_train_seq_len[index])

            x_train_opcode_0 = np.array(x_train_opcode_0)
            x_train_opcode_1 = np.array(x_train_opcode_1)
            x_train_assembly_0 = np.array(x_train_assembly_0)
            x_train_assembly_1 = np.array(x_train_assembly_1)

            min_train_0_1 = min(x_train_opcode_0.shape[0], x_train_opcode_1.shape[0])
            training_set = min_train_0_1 - min_train_0_1 % (self.batch_size // 2)
            training_batches = utils.make_batches(training_set, (self.batch_size // 2))
            ####Seperate dataset

            step_loss = 0.0  # average loss per epoch
            step_time = 0.0
            full_train_accuracy_score = []
            full_train_pre_score = []
            full_train_f1_score = []
            full_train_recall_score = []
            full_train_auc_score = []
            initial_step = self.global_step.eval()
            for step in range(initial_step, initial_step + self.num_train_steps):

                loss_per_batch = 0.0
                start_time = time.time()
                full_y_predic_train = np.array([])
                full_y_target_train = np.array([])
                for batch_idx, (batch_start, batch_end) in enumerate(training_batches):

                    ####Seperate batch
                    batch_x_opcode_0 = x_train_opcode_0[batch_start:batch_end]
                    batch_x_assembly_0 = x_train_assembly_0[batch_start:batch_end]
                    batch_y_0 = y_train_0[batch_start:batch_end]
                    batch_sequence_length_0 = x_train_seq_len_0[batch_start:batch_end]

                    batch_x_opcode_1 = x_train_opcode_1[batch_start:batch_end]
                    batch_x_assembly_1 = x_train_assembly_1[batch_start:batch_end]
                    batch_y_1 = y_train_1[batch_start:batch_end]
                    batch_sequence_length_1 = x_train_seq_len_1[batch_start:batch_end]

                    batch_x_opcode = np.concatenate((batch_x_opcode_0, batch_x_opcode_1), axis=0)
                    batch_x_assembly = np.concatenate((batch_x_assembly_0, batch_x_assembly_1), axis=0)
                    batch_y = batch_y_0 + batch_y_1
                    batch_sequence_length = batch_sequence_length_0 + batch_sequence_length_1
                    ####Seperate batch
                    full_y_target_train = np.append(full_y_target_train, batch_y)

                    feed_dict = {
                        self.X_opcode: batch_x_opcode,
                        self.X_assembly: batch_x_assembly,
                        self.Y: batch_y,
                        self.sequence_length: batch_sequence_length,
                    }
                    _, summary, batch_loss, batch_y_pred_train = sess.run(
                        [self.training_op, self.summary_op, self.loss, self.y_pred_svm],
                        feed_dict=feed_dict)
                    full_y_predic_train = np.append(full_y_predic_train, batch_y_pred_train)

                    if (batch_idx + 1) % (len(training_batches) // 10) == 0:
                        writer.add_summary(summary, global_step=step)

                    loss_per_batch += batch_loss / len(training_batches)

                batch_train_accuracy_score = mt.accuracy_score(y_true=full_y_target_train, y_pred=full_y_predic_train)
                batch_train_pre_score = mt.precision_score(y_true=full_y_target_train, y_pred=full_y_predic_train)
                batch_train_f1_score = mt.f1_score(y_true=full_y_target_train, y_pred=full_y_predic_train)
                batch_train_recall_score = mt.recall_score(y_true=full_y_target_train, y_pred=full_y_predic_train)
                batch_train_auc_score = mt.roc_auc_score(y_true=full_y_target_train, y_score=full_y_predic_train)
                full_y_predic_train = np.array([])
                full_y_target_train = np.array([])
                full_train_accuracy_score.append(batch_train_accuracy_score)
                full_train_pre_score.append(batch_train_pre_score)
                full_train_f1_score.append(batch_train_f1_score)
                full_train_recall_score.append(batch_train_recall_score)
                full_train_auc_score.append(batch_train_auc_score)

                step_time += (time.time() - start_time)
                step_loss += loss_per_batch

                # if (step + 1) % 10 == 0:
                #     # Save checkpoint and zero timer and loss.
                #     checkpoint_path = os.path.join(self.checkpoint_path, "rnn_classifier_" + self.data_size + ".ckpt")
                #     saver.save(sess, checkpoint_path, global_step=step)

                if (step + 1) % self.display_step == 0:

                	#Train plot
                    ave_train_accuracy_score = np.mean(full_train_accuracy_score)
                    ave_train_pre_score = np.mean(full_train_pre_score)
                    ave_train_f1_score = np.mean(full_train_f1_score)
                    ave_train_recall_score = np.mean(full_train_recall_score)
                    ave_train_auc_score = np.mean(full_train_auc_score)

                    full_train_accuracy_score = []
                    full_train_pre_score = []
                    full_train_f1_score = []
                    full_train_recall_score = []
                    full_train_auc_score = []
                    message = "global step %d/%d step-time %.2fs average loss %.5f acc %.2f pre %.2f f1 %.2f rec %.2f auc %.2f\n" % (
                        step, self.num_train_steps - 1, step_time, step_loss, ave_train_accuracy_score, ave_train_pre_score, ave_train_f1_score, ave_train_recall_score, ave_train_auc_score)
                    utils.print_and_write_logging_file(self.logging_path, message, self.running_mode)

                    outFile.write("%.2f\n" %(ave_train_accuracy_score * 100))
                    outFile.write("%.2f\n" %(ave_train_pre_score * 100))
                    outFile.write("%.2f\n" %(ave_train_f1_score * 100))
                    outFile.write("%.2f\n" %(ave_train_recall_score * 100))
                    outFile.write("%.2f\n" %(ave_train_auc_score * 100))
                    #Train plot

                    #Dev plot

                    step_time, step_loss = 0.0, 0.0

                    dev_set = x_valid_opcode.shape[0] - x_valid_opcode.shape[0] % self.batch_size
                    dev_batches = utils.make_batches(dev_set, self.batch_size)
                    ####Seperate dataset

                    average_dev_loss = 0.0
                    full_y_pred_svm = np.array([])
                    for batch_idx, (batch_start, batch_end) in enumerate(dev_batches):

                        valid_x_opcode = x_valid_opcode[batch_start:batch_end]
                        valid_x_assembly = x_valid_assembly[batch_start:batch_end]

                        valid_y = y_valid[batch_start:batch_end]
                        valid_seq_len = x_valid_seq_len[batch_start:batch_end]
                        ####Seperate batch

                        feed_dict = {
                            self.X_opcode: valid_x_opcode,
                            self.X_assembly: valid_x_assembly,
                            self.Y: valid_y,
                            self.sequence_length: valid_seq_len,
                        }

                        batch_dev_loss, batch_y_pred = sess.run([self.loss, self.y_pred_svm], feed_dict=feed_dict)
                        full_y_pred_svm = np.append(full_y_pred_svm, batch_y_pred)

                        average_dev_loss += batch_dev_loss / len(dev_batches)
                    message = "eval: accuracy_svm %.2f\n" % (
                                mt.accuracy_score(y_true=y_valid[:dev_set], y_pred=full_y_pred_svm) * 100)
                    message += "eval: precision_svm %.2f\n" % (
                                mt.precision_score(y_true=y_valid[:dev_set], y_pred=full_y_pred_svm) * 100)
                    message += "eval: f1_svm %.2f\n" % (
                                mt.f1_score(y_true=y_valid[:dev_set], y_pred=full_y_pred_svm) * 100)
                    message += "eval: recall_svm %.2f\n" % (
                                mt.recall_score(y_true=y_valid[:dev_set], y_pred=full_y_pred_svm) * 100)
                    message += "eval: roc_auc_svm %.2f\n" % (
                                mt.roc_auc_score(y_true=y_valid[:dev_set], y_score=full_y_pred_svm) * 100)
                    message += "-----------------------------------------------------\n"
                    outFile.write("%.2f\n" %(mt.accuracy_score(y_true=y_valid[:dev_set], y_pred=full_y_pred_svm) * 100))
                    outFile.write("%.2f\n" %(mt.precision_score(y_true=y_valid[:dev_set], y_pred=full_y_pred_svm) * 100))
                    outFile.write("%.2f\n" %(mt.f1_score(y_true=y_valid[:dev_set], y_pred=full_y_pred_svm) * 100))
                    outFile.write("%.2f\n" %(mt.recall_score(y_true=y_valid[:dev_set], y_pred=full_y_pred_svm) * 100))
                    outFile.write("%.2f\n" %(mt.roc_auc_score(y_true=y_valid[:dev_set], y_score=full_y_pred_svm) * 100))
                    utils.print_and_write_logging_file(self.logging_path, message, self.running_mode)
                    #Dev plot

                    #Test plot
                    #x_train_opcode, x_train_assembly, x_train_seq_len, y_train, 
                    #x_valid_opcode, x_valid_assembly, x_valid_seq_len, y_valid, 
                    #x_test_opcode, x_test_assembly, x_test_seq_len, y_test
                    step_time, step_loss = 0.0, 0.0

                    test_set = x_test_opcode.shape[0] - x_test_opcode.shape[0] % self.batch_size
                    test_batches = utils.make_batches(test_set, self.batch_size)
                    ####Seperate dataset

                    average_test_loss = 0.0
                    full_y_pred_svm_test = np.array([])
                    for batch_idx, (batch_start, batch_end) in enumerate(test_batches):

                        test_x_opcode = x_test_opcode[batch_start:batch_end]
                        test_x_assembly = x_test_assembly[batch_start:batch_end]

                        test_y = y_test[batch_start:batch_end]
                        test_seq_len = x_test_seq_len[batch_start:batch_end]
                        ####Seperate batch

                        feed_dict = {
                            self.X_opcode: test_x_opcode,
                            self.X_assembly: test_x_assembly,
                            self.Y: test_y,
                            self.sequence_length: test_seq_len,
                        }

                        batch_test_loss, batch_y_pred_test = sess.run([self.loss, self.y_pred_svm], feed_dict=feed_dict)
                        full_y_pred_svm_test = np.append(full_y_pred_svm_test, batch_y_pred_test)

                        average_test_loss += batch_test_loss / len(test_batches)

                    message = "test: accuracy_svm %.2f\n" % (
                                mt.accuracy_score(y_true=y_test[:test_set], y_pred=full_y_pred_svm_test) * 100)
                    message += "test: precision_svm %.2f\n" % (
                                mt.precision_score(y_true=y_test[:test_set], y_pred=full_y_pred_svm_test) * 100)
                    message += "test: f1_svm %.2f\n" % (
                                mt.f1_score(y_true=y_test[:test_set], y_pred=full_y_pred_svm_test) * 100)
                    message += "test: recall_svm %.2f\n" % (
                                mt.recall_score(y_true=y_test[:test_set], y_pred=full_y_pred_svm_test) * 100)
                    message += "test: roc_auc_svm %.2f\n" % (
                                mt.roc_auc_score(y_true=y_test[:test_set], y_score=full_y_pred_svm_test) * 100)
                    message += "-----------------------------------------------------\n"
                    outFile.write("%.2f\n" %(mt.accuracy_score(y_true=y_test[:test_set], y_pred=full_y_pred_svm_test) * 100))
                    outFile.write("%.2f\n" %(mt.precision_score(y_true=y_test[:test_set], y_pred=full_y_pred_svm_test) * 100))
                    outFile.write("%.2f\n" %(mt.f1_score(y_true=y_test[:test_set], y_pred=full_y_pred_svm_test) * 100))
                    outFile.write("%.2f\n" %(mt.recall_score(y_true=y_test[:test_set], y_pred=full_y_pred_svm_test) * 100))
                    outFile.write("%.2f\n" %(mt.roc_auc_score(y_true=y_test[:test_set], y_score=full_y_pred_svm_test) * 100))
                    utils.print_and_write_logging_file(self.logging_path, message, self.running_mode)
                    #Test plot                    
            writer.close()
        message = "Finish training process.\n"
        utils.print_and_write_logging_file(self.logging_path, message, self.running_mode)
        outFile.close()

    def visualization(self, x_test_opcode, x_test_assembly, x_test_seq_len, y_test):
        with tf.Session(config=utils.get_default_config()) as sess:
            self.checkpoint_path = 'saved-model/full_dataset/good_result/2018-5-8-19-36-50'
            saver = tf.train.Saver(tf.global_variables())

            check_point = tf.train.get_checkpoint_state(self.checkpoint_path)
            if check_point and tf.train.checkpoint_exists(check_point.model_checkpoint_path):
                message = "Load model parameters from %s\n" % check_point.model_checkpoint_path
                utils.print_and_write_logging_file(self.logging_path, message, self.running_mode, self.datetime)
                saver.restore(sess, check_point.model_checkpoint_path)
            else:
                raise Exception('Saved model not found.')
            # model_path = 'saved-model/full_dataset/good_result/2018-5-8-19-36-50'

            testing_set = x_test_opcode.shape[0] - x_test_opcode.shape[0] % self.batch_size
            batch_x_opcode = []
            batch_x_assembly = []
            batch_y = []
            batch_sequence_length = []

            for i in range(3498, testing_set):
                # if y_test[i] == 0:
                if y_test[i] == 1:
                    batch_x_opcode.append(x_test_opcode[i])
                    batch_x_assembly.append(x_test_assembly[i])
                    batch_sequence_length.append(x_test_seq_len[i])
                    batch_y.append(y_test[i])
                if len(batch_y) == self.batch_size:
                    break

            feed_dict = {
                self.X_opcode: batch_x_opcode,
                self.X_assembly: batch_x_assembly,
                self.Y: batch_y,
                self.sequence_length: batch_sequence_length,
            }

            image = sess.run(
                [self.cnn_input],
                feed_dict=feed_dict)

            layers = ["r", "p", "c"]
            path_log = os.path.join('visualize', '_epoch1000000_good_log_1_cap_tuong_tu')
            path_output = os.path.join('visualize', '_epoch1000000_good_output_1_cap_tuong_tu')

            # activation_visualization(sess_graph_path=sess, value_feed_dict=feed_dict, layers=layers, path_logdir=path_log,
            #                              path_outdir=path_output)
            # deconv_visualization(sess_graph_path=sess, value_feed_dict=feed_dict, input_tensor=self.cnn_input, layers=layers, path_logdir=path_log,
            #                          path_outdir=path_output)
            # img_normalize = image_normalization(image[0][0])
            # imsave(os.path.join('visualize', '_epoch100_good_image_1_cap_tuong_tu.png'), np.reshape(img_normalize, [img_normalize.shape[0], img_normalize.shape[1]]))

            layer = 'cnn/relu3/Relu'
            deepdream_visualization(sess_graph_path=sess,input_tensor=self.two_dimension_image, value_feed_dict=feed_dict, layer=layer, classes=[1, 2, 3, 4, 5], path_logdir=path_log,
                                         path_outdir=path_output)

            print('Ok, I got it.')	

    def test_and_compute_score(self, x_test_opcode, x_test_assembly, x_test_seq_len, y_test):
        with tf.Session(config=utils.get_default_config()) as sess:
            saver = tf.train.Saver(tf.global_variables())

            check_point = tf.train.get_checkpoint_state(self.checkpoint_path)
            if check_point and tf.train.checkpoint_exists(check_point.model_checkpoint_path):
                message = "Load model parameters from %s\n" % check_point.model_checkpoint_path
                utils.print_and_write_logging_file(self.logging_path, message, self.running_mode, self.datetime)
                saver.restore(sess, check_point.model_checkpoint_path)
            else:
                raise Exception('Saved model not found.')

            testing_set = x_test_opcode.shape[0] - x_test_opcode.shape[0] % self.batch_size
            testing_batches = utils.make_batches(testing_set, self.batch_size)

            average_test_loss = 0.0
            average_accuracy_rnn = 0.0
            full_y_pred = np.array([])
            for batch_idx, (batch_start, batch_end) in enumerate(testing_batches):
                batch_x_opcode = x_test_opcode[batch_start:batch_end]
                batch_x_assembly = x_test_assembly[batch_start:batch_end]

                batch_y = y_test[batch_start:batch_end]
                batch_sequence_length = x_test_seq_len[batch_start:batch_end]

                feed_dict = {
                    self.X_opcode: batch_x_opcode,
                    self.X_assembly: batch_x_assembly,
                    self.Y: batch_y,
                    self.sequence_length: batch_sequence_length,
                }

                # batch_test_loss, accuracy_rnn = sess.run(
                #     [self.loss, self.accuracy_bi_rnn],
                #     feed_dict=feed_dict)

                batch_test_loss = sess.run(self.loss, feed_dict=feed_dict)

                batch_y_pred = sess.run(self.y_pred_svm, feed_dict=feed_dict)
                full_y_pred = np.append(full_y_pred, batch_y_pred)

                average_test_loss += batch_test_loss / len(testing_batches)
                # average_accuracy_rnn += accuracy_rnn / len(testing_batches)

            full_accuracy_score = mt.accuracy_score(y_true=y_test[:testing_set], y_pred=full_y_pred)
            full_pre_score = mt.precision_score(y_true=y_test[:testing_set], y_pred=full_y_pred)
            full_f1_score = mt.f1_score(y_true=y_test[:testing_set], y_pred=full_y_pred)
            full_recall_score = mt.recall_score(y_true=y_test[:testing_set], y_pred=full_y_pred)
            full_auc_score = mt.roc_auc_score(y_true=y_test[:testing_set], y_score=full_y_pred)

            message = "testing loss %.5f\n" % average_test_loss
            message += "accuracy %.2f\n" % (full_accuracy_score * 100)
            message += "compute score:\n"
            message += '\tprecision score %.5f\n' % (full_pre_score * 100)
            message += '\tf1 score %.5f\n' % (full_f1_score * 100)
            message += '\trecall score %.5f\n' % (full_recall_score * 100)
            message += '\tAUC score %.5f\n' % (full_auc_score * 100)
            message += "-----------------------------------------------------\n"
            message += "Finish computing score process.\n"
            utils.print_and_write_logging_file(self.logging_path, message, self.running_mode, self.datetime)


def main():
    model = RNNClassifier(sys.argv[1])
    if model.running_mode == 1:  # train
        if sys.argv[1] == 'All':
            list_arch_os = ['32-windows', '64-windows', '32-ubuntu', '64-ubuntu']
        elif sys.argv[1] == 'Window':
            list_arch_os = ['32-windows', '64-windows']
        else:
            list_arch_os = ['32-ubuntu', '64-ubuntu']

        print(list_arch_os)

        x_train_opcode, x_train_assembly, x_train_seq_len, y_train, \
        x_valid_opcode, x_valid_assembly, x_valid_seq_len, y_valid, \
        x_test_opcode, x_test_assembly, x_test_seq_len, y_test, model.time_steps, model.vocab_opcode_size = utils.load_vul_deepacker(
            list_arch_os)

        model.build_model()
        model.train(x_train_opcode, x_train_assembly, x_train_seq_len, y_train,
                    x_valid_opcode, x_valid_assembly, x_valid_seq_len, y_valid,
                    x_test_opcode, x_test_assembly, x_test_seq_len, y_test)
    elif model.running_mode == 0:  # test
        if model.data_size == 'full_dataset':
            list_arch_os = ['32-windows', '64-windows', '32-ubuntu', '64-ubuntu']
        else:
            list_arch_os = ['32-windows']

        _, _, _, _, \
        _, _, _, _, \
        x_test_opcode, x_test_assembly, x_test_seq_len, y_test, model.time_steps, model.vocab_opcode_size = \
            utils.load_vul_deepacker(list_arch_os)
        model.build_model()
        model.test_and_compute_score(x_test_opcode, x_test_assembly, x_test_seq_len, y_test)
    else:  # visualization
        if model.data_size == 'full_dataset':
            list_arch_os = ['32-windows', '64-windows', '32-ubuntu', '64-ubuntu']
        else:
            list_arch_os = ['32-windows']

        # _, _, _, _, \
        # _, _, _, _, \
        # x_test_opcode, x_test_assembly, x_test_seq_len, y_test, model.time_steps, model.vocab_opcode_size = \
        #     utils.load_vul_deepacker(list_arch_os)

        x_train_opcode, x_train_assembly, x_train_seq_len, y_train, \
        x_valid_opcode, x_valid_assembly, x_valid_seq_len, y_valid, \
        _, _, _, _, model.time_steps, model.vocab_opcode_size = utils.load_vul_deepacker(
            list_arch_os, shuffle_data=False)

        model.build_model()
        model.visualization(x_train_opcode, x_train_assembly, x_train_seq_len, y_train)

if __name__ == '__main__':
    main()
