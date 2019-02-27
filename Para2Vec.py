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

class RNNClassifier:
    def __init__(self, iSize, iFeatureSize):
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
        self.display_step = 1
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.learning_rate = 0.001

        self.iSize = iSize
        self.FeatureSize = int(iFeatureSize)
        self.OutName = 'txtLog/lr1e3_Doc2Vec' + self.iSize + iFeatureSize + '.txt'

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
            self.X_opcode = tf.placeholder(tf.float32, [None, self.FeatureSize], name='x_opcode_input')
            self.Y = tf.placeholder(tf.int32, [None], name='true_label')

 

    def _create_oc_svm(self):
        with tf.name_scope("oc-svm"):
            self.w_rf = tf.Variable(tf.truncated_normal((self.FeatureSize, 1), stddev=0.05),
                                    name='w_rf', trainable=True)  # (2*n_random_features, 1)

            # self.predict = tf.nn.tanh(tf.matmul(self.X_opcode, self.w_rf))  # (batch_size, 1)
            self.predict = tf.matmul(self.X_opcode, self.w_rf)  # (batch_size, 1)
            self.l2_regularization = self.lamda_l2 * tf.reduce_sum(tf.square(self.w_rf))

    def _create_loss(self):
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
            self.training_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

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

        self.summary_op = tf.summary.merge_all()

    def _create_logging_files(self):
        self.graph_path = os.path.join('graphs', self.data_size, self.datetime)
        self.checkpoint_path = os.path.join('saved-model', self.data_size, self.datetime)
        # self.save_results = 'results'
        utils.make_dir(self.checkpoint_path)
        # utils.make_dir(self.save_results)

    def build_model(self):
        self._create_placeholders()
        self._create_oc_svm()
        self._create_loss()
        self._create_optimizer()
        self._create_predict_svm()
        self._create_summaries()
        self._create_logging_files()

    def train(self, x_train_opcode, y_train, x_valid_opcode, y_valid, x_test_opcode, y_test ):
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
            y_train_0 = []
            y_train_1 = []

            for index, aLabel in enumerate(y_train):
                 if (aLabel == 0.0):
                    x_train_opcode_0.append(x_train_opcode[index,:])
                    y_train_0.append(y_train[index])
                 else:
                    x_train_opcode_1.append(x_train_opcode[index,:])
                    y_train_1.append(y_train[index])

            x_train_opcode_0 = np.array(x_train_opcode_0)
            x_train_opcode_1 = np.array(x_train_opcode_1)
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
                    batch_y_0 = y_train_0[batch_start:batch_end]

                    batch_x_opcode_1 = x_train_opcode_1[batch_start:batch_end]
                    batch_y_1 = y_train_1[batch_start:batch_end]

                    batch_x_opcode = np.concatenate((batch_x_opcode_0, batch_x_opcode_1), axis=0)
                    batch_y = batch_y_0 + batch_y_1
                    ####Seperate batch
                    full_y_target_train = np.append(full_y_target_train, batch_y)

                    feed_dict = {
                        self.X_opcode: batch_x_opcode,
                        self.Y: batch_y,
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

                        valid_y = y_valid[batch_start:batch_end]
                        ####Seperate batch

                        feed_dict = {
                            self.X_opcode: valid_x_opcode,
                            self.Y: valid_y,
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
                    #x_train_opcode, y_train, x_valid_opcode, y_valid, x_test_opcode, y_test 
                    step_time, step_loss = 0.0, 0.0

                    test_set = x_test_opcode.shape[0] - x_test_opcode.shape[0] % self.batch_size
                    test_batches = utils.make_batches(test_set, self.batch_size)
                    ####Seperate dataset

                    average_test_loss = 0.0
                    full_y_pred_svm_test = np.array([])
                    for batch_idx, (batch_start, batch_end) in enumerate(test_batches):

                        test_x_opcode = x_test_opcode[batch_start:batch_end]
                        test_y = y_test[batch_start:batch_end]
                        ####Seperate batch

                        feed_dict = {
                            self.X_opcode: test_x_opcode,
                            self.Y: test_y,
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

def main():
    model = RNNClassifier(sys.argv[1], 256)
    if sys.argv[1] == 'All':
    	list_arch_os = ['32-windows', '64-windows', '32-ubuntu', '64-ubuntu']
    elif sys.argv[1] == 'Window':
        list_arch_os = ['32-windows', '64-windows']
    elif sys.argv[1] == 'Ubuntu':
        list_arch_os = ['32-ubuntu', '64-ubuntu']
    print(list_arch_os)

    x_train_opcode, y_train, x_valid_opcode, y_valid, x_test_opcode, y_test = utils.load_doc_vec(list_arch_os, 256)

    model.build_model()
    model.train(x_train_opcode, y_train, x_valid_opcode, y_valid, x_test_opcode, y_test)

if __name__ == '__main__':
    main()
