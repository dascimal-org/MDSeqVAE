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

import numpy as np
import os
import pandas as pd
from collections import Counter
import tensorflow as tf
import datetime
import inspect
import json
import sys

_NOW = datetime.datetime.now()
_DATETIME = str(_NOW.year) + '-' + str(_NOW.day) + '-' + str(_NOW.month) + '-' + str(_NOW.hour) + '-' + str(_NOW.minute) + '-' + str(_NOW.second)
_LOG = 'log'
_RANDOM_SEED = 6789

def load_doc_vec(list_arch_os, FeatureSize, shuffle_data=True, randseed='default'):
    # list_arch_os = ['32-windows', '64-windows', '32-ubuntu', '64-ubuntu']
    # list_arch_os = ['32-windows']

    X_opcode = []
    Y_full = np.array([])
    for arch_os in list_arch_os:
        decimal_functions_path = 'dataset/doc2vec' + FeatureSize +'_vocab_opcode-' + arch_os + '.data'
        print(decimal_functions_path)
        label_path = 'dataset/labels-' + arch_os + '.data'
        with open(decimal_functions_path, 'r') as f:
            X_lines = f.readlines()
        with open(label_path, 'r') as f:
            Y_lines = f.readlines()

        # X = [np.array([int(number) for number in line.split()]) for line in X_lines if len(line) != 0]
        Y = np.array([int(number) for number in Y_lines[0].split()])
        X_lines = X_lines[1:]
        for item in X_lines:
            subList = item.split(",")
            subList = [float(x) for x in subList]
            X_opcode.append(subList)
        Y_full = np.concatenate((Y_full, Y), axis=0)

    # X_opcode, X_assembly, sequence_length, max_length, vocab_opcode_size = process_opcode_assembly_code(X_full)

    X_opcode = np.array(X_opcode)
    # print(Y_full.shape)
    # #(8991,)
    # print(X_opcode.shape)
    # print(X_opcode[:3])
    # print(X_opcode[-3:])
    #(8991, 158, 1, 64)
    # print(X_assembly.shape)
    #(8991, 158, 1, 64)

    # shuffle X and Y
    if randseed == 'default':
        np.random.seed(_RANDOM_SEED)
    elif isinstance(randseed, int):
        np.random.seed(randseed)
    if shuffle_data:
        tuple_X_Y_seq = list(zip(X_opcode, Y_full))
        np.random.shuffle(tuple_X_Y_seq)
        X_opcode, Y_full = zip(*tuple_X_Y_seq)

    X_opcode = np.asarray(X_opcode)
    Y_full = np.array(Y_full)

    n_training_examples = int(round(len(X_opcode) * 0.8, 0))
    n_valid_examples = int(round(len(X_opcode) * 0.1, 0))

    x_train_opcode = X_opcode[0:n_training_examples, :]
    y_train = [Y_full[i] for i in range(n_training_examples)]

    end_id_valid_set = n_training_examples + n_valid_examples
    x_valid_opcode = X_opcode[n_training_examples:end_id_valid_set, :]
    y_valid = [Y_full[i] for i in range(n_training_examples, end_id_valid_set)]

    end_id_test_set = X_opcode.shape[0]
    x_test_opcode =  X_opcode[end_id_valid_set:, :]
    y_test = [Y_full[i] for i in range(end_id_valid_set, end_id_test_set)]

    # x_train, x_train_sequence_length = padding_zero_and_get_sequence_length(x_train, max_length)
    # x_test, x_test_sequence_length = padding_zero_and_get_sequence_length(x_test, max_length)

    return x_train_opcode, y_train, x_valid_opcode, y_valid, x_test_opcode, y_test

def load_vul_deepacker(list_arch_os, shuffle_data=True, randseed='default'):
    # list_arch_os = ['32-windows', '64-windows', '32-ubuntu', '64-ubuntu']
    # list_arch_os = ['32-windows']

    X_full = []
    Y_full = np.array([])
    for arch_os in list_arch_os:
        decimal_functions_path = 'dataset/vocab_opcode-' + arch_os + '.data'
        label_path = 'dataset/labels-' + arch_os + '.data'
        with open(decimal_functions_path, 'r') as f:
            X_lines = f.readlines()
        with open(label_path, 'r') as f:
            Y_lines = f.readlines()

        # X = [np.array([int(number) for number in line.split()]) for line in X_lines if len(line) != 0]
        Y = np.array([int(number) for number in Y_lines[0].split()])
        X_full += X_lines
        Y_full = np.concatenate((Y_full, Y), axis=0)

    X_opcode, X_assembly, sequence_length, max_length, vocab_opcode_size = process_opcode_assembly_code(X_full)

    # shuffle X and Y
    if randseed == 'default':
        np.random.seed(_RANDOM_SEED)
    elif isinstance(randseed, int):
        np.random.seed(randseed)
    if shuffle_data:
        tuple_X_Y_seq = list(zip(X_opcode, X_assembly, sequence_length, Y_full))
        np.random.shuffle(tuple_X_Y_seq)
        X_opcode, X_assembly, sequence_length, Y_full = zip(*tuple_X_Y_seq)

    X_opcode = np.asarray(X_opcode)
    X_assembly= np.asarray(X_assembly)

    n_training_examples = int(round(len(X_opcode) * 0.8, 0))
    n_valid_examples = int(round(len(X_opcode) * 0.1, 0))

    x_train_opcode = X_opcode[0:n_training_examples, :, :, :]
    x_train_assembly = X_assembly[0:n_training_examples, :, :, :]
    x_train_seq_len = [sequence_length[i] for i in range(n_training_examples)]
    y_train = [Y_full[i] for i in range(n_training_examples)]

    end_id_valid_set = n_training_examples + n_valid_examples
    x_valid_opcode = X_opcode[n_training_examples:end_id_valid_set, :, :, :]
    x_valid_assembly = X_assembly[n_training_examples:end_id_valid_set, :, :, :]
    x_valid_seq_len = [sequence_length[i] for i in range(n_training_examples, end_id_valid_set)]
    y_valid = [Y_full[i] for i in range(n_training_examples, end_id_valid_set)]

    end_id_test_set = X_opcode.shape[0]
    x_test_opcode =  X_opcode[end_id_valid_set:, :, :, :]
    x_test_assembly = X_assembly[end_id_valid_set:, :, :, :]
    x_test_seq_len = [sequence_length[i] for i in range(end_id_valid_set, end_id_test_set)]
    y_test = [Y_full[i] for i in range(end_id_valid_set, end_id_test_set)]

    # x_train, x_train_sequence_length = padding_zero_and_get_sequence_length(x_train, max_length)
    # x_test, x_test_sequence_length = padding_zero_and_get_sequence_length(x_test, max_length)

    return x_train_opcode, x_train_assembly, x_train_seq_len, y_train, \
    x_valid_opcode, x_valid_assembly, x_valid_seq_len, y_valid, \
    x_test_opcode, x_test_assembly, x_test_seq_len, y_test, max_length, vocab_opcode_size

def load_vul_deepackerFixedLength(list_arch_os, ourLength, shuffle_data=True, randseed='default'):
    # list_arch_os = ['32-windows', '64-windows', '32-ubuntu', '64-ubuntu']
    # list_arch_os = ['32-windows']

    X_full = []
    Y_full = np.array([])
    for arch_os in list_arch_os:
        decimal_functions_path = 'dataset/vocab_opcode-' + arch_os + '.data'
        label_path = 'dataset/labels-' + arch_os + '.data'
        with open(decimal_functions_path, 'r') as f:
            X_lines = f.readlines()
        with open(label_path, 'r') as f:
            Y_lines = f.readlines()

        # X = [np.array([int(number) for number in line.split()]) for line in X_lines if len(line) != 0]
        Y = np.array([int(number) for number in Y_lines[0].split()])
        X_full += X_lines
        Y_full = np.concatenate((Y_full, Y), axis=0)

    X_opcode, X_assembly, sequence_length, max_length, vocab_opcode_size = process_opcode_assembly_codeFixedLength(X_full, ourLength)

    # shuffle X and Y
    if randseed == 'default':
        np.random.seed(_RANDOM_SEED)
    elif isinstance(randseed, int):
        np.random.seed(randseed)
    if shuffle_data:
        tuple_X_Y_seq = list(zip(X_opcode, X_assembly, sequence_length, Y_full))
        np.random.shuffle(tuple_X_Y_seq)
        X_opcode, X_assembly, sequence_length, Y_full = zip(*tuple_X_Y_seq)

    X_opcode = np.asarray(X_opcode)
    X_assembly= np.asarray(X_assembly)

    n_training_examples = int(round(len(X_opcode) * 0.8, 0))
    n_valid_examples = int(round(len(X_opcode) * 0.1, 0))

    x_train_opcode = X_opcode[0:n_training_examples, :, :, :]
    x_train_assembly = X_assembly[0:n_training_examples, :, :, :]
    x_train_seq_len = [sequence_length[i] for i in range(n_training_examples)]
    y_train = [Y_full[i] for i in range(n_training_examples)]

    end_id_valid_set = n_training_examples + n_valid_examples
    x_valid_opcode = X_opcode[n_training_examples:end_id_valid_set, :, :, :]
    x_valid_assembly = X_assembly[n_training_examples:end_id_valid_set, :, :, :]
    x_valid_seq_len = [sequence_length[i] for i in range(n_training_examples, end_id_valid_set)]
    y_valid = [Y_full[i] for i in range(n_training_examples, end_id_valid_set)]

    end_id_test_set = X_opcode.shape[0]
    x_test_opcode =  X_opcode[end_id_valid_set:, :, :, :]
    x_test_assembly = X_assembly[end_id_valid_set:, :, :, :]
    x_test_seq_len = [sequence_length[i] for i in range(end_id_valid_set, end_id_test_set)]
    y_test = [Y_full[i] for i in range(end_id_valid_set, end_id_test_set)]

    # x_train, x_train_sequence_length = padding_zero_and_get_sequence_length(x_train, max_length)
    # x_test, x_test_sequence_length = padding_zero_and_get_sequence_length(x_test, max_length)

    return x_train_opcode, x_train_assembly, x_train_seq_len, y_train, \
    x_valid_opcode, x_valid_assembly, x_valid_seq_len, y_valid, \
    x_test_opcode, x_test_assembly, x_test_seq_len, y_test, max_length, vocab_opcode_size

def normalize_unequal_data_feature_lengths(A):
    B = []
    C = []
    sum = 0.
    count = 0
    for_computing_variance = 0
    for a in A:
        sum += np.sum(a)
        count += len(a)
    mean = sum / count

    for a in A:
        a = a - mean
        B.append(a)

    for b in B:
        for_computing_variance += np.sum(b ** 2)

    variance = np.sqrt((1. / count) * for_computing_variance)

    for b in B:
        b = b / variance
        C.append(b)
    return C


def padding_zero_and_get_sequence_length(list_A, max_length):
    padding_A = np.zeros((len(list_A), max_length))
    sequence_length = np.array([])
    for i in range(len(list_A)):
        padding_A[i][:len(list_A[i])] = list_A[i]
        sequence_length = np.append(sequence_length, len(list_A[i]))
    return padding_A, sequence_length


def get_max_length(list_A):
    return max([max(len(list_A[i]) for i in range(len(list_A)))])


def make_batches(size, batch_size):
    '''Returns a list of batch indices (tuples of indices).
    '''
    return [(i, min(size, i + batch_size)) for i in range(0, size, batch_size)]


def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# def show_params(self, save=False, start_time=0):
#     f_datetime = str(self.datetime)
#     f_dataset = "full MNIST"
#     f_num_epoches = str(self.num_epochs)
#     f_batch_size = str(self.batch_size)
#     f_learning_rate = str(self.learning_rate)
#     f_trade_off = str(self.trade_off)
#     f_num_random_features = str(self.num_random_features)
#     f_gamma_init = str(self.gamma_init)
#     f_lambda = str(self.lbd)
#     f_optimizer = "SGD"
#
#     print(
#         "Parameter list: \n"
#         "{"
#         "\n\tdataset=\"%s\","
#         "\n\tnum_epoches=%s,"
#         "\n\tbatch_size=%s,"
#         "\n\tlearning_rate=%s,"
#         "\n\tf_trade_off=%s,"
#         "\n\tnum_random_features=%s,"
#         "\n\tgamma_init=%s,"
#         "\n\tlambda=%s, "
#         "\n\toptimizer=\"%s\""
#         "\n}" %
#         (f_dataset, f_num_epoches, f_batch_size, f_learning_rate, f_trade_off, f_num_random_features,
#          f_gamma_init, f_lambda, f_optimizer)
#     )
#
#     if not save:
#         with open('./results/params.csv', 'a') as fo:
#             fo.write(
#                 'datetime, dataset, num_epoches, batch_size, learning_rate, num_random_features, gamma_init, lambda, optimizer, time[s]\n')
#
#             f_time = str(time.time() - start_time)
#             print('Total time: {0} seconds'.format(f_time))
#             fo.write("%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n\n" % (
#                 f_datetime, f_dataset, f_num_epoches, f_batch_size, f_learning_rate, f_trade_off,
#                 f_num_random_features, f_gamma_init, f_lambda, f_optimizer, f_time))


def write_value_and_gradient_to_file(gradient_and_value, tensor_name_and_shape, epoch, batch_start, batch_end):
    # no need to remove older file because pd.DataFrame will overwrite the existing files
    for index, (tensor_name, shape) in enumerate(tensor_name_and_shape.items()):
        if len(shape) == 2:
            shape_string = '(' + str(shape[0]) + ',' + str(shape[1]) + ')'
            path_save_tensors = os.path.join('gradient_and_value', tensor_name.replace('/', '~') + shape_string)

            make_dir(path_save_tensors)

            filename_gradient = str(epoch) + "(" + str(batch_start) + "," + str(batch_end) + ")" + '-gradient' + ".csv"
            filename_value = str(epoch) + "(" + str(batch_start) + "," + str(batch_end) + ")" + '-value' + ".csv"

            df = pd.DataFrame(gradient_and_value[index][0])
            df.to_csv(os.path.join(path_save_tensors, filename_gradient))

            df = pd.DataFrame(gradient_and_value[index][1])
            df.to_csv(os.path.join(path_save_tensors, filename_value))


def build_vocab(words):
    """ Build vocabulary of VOCAB_SIZE most frequent words """
    dictionary = dict()
    count = []
    count.extend(Counter(words).most_common())
    index = 0
    make_dir('vocab_opcode')
    with open('vocab_opcode/opcode_n_occurs.tsv', "w") as f:
        for word, occurs in count:
            dictionary[word] = index
            f.write(word + '\tn_occurs:\t' + str(occurs) + "\n")
            index += 1
    index_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, index_dictionary


def create_one_hot_vector_for_opcode(aa, dic_id_opcode, all_zeros=False):
    bb = np.zeros((1, len(dic_id_opcode)))
    if all_zeros:
        return bb
    else:
        bb[0][dic_id_opcode[aa]] = 1
        return bb


def create_one_hot_vector_for_assembly(list_tuple=[], all_zeros=False):
    bb = np.zeros((1, 256))
    if all_zeros: # for padding
        return bb
    else:
        # count on each line of function, and assign at index the value of num_occurs
        for tuple_hex_times in list_tuple:
            decimal = int(tuple_hex_times[0])
            n_occures = tuple_hex_times[1]
            bb[0][decimal] = n_occures
        return bb


def convert_to_one_hot(list_function_opcode, list_function_assembly_code, dic_opcode, max_length):

    # process opcode
    function_opcode_one_hot = []
    for function_opcode in list_function_opcode:

        opcode_one_hot = []
        for opcode in function_opcode:
            one_hex = create_one_hot_vector_for_opcode(opcode, dic_opcode)
            opcode_one_hot.append(one_hex)

        while len(opcode_one_hot) < max_length:
            opcode_one_hot.append(create_one_hot_vector_for_opcode(opcode, dic_opcode, all_zeros=True))

        function_opcode_one_hot.append(opcode_one_hot)

    function_opcode_one_hot = np.asarray(function_opcode_one_hot)

    # process one-hot
    function_assembly_one_hot = []
    for function_assembly in list_function_assembly_code:

        assembly_one_hot = []
        list_tuple = []
        for list_hex in function_assembly:
            list_tuple.extend(Counter(list_hex).most_common())
            one_line = create_one_hot_vector_for_assembly(list_tuple)
            assembly_one_hot.append(one_line)
            list_tuple = []

        while len(assembly_one_hot) < max_length:
            assembly_one_hot.append(create_one_hot_vector_for_assembly(all_zeros=True))

        function_assembly_one_hot.append(assembly_one_hot)

    function_assembly_one_hot = np.asarray(function_assembly_one_hot)

    return function_opcode_one_hot, function_assembly_one_hot


# process opcode and assembly code
def process_opcode_assembly_code(raw_X):
    list_function_opcode = []
    list_function_assembly_code = []
    words_opcode = []

    list_opcode = []
    list_assembly_code = []
    max_length = -1
    length = 0
    sequence_length = np.array([]).astype(int) # actual sequence_length of each function
    for id, opcode_assembly_code in enumerate(raw_X):
        if opcode_assembly_code != '-----\n':
            opcode_assembly_code = opcode_assembly_code[:-1]
            if len(opcode_assembly_code.split('|')) == 2: # opcode co 1 byte
                opcode = opcode_assembly_code.split('|')[0]
                list_hex_code = opcode_assembly_code.split('|')[1]
            else:
                opcode = ' '.join(opcode_assembly_code.split('|')[:-1])
                list_hex_code = opcode_assembly_code.split('|')[-1]
            list_opcode.append(opcode)
            words_opcode.append(opcode)
            list_assembly_code.append(list_hex_code.split(','))

            length += 1
        else:
            list_function_opcode.append(list_opcode)
            list_function_assembly_code.append(list_assembly_code)
            list_opcode = []
            list_assembly_code = []

            if length > max_length:
                max_length = length

            sequence_length = np.append(sequence_length, length)
            length = 0

    dictionary_index, index_dictionary = build_vocab(words_opcode)

    function_opcode_one_hot, function_assembly_one_hot = convert_to_one_hot(list_function_opcode, list_function_assembly_code, dictionary_index, max_length)
    return function_opcode_one_hot, function_assembly_one_hot, sequence_length, max_length, len(dictionary_index)

def process_opcode_assembly_codeFixedLength(raw_X, max_length):
    list_function_opcode = []
    list_function_assembly_code = []
    words_opcode = []

    list_opcode = []
    list_assembly_code = []
    # max_length = -1
    length = 0
    sequence_length = np.array([]).astype(int) # actual sequence_length of each function
    for id, opcode_assembly_code in enumerate(raw_X):
        if opcode_assembly_code != '-----\n':
            opcode_assembly_code = opcode_assembly_code[:-1]
            if len(opcode_assembly_code.split('|')) == 2: # opcode co 1 byte
                opcode = opcode_assembly_code.split('|')[0]
                list_hex_code = opcode_assembly_code.split('|')[1]
            else:
                opcode = ' '.join(opcode_assembly_code.split('|')[:-1])
                list_hex_code = opcode_assembly_code.split('|')[-1]

            length += 1
            
            if length <= max_length:
                list_opcode.append(opcode)
                words_opcode.append(opcode)
                list_assembly_code.append(list_hex_code.split(','))

        else:
            list_function_opcode.append(list_opcode)
            list_function_assembly_code.append(list_assembly_code)
            list_opcode = []
            list_assembly_code = []

            # if length > max_length:
            #     max_length = length

            sequence_length = np.append(sequence_length, max_length)
            length = 0

    dictionary_index, index_dictionary = build_vocab(words_opcode)

    function_opcode_one_hot, function_assembly_one_hot = convert_to_one_hot(list_function_opcode, list_function_assembly_code, dictionary_index, max_length)
    return function_opcode_one_hot, function_assembly_one_hot, sequence_length, max_length, len(dictionary_index)

def get_default_config():
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = False
    tf_config.allow_soft_placement = True
    return tf_config


def print_and_write_logging_file(dir, txt, running_mode, get_datetime_from_training=_DATETIME):
    print(txt[:-1])
    if running_mode == 1:
        with open(os.path.join(dir, 'training_log_' + _DATETIME + '.txt'), 'a') as f:
            f.write(txt)
    elif running_mode == 0:
        with open(os.path.join(dir, 'testing_log_' + get_datetime_from_training + '.txt'), 'a') as f:
            f.write(txt)
    else:
        with open(os.path.join(dir, 'visualization_log_' + get_datetime_from_training + '.txt'), 'a') as f:
            f.write(txt)

def save_all_params(class_object):
    attributes = inspect.getmembers(class_object, lambda a: not (inspect.isroutine(a)))
    list_params = [a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))]
    message = 'List parameters'
    message += '{\n'
    for params in list_params:
        try:
            message += '\t' + str(params[0]) + ': ' + str(params[1]) + '\n'
        except:
            continue
    message += '}\n'
    if class_object.running_mode == 1:
        message += "Start training process.\n"
    elif class_object.running_mode == 0:
        message += "Start testing process.\n"
    else:
        message += "Start visualization process.\n"
    message += "-----------------------------------------------------\n"

    make_dir(_LOG)
    print_and_write_logging_file(_LOG, message, class_object.running_mode)

class CNN:
    def new_conv_layer(input, num_input_channels, filter_size, num_filters, name):
        with tf.variable_scope(name) as scope:
            # Shape of the filter-weights for the convolution
            shape = [filter_size, filter_size, num_input_channels, num_filters]

            # Create new weights (filters) with the given shape
            # weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))
            weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05), name='w')

            # Create new biases, one for each filter
            # biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))
            biases = tf.Variable(tf.constant(0.05, shape=[num_filters]), name='b')

            # TensorFlow operation for convolution
            layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

            # Add the biases to the results of the convolution.
            layer += biases

            return layer, weights

    def new_pool_layer(input, name):
        with tf.variable_scope(name) as scope:
            # TensorFlow operation for convolution
            layer = tf.nn.max_pool(value=input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            return layer

    def new_relu_layer(input, name):
        with tf.variable_scope(name) as scope:
            # TensorFlow operation for convolution
            layer = tf.nn.relu(input)
            return layer

    def new_fc_layer(input, num_inputs, num_outputs, name):
        with tf.variable_scope(name) as scope:
            # Create new weights and biases.
            # weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))
            weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05), name='w')
            # biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
            biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]), name='b')

            # Multiply the input and weights, and then add the bias-values.
            layer = tf.matmul(input, weights) + biases

            return layer

class TimeLiner:
    _timeline_dict = None

    def update_timeline(self, chrome_trace):
        # convert crome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)
        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict
        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, f_name):
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)