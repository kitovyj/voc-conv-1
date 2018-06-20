import sys
import numpy as np
import argparse
import time
import re
import augment
import scipy.misc
import random
import tensorflow as tf
import tf_visualization
import math
import string
import model_persistency
import glob
import traceback
import os

parser = argparse.ArgumentParser()

parser.add_argument('--kernel-sizes', dest = 'kernel_sizes', type = int, nargs = '+', default = [], help = 'convolutional layers kernel sizes')
parser.add_argument('--features', dest = 'features', type = int, nargs = '+', default = [], help = 'convolutional layers features')
parser.add_argument('--strides', dest = 'strides', type = int, nargs = '+', default = [1, 1], help = 'convolutional layers strides')
parser.add_argument('--max-pooling', dest = 'max_pooling', type = int, nargs = '+', default = [], help = 'convolutional layers max pooling')
parser.add_argument('--fc-sizes', dest = 'fc_sizes', type = int, nargs = '+', default = 1024, help = 'fully connected layer size')
parser.add_argument('--fc-num', dest = 'fc_num', type = int, default = 1, help = 'fully connected layers number')
parser.add_argument('--learning-rate', dest = 'learning_rate', type = float, default = 0.0001, help = 'learning rate')
parser.add_argument('--initial-weights-seed', dest = 'initial_weights_seed', type = int, default = None, help = 'initial weights seed')
parser.add_argument('--dropout', dest = 'dropout', type = float, default = 0.0, help = 'drop out probability')
parser.add_argument('--epochs', dest = 'epochs', type = int, default = 40, help = 'number of training epochs')
parser.add_argument('--train-amount', dest = 'train_amount', type = int, default = 11020, help = 'number of training samples')
parser.add_argument('--data-path', dest = 'data_path', default = './data_unbalanced_new/', help = 'the path where input data are stored')
parser.add_argument('--test-data-path', dest = 'test_data_path', default = None, help = 'the path where input test data are stored')
parser.add_argument('--test-amount', dest = 'test_amount', type = int, default = 250, help = 'number of test samples')
parser.add_argument('--summary-file', dest = 'summary_file', default = None, help = 'the summary file where the trained weights and network parameters are stored')
parser.add_argument('--regularization', dest = 'regularization_coeff', type = float, default = 100*5E-4, help = 'fully connected layers weights regularization')
parser.add_argument('--batch-normalization', action = 'store_true', dest='batch_normalization', help = 'if \'batch normalization\' is enabled')
parser.add_argument('--summary-records', dest = 'summary_records', type = int, default = 500, help = 'how much summary records should be written')
parser.add_argument('--test-chunk', dest = 'test_chunk', type = int, default = 0, help = 'the test chunk for cross validation')
parser.add_argument('--shuffled', action = 'store_true', dest='shuffled', help = 'shuffle labels')
parser.add_argument('--classify', action = 'store_true', dest='classify', help = 'just classify')
parser.add_argument('--test-recordings', action = 'store_true', dest='test_recordings', help = 'use recording as test sets')

args = parser.parse_args()

kernel_sizes = args.kernel_sizes
features = args.features
strides = args.strides
max_pooling = args.max_pooling
fc_sizes = args.fc_sizes

if not isinstance(fc_sizes, list):
   fc_sizes = [fc_sizes]

if not isinstance(kernel_sizes, list):
   kernel_sizes = [kernel_sizes]

if not isinstance(strides, list):
   max_pooling = [max_pooling]

if not isinstance(max_pooling, list):
   max_pooling = [max_pooling]

conv_layers_n = len(kernel_sizes)
hidden_layers_n = args.fc_num
initial_weights_seed = args.initial_weights_seed

# Parameters
#learning_rate = 0.000005
#learning_rate = 0.0005
learning_rate = args.learning_rate
dropout = args.dropout # Dropout, probability to drop units out
epochs = args.epochs
train_amount = args.train_amount
test_amount = args.test_amount
data_path = args.data_path
test_data_path = args.test_data_path
summary_file = args.summary_file
regularization_coeff = args.regularization_coeff
batch_normalization = args.batch_normalization
summary_records = args.summary_records

#image_width = 128
#image_height = 128

image_width = 100
image_height = 100

# Network Parameters
n_input = image_width * image_height

# 3 + 4 + 4 + 1 + 1 + 1
    
n_classes = 14 # Mtotal classes

batch_size = 64

#eval_batch_size = n_classes * 100
eval_batch_size = 50

if n_classes == 2:
    n_outputs = 1
else:
    n_outputs = n_classes
    
tf.set_random_seed(0)
np.random.seed(0)
random.seed(0)

# tf Graph input
x_batch_ph = tf.placeholder(tf.float32, [None, n_input], name = 'x_batch')
incline_batch_ph = tf.placeholder(tf.float32, [None, 3], name = 'incline_batch')
peaks_batch_ph = tf.placeholder(tf.float32, [None, 4], name = 'peaks_batch')
breaks_batch_ph = tf.placeholder(tf.float32, [None, 4], name = 'breaks_batch')
broadband_batch_ph = tf.placeholder(tf.float32, [None, 1], name = 'broadband_batch')
tremolo_batch_ph = tf.placeholder(tf.float32, [None, 1], name = 'tremolo_batch')
complex_batch_ph = tf.placeholder(tf.float32, [None, 1], name = 'complex_batch')

pred_batch_ph = tf.placeholder(tf.float32, [None, n_outputs], name = 'pred_batch')

dropout_ph = tf.placeholder(tf.float32, name = "dropout") #dropout (keep probability)

incline_accuracy_ph = tf.placeholder(tf.float32)
incline_train_accuracy_ph = tf.placeholder(tf.float32)

incline_classes = 3
peaks_classes = 4
breaks_classes = 4

incline_accuracies_ph = []
for i in range(incline_classes):
    incline_accuracies_ph.append(tf.placeholder(tf.float32))

peaks_accuracies_ph = []
for i in range(peaks_classes):
    peaks_accuracies_ph.append(tf.placeholder(tf.float32))

breaks_accuracies_ph = []
for i in range(breaks_classes):
    breaks_accuracies_ph.append(tf.placeholder(tf.float32))
    
peaks_accuracy_ph = tf.placeholder(tf.float32)
peaks_train_accuracy_ph = tf.placeholder(tf.float32)

breaks_accuracy_ph = tf.placeholder(tf.float32)
breaks_train_accuracy_ph = tf.placeholder(tf.float32)

broadband_accuracy_ph = tf.placeholder(tf.float32)
broadband_train_accuracy_ph = tf.placeholder(tf.float32)

tremolo_accuracy_ph = tf.placeholder(tf.float32)
tremolo_train_accuracy_ph = tf.placeholder(tf.float32)

complex_accuracy_ph = tf.placeholder(tf.float32)
complex_train_accuracy_ph = tf.placeholder(tf.float32)

loss_ph = tf.placeholder(tf.float32)
cost_ph = tf.placeholder(tf.float32)
learning_rate_ph = tf.placeholder(tf.float32)
is_training_ph = tf.placeholder(tf.bool)

# Create some wrappers for simplicity

def random_string(length = 10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

# Create some wrappers for simplicity

def maxpool2d(x, k = 2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Create model
def conv_net(x, weights, biases, normalization_data, dropout, is_training, out_name = None):
    # Reshape input picture
    x = tf.reshape(x, shape = [-1, image_width, image_height, 1])

    conv = x

    for i in range(conv_layers_n):
        
        ks = kernel_sizes[i]
        fs = features[i]
        mp = max_pooling[i]        

        # Convolution Layer
        
        conv = tf.nn.conv2d(conv, weights['wc'][i], strides = [1, strides[i], strides[i], 1], padding = 'SAME')
        
        if batch_normalization:
        
            layer_name = random_string()

            if len(normalization_data['nc']) > i:            

                data = normalization_data['nc'][i]

                mmi = tf.constant_initializer(data[0])
                mvi = tf.constant_initializer(data[1])

                if data[2] is not None:
                   bi = tf.constant_initializer(data[2])
                   gi = tf.constant_initializer(data[3])
                else:
                   bi = tf.zeros_initializer()
                   gi = tf.ones_initializer()


            else:
                mmi = tf.zeros_initializer()
                mvi = tf.ones_initializer()
                bi = tf.zeros_initializer()
                gi = tf.ones_initializer()
                normalization_data['nc'].append(None)

            conv = tf.layers.batch_normalization(conv, training = is_training, name = layer_name, moving_mean_initializer = mmi, \
                moving_variance_initializer = mvi, gamma_initializer = gi, beta_initializer = bi)
            
            # the only way to get the layer variables...
            with tf.variable_scope(layer_name, reuse = True):
                mm = tf.get_variable('moving_mean')
                mv = tf.get_variable('moving_variance')
                beta = tf.get_variable('beta')
                gamma = tf.get_variable('gamma')

            normalization_data['nc'][i] = [mm, mv, beta, gamma]
            
        conv = tf.nn.bias_add(conv, biases['bc'][i])
        
        conv = tf.nn.relu(conv)
        
        # Max Pooling (down-sampling)
        
        if mp > 1:
           conv = maxpool2d(conv, k = mp)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc = tf.reshape(conv, [-1, weights['wd'][0].get_shape().as_list()[0]])

    for i in range(hidden_layers_n):
      
        fc = tf.matmul(fc, weights['wd'][i])
        
        if batch_normalization:
            layer_name = random_string()

            if len(normalization_data['nd']) > i:

                data = normalization_data['nd'][i]

                mmi = tf.constant_initializer(data[0])
                mvi = tf.constant_initializer(data[1])

                if data[2] is not None:
                   bi = tf.constant_initializer(data[2])
                   gi = tf.constant_initializer(data[3])
                else:
                   bi = tf.zeros_initializer()
                   gi = tf.ones_initializer()

            else:
                mmi = tf.zeros_initializer()
                mvi = tf.ones_initializer()
                bi = tf.zeros_initializer()
                gi = tf.ones_initializer()
                normalization_data['nd'].append(None)

            fc = tf.layers.batch_normalization(fc, training = is_training, name = layer_name, moving_mean_initializer = mmi, \
                moving_variance_initializer = mvi, gamma_initializer = gi, beta_initializer = bi)

            # the only way to get the layer variables...
            with tf.variable_scope(layer_name, reuse = True):
                mm = tf.get_variable('moving_mean')
                mv = tf.get_variable('moving_variance')
                beta = tf.get_variable('beta')
                gamma = tf.get_variable('gamma')
            normalization_data['nd'][i] = [mm, mv, beta, gamma]
        
        fc = tf.add(fc, biases['bd'][i])
        
        fc = tf.nn.relu(fc)
        
        # Apply Dropout
        fc = tf.nn.dropout(fc, 1.0 - dropout)


    # Output, class prediction
    out = tf.add(tf.matmul(fc, weights['out']), biases['out'], name = out_name)
    return out

biases = {
    'bc': [],
    'bd': [],
    'out': None
}

weights = {
    'wc': [],
    'wd': [],
    'out': None
}

normalization_data = {
    'nc': [],
    'nd': []
}

# Store layers weight & bias

weights_copy = {
    'wc': [],
    'wd': [],
    'out': None
}

if summary_file is None:

   pk = 1

   inputs_n = 1

   for i in range(conv_layers_n):
      ks = kernel_sizes[i]
      fs = features[i]
      mp = max_pooling[i]
      s = strides[i]

      pk = pk * mp * s

      if i == 0:
         biases['bc'].append(tf.Variable(tf.zeros([fs])))
      else:
         biases['bc'].append(tf.Variable(tf.constant(0.1, shape=[fs], dtype=tf.float32)))

      # calculate variance as 2 / (inputs + outputs)
      # Glorot & Bengio => 2 / inputs

      total_inputs = ks * ks * inputs_n;
      total_outputs = fs;
      # var = 2. / (total_inputs + total_outputs)
      var = 2. / (total_inputs)
      stddev = math.sqrt(var)
      print('stddev = ' + str(stddev))

      weights['wc'].append(tf.Variable(tf.truncated_normal([ks, ks, inputs_n, fs], stddev = stddev, seed = initial_weights_seed)))

      inputs_n = fs


   # fully connected, 7*7*64 inputs, 1024 outputs

   for i in range(hidden_layers_n):

      # calculate variance as 2 / (inputs + outputs)
      # Glorot & Bengio => 2 / inputs

      if i == 0:

         total_inputs = int((image_width / pk) * (image_height / pk) * inputs_n)
         total_outputs = fc_sizes[i];
         #var = 2. / (total_inputs + total_outputs)
         var = 2. / (total_inputs)
         stddev = math.sqrt(var)
         print('stddev = ' + str(stddev))

         weights['wd'].append(tf.Variable(tf.truncated_normal([total_inputs, fc_sizes[i]], stddev = stddev, seed = initial_weights_seed)))

      else:

         total_inputs = fc_sizes[i - 1]
         total_outputs = fc_sizes[i];
         var = 2. / (total_inputs)
         #var = 2. / (total_inputs + total_outputs)
         stddev = math.sqrt(var)
         print('stddev = ' + str(stddev))

         weights['wd'].append(tf.Variable(tf.truncated_normal([fc_sizes[i - 1], fc_sizes[i]], stddev = stddev, seed = initial_weights_seed)))

      biases['bd'].append(tf.Variable(tf.constant(0.1, shape = [fc_sizes[i]])))

   total_inputs = fc_sizes[-1]

   if n_classes == 2:
       total_outputs = 1
   else:    
       total_outputs = n_classes
        
   var = 2. / (total_inputs)
   #var = 2. / (total_inputs + total_outputs)

   weights['out'] = tf.Variable(tf.truncated_normal([fc_sizes[-1], total_outputs], stddev = math.sqrt(var), seed = initial_weights_seed))

   biases['out'] = tf.Variable(tf.constant(0.1, shape = [total_outputs]))


if summary_file is not None:
   
    weights, biases, normalization_data, kernel_sizes, features, strides, max_pooling, fc_sizes, weights_copy, biases_copy = model_persistency.load_summary_file(summary_file)
    hidden_layers_n = len(weights['wd'])
    conv_layers_n = len(weights['wc'])
    if weights_copy['wc'][0] is None:    
        weights_copy['wc'] = []        
    if weights_copy['wd'][0] is None:    
        weights_copy['wd'] = []
        
if len(weights_copy['wd']) == 0: 

   for i in range(hidden_layers_n):
      weights_copy['wd'].append(tf.Variable(weights['wd'][i].initialized_value()))

      
if len(weights_copy['wc']) == 0: 
      
   print('here')   
   for i in range(conv_layers_n):
      print('here 0')   
      weights_copy['wc'].append(tf.Variable(weights['wc'][i].initialized_value()))

if weights_copy['out'] is None: 
      
   weights_copy['out'] = tf.Variable(weights['out'].initialized_value())

def euclidean_norm(a):
    return tf.sqrt(tf.reduce_sum(tf.square(a)))

def normalize(a):
    return tf.div(a, euclidean_norm(a))

def weights_change(a, b):
    distance = euclidean_norm(tf.subtract(normalize(a), normalize(b)))
    return distance

def weights_change_absolute(a, b):
    distance = euclidean_norm(tf.subtract(a, b))
    return distance
    
def weights_change_summary():
    l = []

    for i in range(conv_layers_n):
        wc = weights_change(weights['wc'][i], weights_copy['wc'][i])
        l.append(tf.summary.scalar('wc' + str(i + 1), wc))

    for i in range(hidden_layers_n):
        wd = weights_change(weights['wd'][i], weights_copy['wd'][i])
        l.append(tf.summary.scalar('wd' + str(i + 1), wd))

    out = weights_change(weights['out'], weights_copy['out'])
    l.append(tf.summary.scalar('out', out))

    # absolute
    for i in range(conv_layers_n):
        wca = weights_change_absolute(weights['wc'][i], weights_copy['wc'][i])
        l.append(tf.summary.scalar('wca' + str(i + 1), wca))

    for i in range(hidden_layers_n):
        wda = weights_change_absolute(weights['wd'][i], weights_copy['wd'][i])
        l.append(tf.summary.scalar('wda' + str(i + 1), wda))

    out_a = weights_change_absolute(weights['out'], weights_copy['out'])
    l.append(tf.summary.scalar('out_a', out_a))

    return tf.summary.merge(l)

def just_prepare(gray8):
    return augment.prepare(gray8, do_augment = False)

def prepare_and_augment(gray8):
    return augment.prepare(gray8, do_augment = True)

# initialize train data

train_data = np.genfromtxt("labeling-hugo.csv", delimiter = ',')

# fileid incline peaks breaks broadband tremolo complex
    
# modify source data
train_data[:, 1] += 1
train_data[train_data[:, 2] > 3, 2] = 3
train_data[train_data[:, 3] > 3, 3] = 3   

np.random.shuffle(train_data) 

#np.savetxt("labels-modified.csv", train_data)

# calculate incline & peaks & breaks weights to balance loss function
    
incline_weights = []
for i in range(3):
    incline_weights.append(len(train_data[train_data[:, 1] == i]))    
incline_weights = 1.0 / (np.array(incline_weights, dtype = np.float) / len(train_data))
incline_weights = incline_weights / np.sum(incline_weights) 

print("incline weights:", incline_weights)

peaks_weights = []
for i in range(4):
    peaks_weights.append(len(train_data[train_data[:, 2] == i]))    
peaks_weights = 1.0 / (np.array(peaks_weights, dtype = np.float) / len(train_data))
peaks_weights = peaks_weights / np.sum(peaks_weights)

print("peaks weights:", peaks_weights)

breaks_weights = []
for i in range(4):
    breaks_weights.append(len(train_data[train_data[:, 3] == i]))    
breaks_weights = 1.0 / (np.array(breaks_weights, dtype = np.float) / len(train_data))
breaks_weights = breaks_weights / np.sum(breaks_weights)

print("breaks weights:", breaks_weights)

#
    
cross_validation_chunks = 10
test_amount = 0

max_train_amount = len(train_data)        
#max_test_amount = int(max_train_amount / cross_validation_chunks)

print('processing train input')
data_per_class = max_train_amount 
print('data per class:', data_per_class)        

train_cv_data = []
train_cv_data_lengths = []
test_amount = float(len(train_data)) / cross_validation_chunks

print(train_data.shape)

cv_d = np.concatenate((train_data[:int(args.test_chunk * test_amount), :], train_data[int((args.test_chunk + 1) * test_amount):, :]))

#print(cv_d)

train_cv_data_lengths.append(len(cv_d))        
        
train_cv_data = cv_d    

   
test_cv_data = []

print('test amount:', test_amount)                
    
cv_data = train_data[int(args.test_chunk * test_amount):int((args.test_chunk + 1) * test_amount)]
    
#print('test data:', cv_data)
    
test_cv_data = cv_data
    
# input generator
def input_data(is_test_data):
    
    global test_cv_data, train_cv_data, train_cv_data_lengths, max_train_amount
    
    if not is_test_data:
    
        print('processing train input')
            
        print(np.asarray(train_cv_data).shape)
            
        data_len = len(train_cv_data)        
        cv_data = tf.constant(np.asarray(train_cv_data))    
                
        range_queue = tf.train.range_input_producer(len(train_cv_data), shuffle = True)
    
        value = range_queue.dequeue()
        
        data = tf.gather(cv_data, value)

    else:
    
        data_len = len(train_cv_data)        
        cv_data = tf.constant(np.asarray(test_cv_data))    
                
        range_queue = tf.train.range_input_producer(len(test_cv_data), shuffle = False)
    
        value = range_queue.dequeue()
        
        data = tf.gather(cv_data, value)


    file_id =  tf.reshape(tf.gather(data, tf.constant(0)), [-1])        
    file_id_str = tf.as_string(tf.cast(file_id, tf.int32), width = 9, fill = '0')
    png_file_name = tf.squeeze(tf.string_join([tf.constant(data_path), tf.constant("data"), file_id_str, tf.constant('r.png')]))

    #png_file_name = tf.Print(png_file_name, [png_file_name], message = "This is file name: ")
     
    incline =  tf.reshape(tf.gather(data, tf.constant(1)), [-1])        
    peaks =  tf.reshape(tf.gather(data, tf.constant(2)), [-1])        
    breaks =  tf.reshape(tf.gather(data, tf.constant(3)), [-1])        
    broadband =  tf.reshape(tf.gather(data, tf.constant(4)), [-1])        
    tremolo =  tf.reshape(tf.gather(data, tf.constant(5)), [-1])        
    complex =  tf.reshape(tf.gather(data, tf.constant(6)), [-1])        
    
    incline = tf.squeeze(tf.one_hot(tf.cast(incline, tf.int32), 3))
    peaks = tf.squeeze(tf.one_hot(tf.cast(peaks, tf.int32), 4))
    breaks = tf.squeeze(tf.one_hot(tf.cast(breaks, tf.int32), 4))

    #incline = tf.Print(incline, [incline], message = "incline: ")
    #peaks = tf.Print(peaks, [peaks], message = "peaks: ")
    #breaks = tf.Print(breaks, [breaks], message = "breaks: ")
   
    incline =  tf.cast(incline, tf.float32)        
    peaks =  tf.cast(peaks, tf.float32)        
    breaks =  tf.cast(breaks, tf.float32)        
    broadband =  tf.cast(broadband, tf.float32)        
    tremolo =  tf.cast(tremolo, tf.float32)        
    complex =  tf.cast(complex, tf.float32)        

    #broadband = tf.Print(broadband, [broadband], message = "broadband: ")
    
    #tremolo = tf.Print(tremolo, [tremolo], message = "tremolo: ")
    #if not is_test_data:
    #    tremolo = tf.Print(tremolo, [data, tremolo, value], message = "data: ", summarize = 10)
    #complex = tf.Print(complex, [complex], message = "complex: ")
    
    png_data = tf.read_file(png_file_name)
    decoded_data = tf.image.decode_png(png_data)

    if not is_test_data:
       data1 = tf.py_func(prepare_and_augment, [decoded_data], [tf.float32])[0]
    else:
       data1 = tf.py_func(just_prepare, [decoded_data], [tf.float32])[0]    
    
    data1 = tf.reshape(data1, [-1])
    data1 = tf.to_float(data1)
    
    p_queue = tf.FIFOQueue(400, [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])

    enqueue_op = p_queue.enqueue((data1, incline, peaks, breaks, broadband, tremolo, complex))
    
    if is_test_data:
    
        qr = tf.train.QueueRunner(p_queue, [enqueue_op])
    
    else:

        qr = tf.train.QueueRunner(p_queue, [enqueue_op] * 4)
    
    tf.train.add_queue_runner(qr)
        
    return p_queue, data_len, range_queue

pq, _, _ = input_data(False)

x, incline, peaks, breaks, broadband, tremolo, complex = pq.dequeue()

x.set_shape([image_height * image_width])

incline.set_shape([3])
peaks.set_shape([4])
breaks.set_shape([4])
broadband.set_shape([1])
tremolo.set_shape([1])
complex.set_shape([1])

x_batch, incline_batch, peaks_batch, breaks_batch, broadband_batch, tremolo_batch, complex_batch = tf.train.batch([x, incline, peaks, breaks, broadband, tremolo, complex], batch_size = batch_size)

# Construct model
pred = conv_net(x_batch_ph, weights, biases, normalization_data, dropout_ph, is_training_ph)

incline_pred = tf.slice(pred, [0, 0], [-1, 3]) 
peaks_pred = tf.slice(pred, [0, 3], [-1, 4]) 
breaks_pred = tf.slice(pred, [0, 7], [-1, 4]) 
broadband_pred = tf.slice(pred, [0, 11], [-1, 1]) 
tremolo_pred = tf.slice(pred, [0, 12], [-1, 1]) 
complex_pred = tf.slice(pred, [0, 13], [-1, 1]) 

# Define loss and optimizer

'''

 the class weights could indeed be inversely proportional to their frequency in your train data. Normalizing them so that they sum up to one or to the number of classes also makes sense.
 
# your class weights
class_weights = tf.constant([[1.0, 2.0, 3.0]])
# deduce weights for batch samples based on their true label
weights = tf.reduce_sum(class_weights * onehot_labels, axis=1)
# compute your (unweighted) softmax cross entropy loss
unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(onehot_labels, logits)
# apply the weights, relying on broadcasting of the multiplication
weighted_losses = unweighted_losses * weights
# reduce the result to get your final loss
loss = tf.reduce_mean(weighted_losses)
'''



incline_weights_tf = tf.constant(np.expand_dims(incline_weights, axis = 0), dtype = tf.float32)
unweighted_incline_loss = tf.nn.softmax_cross_entropy_with_logits(logits = incline_pred, labels = incline_batch_ph)
incline_loss = tf.reduce_mean(unweighted_incline_loss * tf.reduce_sum(incline_weights_tf * incline_batch_ph, axis = 1))
loss = incline_loss

peaks_weights_tf = tf.constant(np.expand_dims(peaks_weights, axis = 0), dtype = tf.float32)
unweighted_peaks_loss = tf.nn.softmax_cross_entropy_with_logits(logits = peaks_pred, labels = peaks_batch_ph)
peaks_loss = tf.reduce_mean(unweighted_peaks_loss * tf.reduce_sum(peaks_weights_tf * peaks_batch_ph, axis = 1))
loss = loss + peaks_loss

breaks_weights_tf = tf.constant(np.expand_dims(breaks_weights, axis = 0), dtype = tf.float32)
unweighted_breaks_loss = tf.nn.softmax_cross_entropy_with_logits(logits = breaks_pred, labels = breaks_batch_ph)
breaks_loss = tf.reduce_mean(unweighted_breaks_loss * tf.reduce_sum(breaks_weights_tf * breaks_batch_ph, axis = 1))
loss = loss + breaks_loss

loss = loss + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = broadband_pred, labels = broadband_batch_ph))
loss = loss + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = tremolo_pred, labels = tremolo_batch_ph))
loss = loss + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = complex_pred, labels = complex_batch_ph))

# L2 regularization for the fully connected parameters.

regularizers = tf.nn.l2_loss(weights['out'])

for i in range(hidden_layers_n):
    regularizers = regularizers + tf.nn.l2_loss(weights['wd'][i])  

# Add the regularization term to the loss.
cost = loss + regularization_coeff * regularizers


# Optimizer: set up a variable that's incremented once per batch and
# controls the learning rate decay.
batch = tf.Variable(0, dtype = tf.float32)
# Decay once per epoch, using an exponential schedule starting at 0.01.
train_size = 15000

'''
learning_rate = tf.train.exponential_decay(
    0.001,                # Base learning rate.
    batch * batch_size,  # Current index into the dataset.
    train_size,          # Decay step.
    0.95,                # Decay rate.
    staircase = True)
'''

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
     optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

if len(kernel_sizes) > 0:

   grid = tf_visualization.put_kernels_on_color_grid (weights['wc'][0], grid_Y = 16, grid_X = 16)
   grid_orig = tf_visualization.put_kernels_on_color_grid (weights_copy['wc'][0], grid_Y = 16, grid_X = 16)

#grid = tf_visualization.put_averaged_kernels_on_color_grid (weights['wc2'], grid_Y = 8, grid_X = 8)
#grid = tf_visualization.put_fully_connected_on_grid (weights['wd1'], grid_Y = 25, grid_X = 25)

# the end of graph construction

#sess = tf.Session(config = tf.ConfigProto(operation_timeout_in_ms = 200000, inter_op_parallelism_threads = 1000, intra_op_parallelism_threads = 1))

if args.classify:
    config = tf.ConfigProto(device_count = {'GPU': 0})
    sess = tf.Session(config = config)
else:
    sess = tf.Session()

train_writer = tf.summary.FileWriter('./train',  sess.graph)

# todo : print out 'batch loss'

# could be [0..batch_size] more than epochs * train_amount
iterations = max(1, int((train_amount * epochs + batch_size - 1) / batch_size))

const_summaries = []

const_summaries.append(tf.summary.scalar('fully connected layers', tf.constant(len(fc_sizes))))

for i in range(len(fc_sizes)):
    name = 'fully connected layer ' + str(i + 1) + ' size'
    const_summaries.append(tf.summary.scalar(name, tf.constant(fc_sizes[i])))

const_summaries.append(tf.summary.scalar('convolutional layers', tf.constant(len(kernel_sizes))))

for i in range(len(kernel_sizes)):
    name = 'convolutional_layer_kernel_size_' + str(i + 1)
    const_summaries.append(tf.summary.scalar(name, tf.constant(kernel_sizes[i])))
    name = 'convolutional_layer_features_' + str(i + 1)
    const_summaries.append(tf.summary.scalar(name, tf.constant(features[i])))
    name = 'convolutional_layer_strides_' + str(i + 1)
    const_summaries.append(tf.summary.scalar(name, tf.constant(strides[i])))    
    name = 'convolutional_layer_max_pooling_' + str(i + 1)
    const_summaries.append(tf.summary.scalar(name, tf.constant(max_pooling[i])))

const_summaries.append(tf.summary.scalar('dropout probablility', tf.constant(dropout)))
const_summaries.append(tf.summary.scalar('epochs', tf.constant(epochs)))
const_summaries.append(tf.summary.scalar('train amount', tf.constant(train_amount)))
const_summaries.append(tf.summary.scalar('test amount', tf.constant(test_amount)))
const_summaries.append(tf.summary.scalar('learning rate', tf.constant(learning_rate)))
const_summaries.append(tf.summary.scalar('regularization', tf.constant(regularization_coeff)))

if initial_weights_seed is None:
   const_summaries.append(tf.summary.scalar('initial weights seed', tf.constant(-1)))
else:
   const_summaries.append(tf.summary.scalar('initial weights seed', tf.constant(initial_weights_seed)))

if len(kernel_sizes) > 0:
   const_summaries.append(tf.summary.image('conv1orig', grid_orig, max_outputs = 1))

const_summary = tf.summary.merge(const_summaries)

train_summaries = []

train_summaries.append(weights_change_summary())

if len(kernel_sizes) > 0:
   train_summaries.append(tf.summary.image('conv1/features', grid, max_outputs = 1))

train_summaries.append(tf.summary.scalar('incline_accuracy', incline_accuracy_ph))
train_summaries.append(tf.summary.scalar('incline_train_accuracy', incline_train_accuracy_ph))

train_summaries.append(tf.summary.scalar('peaks_accuracy', peaks_accuracy_ph))
train_summaries.append(tf.summary.scalar('peaks_train_accuracy', peaks_train_accuracy_ph))

train_summaries.append(tf.summary.scalar('breaks_accuracy', breaks_accuracy_ph))
train_summaries.append(tf.summary.scalar('breaks_train_accuracy', breaks_train_accuracy_ph))

train_summaries.append(tf.summary.scalar('broadband_accuracy', broadband_accuracy_ph))
train_summaries.append(tf.summary.scalar('broadband_train_accuracy', broadband_train_accuracy_ph))

train_summaries.append(tf.summary.scalar('tremolo_accuracy', tremolo_accuracy_ph))
train_summaries.append(tf.summary.scalar('tremolo_train_accuracy', tremolo_train_accuracy_ph))

train_summaries.append(tf.summary.scalar('complex_accuracy', complex_accuracy_ph))
train_summaries.append(tf.summary.scalar('complex_train_accuracy', complex_train_accuracy_ph))

for i in range(incline_classes):
    train_summaries.append(tf.summary.scalar('incline_accuracy_' + str(i), incline_accuracies_ph[i]))
        
for i in range(peaks_classes):
    train_summaries.append(tf.summary.scalar('peaks_accuracy_' + str(i), peaks_accuracies_ph[i]))

for i in range(breaks_classes):
    train_summaries.append(tf.summary.scalar('breaks_accuracy_' + str(i), breaks_accuracies_ph[i]))
    
train_summaries.append(tf.summary.scalar('loss', loss_ph))
train_summaries.append(tf.summary.scalar('cost', cost_ph))

train_summary = tf.summary.merge(train_summaries)

start_time = time.time()

print("starting learning session")
print('fully connected layers: ' + str(len(fc_sizes)))
for i in range(len(fc_sizes)):
    print('fully connected layer ' + str(i + 1) + ' size: ' + str(fc_sizes[i]))

for i in range(len(kernel_sizes)):
    print('conv. layer ' + str(i + 1) + ' kernel size: ' + str(kernel_sizes[i]))

for i in range(len(features)):
    print('conv. layer ' + str(i + 1) + ' features: ' + str(features[i]))

for i in range(len(strides)):
    print('conv. layer ' + str(i + 1) + ' strides: ' + str(strides[i]))
    
for i in range(len(max_pooling)):
    print('conv. layer ' + str(i + 1) + ' max pooling: ' + str(max_pooling[i]))

print("dropout probability: " + str(dropout))
print("learning rate: " + str(learning_rate))
print("regularization coefficient: " + str(regularization_coeff))
print("initial weights seed: " + str(initial_weights_seed))
print("train amount: " + str(train_amount))
print("test amount: " + str(test_amount))
print("epochs: " + str(epochs))
print("data path: " + str(data_path))

total_summary_records = 500

summary_interval = int(max(iterations / summary_records, 1))


print("summary interval: " + str(summary_interval))

# Initializing the variables
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

sess.run(init)

#write const summaries

const_summary_result = sess.run(const_summary)
train_writer.add_summary(const_summary_result)

coord = tf.train.Coordinator()

threads = tf.train.start_queue_runners(sess = sess, coord = coord)
    
# accuracy testing routine

loss_value = 0
cost_value = 0
train_accuracy_value = -1

incline_accuracies = np.array([0.0, 0.0, 0.0])
peaks_accuracies = np.array([0.0, 0.0, 0.0, 0.0])
breaks_accuracies = np.array([0.0, 0.0, 0.0, 0.0])
broadband_accuracy = 0.0
tremolo_accuracy = 0.0
complex_accuracy = 0.0

incline_train_accuracies = np.array([0.0, 0.0, 0.0])
peaks_train_accuracies = np.array([0.0, 0.0, 0.0, 0.0])
breaks_train_accuracies = np.array([0.0, 0.0, 0.0, 0.0])
broadband_train_accuracy = 0.0
tremolo_train_accuracy = 0.0
complex_train_accuracy = -1.0


def sigmoid(x):
    return (1 / (1 + np.exp(-x)))
    
def calc_accuracies(ip, pp, bp, bbp, tp, cp, incline, peaks, breaks, broadband, tremolo, complex):

    incline_accuracies = np.array([0.0, 0.0, 0.0])
    peaks_accuracies = np.array([0.0, 0.0, 0.0, 0.0])
    breaks_accuracies = np.array([0.0, 0.0, 0.0, 0.0])
    broadband_accuracy = 0.0
    tremolo_accuracy = 0.0
    complex_accuracy = 0.0
                    
    # assess incline accuracy

    ip = np.argmax(ip, axis = 1)
    i_labels = np.argmax(incline, axis = 1)
    
    for j in range(3):
        w = np.argwhere(i_labels == j)
        if len(w) > 0:
            equal = np.sum(ip[w] == i_labels[w])
            incline_accuracies[j] += (float(equal) / len(w))
            
    # assess peaks accuracy

    pp = np.argmax(pp, axis = 1)
    i_peaks = np.argmax(peaks, axis = 1)
        
    for j in range(4):
        w = np.argwhere(i_peaks == j)
        if len(w) > 0:
            equal = np.sum(pp[w] == i_peaks[w])
            peaks_accuracies[j] += (float(equal) / len(w)) 

    # assess break accuracy

    bp = np.argmax(bp, axis = 1)
    i_breaks = np.argmax(breaks, axis = 1)
        
    for j in range(4):
        w = np.argwhere(i_breaks == j)
        if len(w) > 0:
            equal = np.sum(bp[w] == i_breaks[w])
            breaks_accuracies[j] += (float(equal) / len(w))

    # assess broadband accuracy

    broadband_accuracy += (np.sum(np.absolute(sigmoid(bbp) - broadband)) / len(bbp))
    
    # assess tremolo accuracy

    tremolo_accuracy += (np.sum(np.absolute(sigmoid(tp) - tremolo)) / len(tp))
    
    # assess complex accuracy

    complex_accuracy += (np.sum(np.absolute(sigmoid(cp) - complex)) / len(cp))
    
    return incline_accuracies, peaks_accuracies, breaks_accuracies, broadband_accuracy, tremolo_accuracy, complex_accuracy 
    
def append_rows(source, appended):
    if len(source) == 0:
        source = appended
    else:
        source = np.append(source, appended, axis = 0)                      
    return source
    

def calc_test_accuracy():

    global incline_accuracies
    global peaks_accuracies
    global breaks_accuracies
    global broadband_accuracy
    global tremolo_accuracy
    global complex_accuracy

    incline_accuracies = np.array([0.0, 0.0, 0.0])
    peaks_accuracies = np.array([0.0, 0.0, 0.0, 0.0])
    breaks_accuracies = np.array([0.0, 0.0, 0.0, 0.0])
    broadband_accuracy = 0.0
    tremolo_accuracy = 0.0
    complex_accuracy = 0.0
    
    print("evaluating test data...")
    
    test_batches = []
    
    class_predictions = [[] for i in range(n_classes)]

    with tf.Graph().as_default() as graph:

        tf.set_random_seed(0)
    
        if args.classify:
            config = tf.ConfigProto(device_count = {'GPU': 0})
            test_sess = tf.Session(graph = graph, config = config)
        else:
            test_sess = tf.Session(graph = graph)
    
        with test_sess.as_default():

            try:

                pq, total, q = input_data(True)
                x1, incline1, peaks1, breaks1, broadband1, tremolo1, complex1 = pq.dequeue()

                x1.set_shape([image_height * image_width])

                incline1.set_shape([3])
                peaks1.set_shape([4])
                breaks1.set_shape([4])
                broadband1.set_shape([1])
                tremolo1.set_shape([1])
                complex1.set_shape([1])

                x_batch1, incline_batch1, peaks_batch1, breaks_batch1, broadband_batch1, tremolo_batch1, complex_batch1 = tf.train.batch([x1, incline1, peaks1, breaks1, broadband1, tremolo1, complex1], batch_size = eval_batch_size, allow_smaller_final_batch = True)
                                            
                print('initializing test session...')
                #test_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
                test_init = tf.local_variables_initializer()

                test_sess.run(test_init)
                
                test_coord = tf.train.Coordinator()

                threads = tf.train.start_queue_runners(sess = test_sess, coord = test_coord)
                
                class_accuracies = []

                print('run tests...')
                                                    
                batch_number = int((test_amount + eval_batch_size - 1) / eval_batch_size)
                                        
                incline = []
                peaks = []
                breaks = []
                broadband = []
                tremolo = []
                complex = []
                ip = []
                pp = []
                bp = []
                bbp = []
                tp = []
                cp = []                
                                        
                for i in range(batch_number):
                    
                    x, incline_b, peaks_b, breaks_b, broadband_b, tremolo_b, complex_b = test_sess.run([x_batch1, incline_batch1, peaks_batch1, breaks_batch1, broadband_batch1, tremolo_batch1, complex_batch1])
                    
                    ip_b, pp_b, bp_b, bbp_b, tp_b, cp_b  = sess.run([incline_pred, peaks_pred, breaks_pred, broadband_pred, tremolo_pred, complex_pred], 
                        feed_dict = { dropout_ph: 0.0, is_training_ph: False, x_batch_ph: x } )
                    
                    incline = append_rows(incline, incline_b)
                    peaks = append_rows(peaks, peaks_b)
                    breaks = append_rows(breaks, breaks_b)
                    broadband = append_rows(broadband, broadband_b)
                    tremolo = append_rows(tremolo, tremolo_b)
                    complex = append_rows(complex, complex_b)

                    ip = append_rows(ip, ip_b)
                    pp = append_rows(pp, pp_b)
                    bp = append_rows(bp, bp_b)
                    bbp = append_rows(bbp, bbp_b)
                    tp = append_rows(tp, tp_b)
                    cp = append_rows(cp, cp_b)                    
                    
                all = calc_accuracies(ip, pp, bp, bbp, tp, cp, incline, peaks, breaks, broadband, tremolo, complex)
                
                incline_accuracies, peaks_accuracies, breaks_accuracies, broadband_accuracy, tremolo_accuracy, complex_accuracy = all
                                    
                print("incline accuracies:", incline_accuracies)
                print("peaks accuracies:", peaks_accuracies)
                print("breaks accuracies:", breaks_accuracies)
                print("broadband accuracies:", broadband_accuracy)
                print("tremolo accuracies:", tremolo_accuracy)
                print("complex accuracies:", complex_accuracy)
                
                if args.classify:
                
                    pn = 0
                                        
                    f = open("predicted_features_original_predicted_fileid_incline_peaks_breaks_broadband_tremolo_complex.csv", 'a+')            
                    
                    for i in range(len(test_cv_data)):
                    
                        #print(i)
                       
                        tcvd = test_cv_data[i]
                    
                        csv_line = '';

                        csv_line += str(int(tcvd[0])) + ","
                        csv_line += str(int(tcvd[1]) - 1) + ","
                        csv_line += str(np.argmax(ip[i]) - 1) + ","
                        csv_line += str(int(tcvd[2])) + ","
                        csv_line += str(np.argmax(pp[i])) + ","
                        csv_line += str(int(tcvd[3])) + ","
                        csv_line += str(np.argmax(bp[i])) + ","
                        csv_line += str(tcvd[4]) + ","
                        csv_line += str(sigmoid(bbp[i][0])) + ","
                        csv_line += str(tcvd[5]) + ","
                        csv_line += str(sigmoid(tp[i][0])) + ","
                        csv_line += str(tcvd[6]) + ","
                        csv_line += str(sigmoid(cp[i][0]))
                        
                        print(csv_line, file = f)
                        
                    f.close()                
                                
                test_coord.request_stop()

                test_sess.run(pq.close(cancel_pending_enqueues = True))
                test_sess.run(q.close(cancel_pending_enqueues = True))
                
                test_coord.join(stop_grace_period_secs = 5)
                
                #tf.Session.reset("", ["testqueues2"])
                    
                test_sess.close()                        
                                    
            except Exception as e:
                
                print('exception in calc_test_acuracy:', traceback.print_exc(file = sys.stdout))
                    
def moving_average(current, new, alpha):    
    return current * (1 - alpha) + alpha * new
    
def calc_train_accuracy(ip, pp, bp, bbp, tp, cp, incline, peaks, breaks, broadband, tremolo, complex):

    global incline_train_accuracies
    global peaks_train_accuracies
    global breaks_train_accuracies
    global broadband_train_accuracy
    global tremolo_train_accuracy
    global complex_train_accuracy
 
    all = calc_accuracies(ip, pp, bp, bbp, tp, cp, incline, peaks, breaks, broadband, tremolo, complex)
    batch_incline_accuracies, batch_peaks_accuracies, batch_breaks_accuracies, batch_broadband_accuracy, batch_tremolo_accuracy, batch_complex_accuracy = all
 
    alpha = 0.1
    # first run
    if complex_train_accuracy < 0:
        alpha = 1.0

    incline_train_accuracies = moving_average(incline_train_accuracies, batch_incline_accuracies, alpha)        
    peaks_train_accuracies =  moving_average(peaks_train_accuracies, batch_peaks_accuracies, alpha)
    breaks_train_accuracies = moving_average(breaks_train_accuracies, batch_breaks_accuracies, alpha)
    broadband_train_accuracy = moving_average(broadband_train_accuracy, batch_broadband_accuracy, alpha)
    tremolo_train_accuracy = moving_average(tremolo_train_accuracy, batch_tremolo_accuracy, alpha)
    complex_train_accuracy = moving_average(complex_train_accuracy, batch_complex_accuracy, alpha)


def display_info(iteration, total):

    global incline_train_accuracies
    global peaks_train_accuracies
    global breaks_train_accuracies
    global broadband_train_accuracy
    global tremolo_train_accuracy
    global complex_train_accuracy

    global incline_accuracies
    global peaks_accuracies
    global breaks_accuracies
    global broadband_accuracy
    global tremolo_accuracy
    global complex_accuracy
    
    global loss_value
    global cost_value

    epoch = int((iteration * batch_size) / train_amount)
    done = int((iteration * 100) / total)
    batch = iteration

    print(str(done) + "% done" + ", epoch " + str(epoch) + ", batches: " + str(batch) + ", loss: " + "{:.9f}".format(loss_value) +  ", cost: " + "{:.9f}".format(cost_value))
    print("incline:", np.mean(incline_train_accuracies), np.mean(incline_accuracies))
    print("peaks:", np.mean(peaks_train_accuracies), np.mean(peaks_accuracies))
    print("breaks:", np.mean(breaks_train_accuracies), np.mean(breaks_accuracies))
    print("broadband:", broadband_train_accuracy, broadband_accuracy)
    print("tremolo:", tremolo_train_accuracy, tremolo_accuracy)
    print("complex:", complex_train_accuracy, complex_accuracy)


def write_summaries():

    global incline_accuracies
    global peaks_accuracies
    global breaks_accuracies
    global broadband_accuracy
    global tremolo_accuracy
    global complex_accuracy
    
    global incline_train_accuracies
    global peaks_train_accuracies
    global breaks_train_accuracies
    global broadband_train_accuracy
    global tremolo_train_accuracy
    global complex_train_accuracy

    
    global loss_value
    global cost_value

    fd = { loss_ph: loss_value, cost_ph: cost_value }
    
    fd[incline_accuracy_ph] = np.mean(incline_accuracies)
    fd[peaks_accuracy_ph] = np.mean(peaks_accuracies)
    fd[breaks_accuracy_ph] = np.mean(breaks_accuracies)
    fd[broadband_accuracy_ph] = broadband_accuracy
    fd[tremolo_accuracy_ph] = tremolo_accuracy
    fd[complex_accuracy_ph] = complex_accuracy
    
    fd[incline_train_accuracy_ph] = np.mean(incline_train_accuracies)
    fd[peaks_train_accuracy_ph] = np.mean(peaks_train_accuracies)
    fd[breaks_train_accuracy_ph] = np.mean(breaks_train_accuracies)
    fd[broadband_train_accuracy_ph] = broadband_train_accuracy
    fd[tremolo_train_accuracy_ph] = tremolo_train_accuracy
    fd[complex_train_accuracy_ph] = complex_train_accuracy

    for i in range(incline_classes):
        fd[incline_accuracies_ph[i]] = incline_accuracies[i]

    for i in range(peaks_classes):
        fd[peaks_accuracies_ph[i]] = peaks_accuracies[i]

    for i in range(breaks_classes):
        fd[breaks_accuracies_ph[i]] = breaks_accuracies[i]
            
    s = sess.run(train_summary, feed_dict = fd)
    train_writer.add_summary(s)

for i in range(iterations):

    x, incline, peaks, breaks, broadband, tremolo, complex = sess.run([x_batch, incline_batch, peaks_batch, breaks_batch, broadband_batch, tremolo_batch, complex_batch])

    if i % summary_interval == 0:
        calc_test_accuracy()
        if args.classify:
            sys.exit() 
    
    fd = { dropout_ph: dropout, is_training_ph: True, x_batch_ph: x, incline_batch_ph: incline, peaks_batch_ph: peaks, breaks_batch_ph: breaks, broadband_batch_ph: broadband, tremolo_batch_ph: tremolo, complex_batch_ph: complex }
    
    input_tensors = [optimizer, loss, cost, incline_pred, peaks_pred, breaks_pred, broadband_pred, tremolo_pred, complex_pred]
    _, loss_value, cost_value, ip, pp, bp, bbp, tp, cp  = sess.run(input_tensors, feed_dict = fd )
       
            
    calc_train_accuracy(ip, pp, bp, bbp, tp, cp, incline, peaks, breaks, broadband, tremolo, complex)

    if i % summary_interval == 0:
        display_info(i, iterations)
        write_summaries();

calc_test_accuracy()
write_summaries()

end_time = time.time()
passed = end_time - start_time

time_spent_summary = tf.summary.scalar('time spent, s', tf.constant(passed))
time_spent_summary_result = sess.run(time_spent_summary)
train_writer.add_summary(time_spent_summary_result)

print("learning ended, total time spent: " + str(passed) + " s")

# save weights

print("saving weights...")

weights_summaries = []

model_persistency.save_weights_to_summary(weights_summaries, weights, biases, normalization_data, weights_copy, biases)

weights_summary = tf.summary.merge(weights_summaries)

weights_summary_result = sess.run(weights_summary)
train_writer.add_summary(weights_summary_result)
train_writer.close()

coord.request_stop()
coord.join()

sess.close()
