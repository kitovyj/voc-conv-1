import sys
import numpy as np
import argparse
import time
import re
import scipy.misc
import random
import tensorflow as tf
import tf_visualization
import math
import string
import model_persistency
import glob
import traceback

parser = argparse.ArgumentParser()

parser.add_argument('--fc-sizes', dest = 'fc_sizes', type = int, nargs = '+', default = 1024, help = 'fully connected layer size')
parser.add_argument('--fc-num', dest = 'fc_num', type = int, default = 1, help = 'fully connected layers number')
parser.add_argument('--learning-rate', dest = 'learning_rate', type = float, default = 0.0001, help = 'learning rate')
parser.add_argument('--initial-weights-seed', dest = 'initial_weights_seed', type = int, default = None, help = 'initial weights seed')
parser.add_argument('--dropout', dest = 'dropout', type = float, default = 0.0, help = 'drop out probability')
parser.add_argument('--epochs', dest = 'epochs', type = int, default = 40, help = 'number of training epochs')
parser.add_argument('--train-amount', dest = 'train_amount', type = int, default = 11020, help = 'number of training samples')
parser.add_argument('--data-path', dest = 'data_path', default = './raw-data-gender/', help = 'the path where input data are stored')
parser.add_argument('--test-data-path', dest = 'test_data_path', default = None, help = 'the path where input test data are stored')
parser.add_argument('--test-amount', dest = 'test_amount', type = int, default = 250, help = 'number of test samples')
parser.add_argument('--summary-file', dest = 'summary_file', default = None, help = 'the summary file where the trained weights and network parameters are stored')
parser.add_argument('--regularization', dest = 'regularization_coeff', type = float, default = 100*5E-4, help = 'fully connected layers weights regularization')
parser.add_argument('--batch-normalization', action = 'store_true', dest='batch_normalization', help = 'if \'batch normalization\' is enabled')
parser.add_argument('--summary-records', dest = 'summary_records', type = int, default = 500, help = 'how much summary records should be written')
parser.add_argument('--test-chunk', dest = 'test_chunk', type = int, default = 0, help = 'the test chunk for cross validation')
parser.add_argument('--shuffled', action = 'store_true', dest='shuffled', help = 'shuffle labels')
parser.add_argument('--classify', action = 'store_true', dest='classify', help = 'just classify')
parser.add_argument('--hugo-features', action = 'store_true', dest='hugo_features', help = 'just classify')

args = parser.parse_args()

fc_sizes = args.fc_sizes

if not isinstance(fc_sizes, list):
   fc_sizes = [fc_sizes]

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

# Network Parameters

if args.hugo_features:
    n_input = 6
else:
    n_input = 9

n_classes = 2 # Mtotal classes

batch_size = 64

#eval_batch_size = n_classes * 100
eval_batch_size = 512

if n_classes == 2:
    n_outputs = 1
else:
    n_outputs = n_classes
    
tf.set_random_seed(0)
np.random.seed(0)
random.seed(0)

# tf Graph input
pred_batch_ph = tf.placeholder(tf.float32, [None, n_outputs], name = 'pred_batch')
dropout_ph = tf.placeholder(tf.float32, name = "dropout") #dropout (keep probability)
accuracy_ph = tf.placeholder(tf.float32)
train_accuracy_ph = tf.placeholder(tf.float32)
loss_ph = tf.placeholder(tf.float32)
cost_ph = tf.placeholder(tf.float32)
learning_rate_ph = tf.placeholder(tf.float32)

# Create some wrappers for simplicity

def random_string(length = 10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

# Create model
def conv_net(x, weights, biases, normalization_data, dropout, is_training, out_name = None):

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc = x

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
                
            # print("layer name:", layer_name)

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

   inputs_n = n_input

   # fully connected, 7*7*64 inputs, 1024 outputs

   for i in range(hidden_layers_n):

      # calculate variance as 2 / (inputs + outputs)
      # Glorot & Bengio => 2 / inputs

      if i == 0:

         total_inputs = int(inputs_n)
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
   
else:

   for i in range(hidden_layers_n):
      weights_copy['wd'].append(tf.Variable(weights['wd'][i].initialized_value()))

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

    for i in range(hidden_layers_n):
        wd = weights_change(weights['wd'][i], weights_copy['wd'][i])
        l.append(tf.summary.scalar('wd' + str(i + 1), wd))

    out = weights_change(weights['out'], weights_copy['out'])
    l.append(tf.summary.scalar('out', out))

    # absolute

    for i in range(hidden_layers_n):
        wda = weights_change_absolute(weights['wd'][i], weights_copy['wd'][i])
        l.append(tf.summary.scalar('wda' + str(i + 1), wda))

    out_a = weights_change_absolute(weights['out'], weights_copy['out'])
    l.append(tf.summary.scalar('out_a', out_a))

    return tf.summary.merge(l)

# initialize train data
train_data = np.genfromtxt("features.csv", delimiter = ',')
    
gender_weights = []
for i in range(2):
    gender_weights.append(len(train_data[train_data[:, 2] == i])) 
    
gender_weights = 1.0 / (np.array(gender_weights, dtype = np.float) / len(train_data))
gender_weights = gender_weights / np.sum(gender_weights) 
    
print("gender weights:", gender_weights)    
    
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

train_amount = len(train_cv_data)
   
test_cv_data = []

print('test amount:', test_amount)                
    
cv_data = train_data[int(args.test_chunk * test_amount):int((args.test_chunk + 1) * test_amount)]
    
print('test data:', cv_data)
    
test_cv_data = cv_data
    
# input generator
def input_data(is_test_data):
    
    global test_cv_data, train_cv_data, train_cv_data_lengths, max_train_amount
    
    if not is_test_data:
    
        print('processing train input')
            
        print(np.asarray(train_cv_data).shape)
            
        data_len = len(train_cv_data)        
        cv_data = tf.constant(np.asarray(train_cv_data))    
            
        cv_data_lengths = tf.constant(np.asarray(train_cv_data_lengths), dtype = tf.int32)    
    
        range_queue = tf.train.range_input_producer(data_len, shuffle = True)
    
        value = range_queue.dequeue()
        
        data = tf.gather(cv_data, value)

    else:
    
        cv_data = test_cv_data

        data_len = len(cv_data)        
        cv_data = tf.constant(np.asarray(cv_data))    

        print('data len:', data_len)
        
        range_queue = tf.train.range_input_producer(data_len, shuffle = False, num_epochs = 1)
    
        value = range_queue.dequeue()
                    
        data = tf.gather(cv_data, value)
       

    gender =  tf.reshape(tf.gather(data, tf.constant(2)), [-1])

    if args.hugo_features:
        start_index = 6
    else:
        start_index = 3

    data = tf.slice(data, [start_index], [n_input])   
       
    # features array -> gender
       
    data = tf.cast(data, tf.float32)
    gender = tf.cast(gender, tf.float32)
        
    if is_test_data:
    
        allow_smaller_final_batch = True    
        bs = eval_batch_size
        
    else:

        allow_smaller_final_batch = False   
        bs = batch_size
        
    data.set_shape([n_input])

    if n_classes == 2:
        n_outputs = 1
    else:
        n_outputs = n_classes
    gender.set_shape([n_outputs])
        
    # makes a queue!    
    x_batch, y_batch = tf.train.batch([data, gender], batch_size = bs, capacity = 1000, allow_smaller_final_batch = allow_smaller_final_batch)
        
    return x_batch, y_batch, data_len, range_queue

x_batch, y_batch, _, _ = input_data(False)

# Construct model
pred = conv_net(x_batch, weights, biases, normalization_data, dropout, True)

# Define loss and optimizer

# https://stackoverflow.com/questions/43564490/how-correctly-calculate-tf-nn-weighted-cross-entropy-with-logits-pos-weight-vari
# the stored weights are inversed!

pos_weight = gender_weights[1] / gender_weights[0]

loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits = pred, targets = y_batch, pos_weight = pos_weight))

# L2 regularization for the fully connected parameters.

regularizers = tf.nn.l2_loss(weights['out'])

for i in range(hidden_layers_n):
    regularizers = regularizers + tf.nn.l2_loss(weights['wd'][i])  

print("regularization coeff.:", regularization_coeff) 
    
# Add the regularization term to the loss.
cost = loss + regularization_coeff * regularizers

#cost = loss

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

#optimizer = tf.train.MomentumOptimizer(learning_rate, 0.1).minimize(cost, global_step=batch)

#try smaller values
#optimizer = tf.train.MomentumOptimizer(0.001, 0.9).minimize(cost)
#optimizer = tf.train.MomentumOptimizer(0.0001, 0.9).minimize(cost, global_step=batch)

#optimizer = tf.train.MomentumOptimizer(0.001, 0.9).minimize(cost, global_step=batch)

#optimizer = tf.train.MomentumOptimizer(0.001, 0.9).minimize(cost, global_step=batch)

#optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    
#pred1 = conv_net(x1_batch, weights, biases, normalization_data, dropout_ph, is_training_ph)

#pred1 = conv_net(x1_batch, weights, biases, dropout_ph)
#y1_batch = tf.Print(y1_batch, [y1_batch], 'label', summarize = 30)
#pred1 = tf.Print(pred1, [pred1], 'pred ', summarize = 30)
#correct_pred = tf.equal(tf.argmax(pred1, 1), tf.argmax(y1_batch, 1))
#correct_pred = tf.reduce_all(tf.equal(pred1, y1_batch), 1)

#correct_pred = tf.equal(tf.argmax(pred1, 1), tf.argmax(y1_batch, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) 


pred_int = tf.rint(tf.sigmoid(pred))
correct_pred = tf.equal(pred_int, tf.rint(y_batch))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


'''
ones = tf.reduce_sum(tf.cast(y_batch_ph, tf.int32))
total = tf.size(y_batch_ph)
zeros = tf.subtract(total, ones)
pred_bool = tf.cast(tf.rint(tf.sigmoid(pred_batch_ph)), tf.bool)
correct_ones = tf.reduce_sum(tf.logical_and(pred_bool, y_batch_ph))
correct_zeros = tf.reduce_sum(tf.logical_not(tf.logical_or(pred_bool, y_batch_ph)))
'''

cast_to_int = tf.rint(tf.sigmoid(pred_batch_ph))

'''
ones_accuracy = tf.divide(tf.cast(correct_ones, tf.float32), ones)
zeros_accuracy = tf.divide(tf.cast(correct_zeros, tf.float32), zeros)
'''

#correct_pred = tf.equal(tf.rint(tf.sigmoid(pred_batch_ph)), y_batch_ph)
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# the end of graph construction

#sess = tf.Session(config = tf.ConfigProto(operation_timeout_in_ms = 200000, inter_op_parallelism_threads = 1000, intra_op_parallelism_threads = 1))
if args.classify:
    config = tf.ConfigProto(device_count = {'GPU': 0})
    sess = tf.Session(config = config)
else:
    sess = tf.Session()


train_writer = tf.summary.FileWriter('./train',  sess.graph)

# todo : print out 'batch loss'

iterations = max(1, int((train_amount * epochs + batch_size - 1) / batch_size))

const_summaries = []

const_summaries.append(tf.summary.scalar('fully connected layers', tf.constant(len(fc_sizes))))

for i in range(len(fc_sizes)):
    name = 'fully connected layer ' + str(i + 1) + ' size'
    const_summaries.append(tf.summary.scalar(name, tf.constant(fc_sizes[i])))

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

const_summary = tf.summary.merge(const_summaries)

train_summaries = []

train_summaries.append(weights_change_summary())

train_summaries.append(tf.summary.scalar('accuracy', accuracy_ph))
train_summaries.append(tf.summary.scalar('train_accuracy', train_accuracy_ph))
train_summaries.append(tf.summary.scalar('loss', loss_ph))
train_summaries.append(tf.summary.scalar('cost', cost_ph))

class_accuracies_ph = [None]*(n_classes)

for n in range(n_classes):
    class_accuracies_ph[n] = tf.placeholder(tf.float32)
    train_summaries.append(tf.summary.scalar('accuracy_' + str(n + 1), class_accuracies_ph[n]))

train_summary = tf.summary.merge(train_summaries)

start_time = time.time()

print("starting learning session")
print('fully connected layers: ' + str(len(fc_sizes)))
for i in range(len(fc_sizes)):
    print('fully connected layer ' + str(i + 1) + ' size: ' + str(fc_sizes[i]))

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

accuracy_value = 0
class_accuracies = np.zeros(n_classes)
loss_value = 0
cost_value = 0
train_accuracy_value = -1

def append_rows(source, appended):
    if len(source) == 0:
        source = appended
    else:
        source = np.append(source, appended, axis = 0)                      
    return source

def calc_test_accuracy():

    global accuracy_value
    global class_accuracies
    #batches = int(round(test_amount / eval_batch_size + 0.5))

    print("evaluating test data...")

    with tf.Graph().as_default() as graph:

        tf.set_random_seed(0)
    
        if args.classify:
            config = tf.ConfigProto(device_count = {'GPU': 0})
            test_sess = tf.Session(graph = graph, config = config)
        else:
            test_sess = tf.Session(graph = graph)
        
        with test_sess.as_default():

            try:
                
                
                x1_batch, y1_batch, test_amount, q = input_data(True)
                                                          
                # create test net
                
                print("creating testing net from scratch")
                
                test_biases = {
                    'bc': [],
                    'bd': [],
                    'out': None
                }

                test_weights = {
                    'wc': [],
                    'wd': [],
                    'out': None
                }

                test_normalization_data = {
                    'nc': [],
                    'nd': []
                }
                
                for bd in biases['bd']:
                    w = sess.run(bd)
                    test_biases['bd'].append(w)
                    
                w = sess.run(biases['out'])
                test_biases['out'] = w
                
                for wd in weights['wd']:
                    w = sess.run(wd)
                    test_weights['wd'].append(w)
                    
                w = sess.run(weights['out'])
                test_weights['out'] = w

                for nd in normalization_data['nd']:
                    nd1 = sess.run(nd)
                    #print(nd1)
                    test_normalization_data['nd'].append(nd1)

                test_pred = tf.rint(tf.sigmoid(conv_net(x1_batch, test_weights, test_biases, test_normalization_data, 0.0, False)))

                print("done")
                
                test_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
                # test_init = tf.local_variables_initializer()

                test_sess.run(test_init)
                
                test_coord = tf.train.Coordinator()

                threads = tf.train.start_queue_runners(sess = test_sess, coord = test_coord)
                                         
                #print("test amount:", test_amount)
                    
                batch_number = int((test_amount + eval_batch_size - 1) / eval_batch_size)
                    
                total_correct_ones = 0
                total_correct_zeros = 0
                total_ones = 0
                total_zeros = 0

                predictions = []
                
                for i in range(batch_number):
                                                
                    p, y = test_sess.run([test_pred, y1_batch])

                    print('batch size:', len(y))
                    
                    #print(x)
                    #print(p)
                    
                    #p = sess.run(cast_to_int, feed_dict = { pred_batch_ph : p})
                    
                    y = y.astype(np.int)
                    
                    #print(p)
                    
                    ones = np.sum(y)
                    total = y.size
                    zeros = total - ones
                    p = p > 0
                    y = y > 0
                    correct_ones = np.sum((np.logical_and(p, y)).astype(np.int))
                    correct_zeros = np.sum((np.logical_not(np.logical_or(p, y))).astype(np.int))
                    
                    total_ones += ones
                    total_zeros += zeros
                    total_correct_ones += correct_ones
                    total_correct_zeros += correct_zeros
                    
                    predictions = append_rows(predictions, p.astype(np.int))
                        
                    
                print("total ones:", total_ones)   
                print("total zeros:", total_zeros)   
                    
                total_classes = 0
                
                if total_ones > 0:
                    ma = float(total_correct_ones) / total_ones
                    total_classes += 1
                else:
                    ma = 0
                    
                print('male accuracy:', ma)

                if total_zeros > 0:
                    fa = float(total_correct_zeros) / total_zeros
                    total_classes += 1
                else:
                    fa = 0
                    
                print('female accuracy:', fa)
                
                a = (fa + ma) / total_classes
                print('accuracy:', a)

                test_coord.request_stop()

                test_sess.run(q.close(cancel_pending_enqueues = True))
                
                test_coord.join(stop_grace_period_secs = 5)
                
                #tf.Session.reset("", ["testqueues2"])
                    
                test_sess.close()   
                
                if args.classify:
                
                    f = open("features2gender_fileid_recid_origgender_predgender.csv", 'a+')            
                    
                    for i in range(len(test_cv_data)):                    
                        tcvd = test_cv_data[i]                    
                        csv_line = '';
                        csv_line += str(int(tcvd[0])) + ","                        
                        csv_line += str(int(tcvd[1])) + ","
                        csv_line += str(int(tcvd[2])) + ","
                        csv_line += str(predictions[i][0]) + ""                        
                        print(csv_line, file = f)
                        
                    f.close()                                    
                
                
            except Exception as e:
                
                print('exception in calc_test_acuracy:', traceback.print_exc(file = sys.stdout))
                
            accuracy_value = a
            class_accuracies[0] = fa
            class_accuracies[1] = ma
            
            #print("class accuracies:", class_accuracies)


def calc_train_accuracy(acc):
    global train_accuracy_value
    alpha = 0.1
    if train_accuracy_value < 0:
        train_accuracy_value = acc
    else:
        train_accuracy_value = train_accuracy_value * (1 - alpha) + alpha * acc

def display_info(iteration, total):

    global accuracy_value
    global train_accuracy_value
    global class_accuracies
    global loss_value
    global cost_value

    epoch = int((iteration * batch_size) / train_amount)
    done = int((iteration * 100) / total)
    batch = iteration

    print(str(done) + "% done" + ", epoch " + str(epoch) + ", batches: " + str(batch) + ", loss: " + "{:.9f}".format(loss_value) +  ", cost: " + "{:.9f}".format(cost_value) + ", train acc.: " + str(train_accuracy_value) + ", test acc.: " + str(accuracy_value))


def write_summaries():

    global accuracy_value
    global class_accuracies
    global train_accuracy_value
    global loss_value
    global cost_value

    fd = { accuracy_ph: accuracy_value, train_accuracy_ph: train_accuracy_value, loss_ph: loss_value, cost_ph: cost_value }
    for n in range(n_classes):
        fd[class_accuracies_ph[n]] = class_accuracies[n]

    s = sess.run(train_summary, feed_dict = fd)
    train_writer.add_summary(s)

for i in range(iterations):

    if i % summary_interval == 0:
        calc_test_accuracy()
        if args.classify:
            sys.exit() 
        
    _, loss_value, cost_value, a = sess.run([optimizer, loss, cost, accuracy])

    calc_train_accuracy(a)

    if i % summary_interval == 0:
        display_info(i, iterations)
        write_summaries();

    #sys.exit()

#model = td.Model()
#model.add(pred, sess)
#model.save("model.pkl")

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
