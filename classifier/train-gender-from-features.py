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
n_input = 9
n_classes = 2 # Mtotal classes

batch_size = 64

#eval_batch_size = n_classes * 100
eval_batch_size = 50

if n_classes == 2:
    n_outputs = 1
else:
    n_outputs = n_classes
    
tf.set_random_seed(0)

# tf Graph input
x_batch_ph = tf.placeholder(tf.float32, [None, n_input], name = 'x_batch')
y_batch_ph = tf.placeholder(tf.float32, [None, n_outputs], name = 'y_batch')
pred_batch_ph = tf.placeholder(tf.float32, [None, n_outputs], name = 'pred_batch')

dropout_ph = tf.placeholder(tf.float32, name = "dropout") #dropout (keep probability)
accuracy_ph = tf.placeholder(tf.float32)
train_accuracy_ph = tf.placeholder(tf.float32)
loss_ph = tf.placeholder(tf.float32)
cost_ph = tf.placeholder(tf.float32)
learning_rate_ph = tf.placeholder(tf.float32)
is_training_ph = tf.placeholder(tf.bool)


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
train_data_mixed = np.genfromtxt("features.csv", delimiter = ',')

train_data = []

train_data.append(train_data_mixed[train_data_mixed[:, 1] == 0])
train_data.append(train_data_mixed[train_data_mixed[:, 1] == 1])

print(train_data)

if args.shuffled:
    all_files = []
    for i in range(n_classes):
        all_files.extend(train_data[i])
    random.seed(0)
    random.shuffle(all_files)
    train_data = []
    
    index = 0
    per_class = int(len(all_files) / n_classes)
    for i in range(n_classes):    
        last_index = index + per_class
        if i == n_classes - 1:
            last_index = len(all_files)
        train_data.append(all_files[index:last_index]) 
        index = last_index
    
cross_validation_chunks = 10
test_amount = 0

max_train_amount = len(max(train_data, key = len))        
#max_test_amount = int(max_train_amount / cross_validation_chunks)

    
print('processing train input')
data_per_class = max_train_amount 
print('data per class:', data_per_class)        

train_cv_data = []
train_cv_data_lengths = []
for d in train_data:
    test_amount = int(len(d) / cross_validation_chunks) 
    cv_d = np.concatenate((d[:int(args.test_chunk * test_amount)], d[int((args.test_chunk + 1) * test_amount):]))

    train_cv_data_lengths.append(len(cv_d))        
        
    #print(cv_d.shape)    
    #print(np.tile([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ((data_per_class - len(cv_d)), 1)).shape)   
        
    #print(np.repeat([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], (data_per_class - len(cv_d))))
    
        
    if len(cv_d) < data_per_class:
        cv_d = np.concatenate((cv_d, np.tile([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ((data_per_class - len(cv_d)), 1))))
             
    print('train data:', cv_d)
    #print(len(cv_d))
        
    train_cv_data.append(cv_d)    

   
test_cv_data = []
   
for d in train_data:    

    test_amount = int(len(d) / cross_validation_chunks) 

    print('test amount:', test_amount)                
    
    cv_data = d[int(args.test_chunk * test_amount):int((args.test_chunk + 1) * test_amount)]
    
    print('test data:', cv_data)
    
    test_cv_data.append(cv_data)
    
    
# input generator
def input_data(is_test_data, test_chunk_index, test_class = 0):
    
    global test_cv_data, train_cv_data, train_cv_data_lengths, max_train_amount
    
    if not is_test_data:
    
        print('processing train input')
            
        print(np.asarray(train_cv_data).shape)
            
        data_len = len(train_cv_data)        
        cv_data = tf.constant(np.asarray(train_cv_data))    
            
        cv_data_lengths = tf.constant(np.asarray(train_cv_data_lengths), dtype = tf.int32)    
    
        range_queue = tf.train.range_input_producer(max_train_amount * n_classes, shuffle = True)
    
        value = range_queue.dequeue()
    
        #class_value = tf.Print(class_value, [class_value], message = "class value: ")
        #file_value = tf.Print(file_value, [file_value], message = "file value: ")

        class_index = tf.div(value, tf.constant(max_train_amount))

        #class_index = tf.Print(class_index, [class_index], message = "class index: ")
    
        filenames = tf.gather(cv_data, class_index)
        amount = tf.gather(cv_data_lengths, class_index)
    
        file_index = tf.mod(value, amount)
    
        data = tf.gather(filenames, file_index)

    else:
    
        cv_data = test_cv_data[test_class]

        data_len = len(cv_data)        
        cv_data = tf.constant(np.asarray(cv_data))    

        print('data len:', data_len)
        
        range_queue = tf.train.range_input_producer(data_len, shuffle = False, num_epochs = 1)
    
        file_index = range_queue.dequeue()
        
        #file_index = tf.Print(file_index, [file_index], message = "file index: ")        
        
        class_index = tf.constant(test_class)
    
        data = tf.gather(cv_data, file_index)
    
    gender =  tf.reshape(tf.gather(data, tf.constant(1)), [-1])
    data = tf.slice(data, [2], [n_input])   
            
    return data, tf.cast(gender, tf.float32), data_len   


x, y, _ = input_data(False, 0)

x.set_shape([n_input])

if n_classes == 2:
    n_outputs = 1
else:
    n_outputs = n_classes
y.set_shape([n_outputs])

#y = tf.reshape(y, [n_classes])

x_batch, y_batch = tf.train.batch([x, y], batch_size = batch_size)

# Construct model
pred = conv_net(x_batch_ph, weights, biases, normalization_data, dropout_ph, is_training_ph)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = pred, labels = y_batch_ph))

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

# Define evaluation pipeline

def test_input(test_class):
    x1, y1, total = input_data(True, args.test_chunk, test_class)
    x1.set_shape([n_input])
    if n_classes == 2:
        n_outputs = 1
    else:
        n_outputs = n_classes
    y1.set_shape([n_outputs])
    x1_batch, y1_batch = tf.train.batch([x1, y1], batch_size = eval_batch_size, allow_smaller_final_batch = True)
    return (x1_batch, y1_batch, total)
    
    
#pred1 = conv_net(x1_batch, weights, biases, normalization_data, dropout_ph, is_training_ph)

#pred1 = conv_net(x1_batch, weights, biases, dropout_ph)
#y1_batch = tf.Print(y1_batch, [y1_batch], 'label', summarize = 30)
#pred1 = tf.Print(pred1, [pred1], 'pred ', summarize = 30)
#correct_pred = tf.equal(tf.argmax(pred1, 1), tf.argmax(y1_batch, 1))
#correct_pred = tf.reduce_all(tf.equal(pred1, y1_batch), 1)

#correct_pred = tf.equal(tf.argmax(pred1, 1), tf.argmax(y1_batch, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) 


pred_int = tf.rint(tf.sigmoid(pred_batch_ph))
correct_pred = tf.equal(pred_int, tf.rint(y_batch_ph))
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

#grid = tf_visualization.put_averaged_kernels_on_color_grid (weights['wc2'], grid_Y = 8, grid_X = 8)
#grid = tf_visualization.put_fully_connected_on_grid (weights['wd1'], grid_Y = 25, grid_X = 25)

# the end of graph construction

#sess = tf.Session(config = tf.ConfigProto(operation_timeout_in_ms = 200000, inter_op_parallelism_threads = 1000, intra_op_parallelism_threads = 1))
sess = tf.Session()

train_writer = tf.summary.FileWriter('./train',  sess.graph)

# todo : print out 'batch loss'

iterations = max(1, int(train_amount / batch_size)) * epochs

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



def calc_test_accuracy():

    global accuracy_value
    global class_accuracies
    #batches = int(round(test_amount / eval_batch_size + 0.5))

    print("evaluating test data...")

    test_sess = tf.Session()
    
    test_batches = []
    
    with test_sess.as_default():

        try:
    
            for n in range(n_classes):    
                x1_batch, y1_batch, test_amount = test_input(n)       
                test_batches.append((x1_batch, y1_batch, test_amount))
        
            test_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

            test_sess.run(test_init)
            
            test_coord = tf.train.Coordinator()

            tf.train.start_queue_runners(sess = test_sess, coord = test_coord)
            
            for n in range(n_classes):
            
                x1_batch, y1_batch, test_amount = test_batches[n]
                
                #print("test amount:", test_amount)
                
                batch_number = int((test_amount + eval_batch_size - 1) / eval_batch_size)
                
                acc_sum = 0.0
                for i in range(batch_number):
                    
                    x, y = test_sess.run([x1_batch, y1_batch])
                    
                    print('batch size:', len(y))
                    
                    p = sess.run(pred, feed_dict = { dropout_ph: 0.0, is_training_ph: False, x_batch_ph: x } )
                    acc = sess.run(accuracy, feed_dict = { pred_batch_ph : p, y_batch_ph : y } )                       
                    acc_sum = acc_sum + acc * len(y)
                    
                class_accuracies[n] = acc_sum / test_amount
                
                print(class_accuracies[n])

            test_coord.request_stop()
            test_coord.join()

            test_sess.close()
        except Exception as e:
            
            print('exception in calc_test_acuracy:', traceback.print_exc(file = sys.stdout))
            
        accuracy_value = np.mean(class_accuracies)

def calc_train_accuracy(pred, y):
    global train_accuracy_value
    acc = sess.run(accuracy, feed_dict = { pred_batch_ph : pred, y_batch_ph : y } )
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

    batches_per_epoch = train_amount / batch_size
    epoch = int(iteration / batches_per_epoch)
    done = int((iteration * 100) / total)
    batch = int(iteration % batches_per_epoch);

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

    x, y = sess.run([x_batch, y_batch], feed_dict = { dropout_ph: dropout } )

    if i % summary_interval == 0:
        calc_test_accuracy()

    _, loss_value, cost_value, p = sess.run([optimizer, loss, cost, pred], feed_dict = { x_batch_ph: x, y_batch_ph : y, dropout_ph: dropout, is_training_ph: True } )

    calc_train_accuracy(p, y)

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
