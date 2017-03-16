from __future__ import print_function

import sys
import numpy
import tensorflow as tf
import tf_visualization
import argparse
import time
import os
import datetime

parser = argparse.ArgumentParser()

parser.add_argument('--fc-sizes', dest = 'fc_sizes', type = int, nargs = '+', default = [16], help = 'fully connected layer size')
parser.add_argument('--fc-num', dest = 'fc_num', type = int, default = 1, help = 'fully connected layers number')
parser.add_argument('--learning-rate', dest = 'learning_rate', type = float, default = 0.0001, help = 'learning rate')
parser.add_argument('--initial-weights-seed', dest = 'initial_weights_seed', type = int, default = None, help = 'initial weights seed')
parser.add_argument('--dropout', dest = 'dropout', type = float, default = 0.0, help = 'drop out probability')
parser.add_argument('--epochs', dest = 'epochs', type = int, default = 40, help = 'number of training epochs')
parser.add_argument('--train-amount', dest = 'train_amount', type = int, default = 12454, help = 'number of training samples')
parser.add_argument('--data-path', dest = 'data_path', default = './vocs_data2/', help = 'the path where input data are stored')
parser.add_argument('--test-amount', dest = 'test_amount', type = int, default = 500, help = 'number of test samples')

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

# Network Parameters
n_classes = 1 # Mtotal classes
#n_data = 9 + 2
n_data = 2

batch_size = 640

#eval_batch_size = n_classes * 100
eval_batch_size = test_amount

# tf Graph input
#x = tf.placeholder(tf.float32, [None, n_input])
#y = tf.placeholder(tf.float32, [None, n_classes])

dropout_ph = tf.placeholder(tf.float32) #dropout (keep probability)
accuracy_ph = tf.placeholder(tf.float32) #dropout (keep probability)


# Create model
def net(x, weights, biases, dropout):

    fc =  x

    for i in range(hidden_layers_n):
        fc = tf.add(tf.matmul(fc, weights['wd'][i]), biases['bd'][i])
        fc = tf.nn.relu(fc)
        # Apply Dropout
        fc = tf.nn.dropout(fc, 1.0 - dropout)


    # Output, class prediction
    out = tf.add(tf.matmul(fc, weights['out']), biases['out'])
    return out


biases = {
    'bd': [],
    'out': tf.Variable(tf.constant(0.1, shape = [n_classes]))
}


# Store layers weight & bias
weights = {
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd': [],
     # 1024 inputs, n_classes outputs (class prediction)
    'out': None
}


if hidden_layers_n > 0:
   weights['out'] = tf.Variable(tf.truncated_normal([fc_sizes[-1], n_classes], stddev = 0.1, seed = initial_weights_seed))
else:
   weights['out'] = tf.Variable(tf.truncated_normal([n_data, n_classes], stddev = 0.1, seed = initial_weights_seed))


weights_copy = {
    'wd': [],
    'out': tf.Variable(weights['out'].initialized_value())
}

for i in range(hidden_layers_n):
  if i == 0:
     weights['wd'].append(tf.Variable(tf.truncated_normal([n_data, fc_sizes[i]], stddev = 0.1, seed = initial_weights_seed)))
  else:
     weights['wd'].append(tf.Variable(tf.truncated_normal([fc_sizes[i - 1], fc_sizes[i]], stddev = 0.1, seed = initial_weights_seed)))

  biases['bd'].append(tf.Variable(tf.constant(0.1, shape = [fc_sizes[i]])))
  weights_copy['wd'].append(tf.Variable(weights['wd'][i].initialized_value()))


def input_data(file_name_prefix, amount, shuffle):
    
    range_queue = tf.train.range_input_producer(amount, shuffle = shuffle)

    abs_index = range_queue.dequeue()

    abs_index_str = tf.as_string(abs_index, width = 9, fill = '0')
    
    data_file_name = tf.string_join([tf.constant(data_path), tf.constant('f' + file_name_prefix), abs_index_str, tf.constant('.csv')])
    labels_file_name = tf.string_join([tf.constant(data_path), tf.constant(file_name_prefix), abs_index_str, tf.constant('.csv')])

    raw_data = tf.read_file(data_file_name)

    n_actual_data = 9

    data_defaults = [[] for x in range(n_actual_data)]
    unpacked_data = tf.decode_csv(raw_data, record_defaults = data_defaults)

    raw_labels = tf.read_file(labels_file_name)
    labels_defaults = [[] for x in range(n_classes + 3)]
    unpacked_labels = tf.decode_csv(raw_labels, record_defaults = labels_defaults)

    labels = tf.pack([unpacked_labels[0]])

    #prob = tf.mod(abs_index, tf.constant(10))

    #random = tf.random_uniform([])

    #random = tf.Print(random, [random], message = "random: ")

    #other = tf.cond(unpacked_labels[0] > tf.constant(0.0), lambda: tf.constant(0.0), lambda: tf.constant(1.0))

    #random = tf.Print(random, [unpacked_labels[0]], message = "value: ")
    #other = tf.Print(other, [other], message = "other: ")

    #data_value = tf.cond(random > tf.constant(0.7), lambda: unpacked_labels[0], lambda: unpacked_labels[0])
    #data_value = tf.cond(prob > tf.constant(1), lambda: other, lambda: unpacked_labels[0])

    unpacked_data = []
    unpacked_data.append(unpacked_labels[1])
    unpacked_data.append(unpacked_labels[2])
    #unpacked_data.append(unpacked_labels[0])
    #unpacked_data.append(data_value)

#    unpacked_data = tf.Print(unpacked_data, [unpacked_labels[1]], message = "duration: ")

    #unpacked_data.append(unpacked_labels[2])


    data = tf.pack(unpacked_data)

    return data, labels
 
def euclidean_norm(a):
    return tf.sqrt(tf.reduce_sum(tf.square(a)))

def normalize(a):
    return tf.div(a, euclidean_norm(a))
    
def weights_change(a, b):
    distance = euclidean_norm(tf.sub(normalize(a), normalize(b)))
    return distance
    
def weights_change_summary():
    l = []

    for i in range(hidden_layers_n):
        wd = weights_change(weights['wd'][i], weights_copy['wd'][i])
        l.append(tf.summary.scalar('wd' + str(i + 1), wd))

    out = weights_change(weights['out'], weights_copy['out'])
    l.append(tf.summary.scalar('out', out))
    return tf.summary.merge(l)                         
    
x, y = input_data('data', train_amount, shuffle = True)

x.set_shape([n_data])
y.set_shape([n_classes])
#y = tf.reshape(y, [n_classes])

x_batch, y_batch = tf.train.batch([x, y], batch_size = batch_size)

# Construct model
pred = net(x_batch, weights, biases, dropout_ph)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, y_batch))

# L2 regularization for the fully connected parameters.

#regularizers = (tf.nn.l2_loss(weights['wd1']) + tf.nn.l2_loss(biases['bd1']) +
#                tf.nn.l2_loss(weights['out']) + tf.nn.l2_loss(biases['out']))
# Add the regularization term to the loss.
#cost += 5e-4 * regularizers


# Optimizer: set up a variable that's incremented once per batch and
# controls the learning rate decay.
batch = tf.Variable(0, dtype = tf.float32)
# Decay once per epoch, using an exponential schedule starting at 0.01.
#train_size = 15000

'''
learning_rate = tf.train.exponential_decay(
    0.001,                # Base learning rate.
    batch * batch_size,  # Current index into the dataset.
    train_size,          # Decay step.
    0.95,                # Decay rate.
    staircase = True)
'''

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
#optimizer = tf.train.MomentumOptimizer(learning_rate, 0.1).minimize(cost, global_step=batch)

#try smaller values
#optimizer = tf.train.MomentumOptimizer(0.001, 0.9).minimize(cost)
#optimizer = tf.train.MomentumOptimizer(0.0001, 0.9).minimize(cost, global_step=batch)

#optimizer = tf.train.MomentumOptimizer(0.001, 0.9).minimize(cost, global_step=batch)

#optimizer = tf.train.MomentumOptimizer(0.001, 0.9).minimize(cost, global_step=batch)

#optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

# Define evaluation pipeline

x1, y1 = input_data('test', eval_batch_size, shuffle = False)
x1.set_shape([n_data])
y1.set_shape([n_classes])

x1_batch, y1_batch = tf.train.batch([x1, y1], batch_size = eval_batch_size)
pred1 = tf.round(tf.sigmoid(net(x1_batch, weights, biases, dropout_ph)))
#y1_batch = tf.Print(y1_batch, [y1_batch], 'label', summarize = 30)
#pred1 = tf.Print(pred1, [pred1], 'pred ', summarize = 30)
#correct_pred = tf.equal(tf.argmax(pred1, 1), tf.argmax(y1_batch, 1))
#correct_pred = tf.reduce_all(tf.equal(pred1, y1_batch), 1)
correct_pred = tf.equal(pred1, y1_batch)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

accuracy_value = 0

def test_accuracy(iteration, total):
    global accuracy_value   
    batches_per_epoch = train_amount / batch_size    
    epoch = int(iteration / batches_per_epoch)
    done = int((iteration * 100) / total) 
    batch = int(iteration % batches_per_epoch);
    acc = sess.run(accuracy, feed_dict = {dropout_ph: 0.0} )
    accuracy_value = acc
    print(str(done) + "% done" + ", epoch " + str(epoch) + ", batches: " + str(batch) + ", testing accuracy: " + str(acc))

#grid = tf_visualization.put_averaged_kernels_on_color_grid (weights['wc2'], grid_Y = 8, grid_X = 8)
#grid = tf_visualization.put_fully_connected_on_grid (weights['wd1'], grid_Y = 25, grid_X = 25)

# the end of graph construction

sess = tf.Session()

train_writer = tf.summary.FileWriter('./train',  sess.graph)

# Initializing the variables
init = tf.global_variables_initializer()
    
sess.run(init)

coord = tf.train.Coordinator()

threads = tf.train.start_queue_runners(sess = sess, coord = coord)

# todo : print out 'batch loss'

#iterations = max(1, int(math.floor((examples_amount + batch_size - 1) / batch_size))) * epochs

iterations = max(1, int(train_amount / batch_size)) * epochs

'''
array = sess.run(weights['wd1'])
fname = 'wd1first.csv'
numpy.savetxt(fname, array.flatten(), "%10.10f")
'''


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
if initial_weights_seed is None:
   const_summaries.append(tf.summary.scalar('initial weights seed', tf.constant(-1)))
else:
   const_summaries.append(tf.summary.scalar('initial weights seed', tf.constant(initial_weights_seed)))

const_summary = tf.summary.merge(const_summaries)

#write const summaries

const_summary_result = sess.run(const_summary)
train_writer.add_summary(const_summary_result)    

#_, summary = sess.run([optimizer, wc1_summary], feed_dict = {keep_prob: dropout} )
# _ = sess.run([optimizer], feed_dict = {keep_prob: dropout} )
# print((i * 100) / iterations, "% done" )    
        
train_summaries = []

train_summaries.append(weights_change_summary())
train_summaries.append(tf.summary.scalar('accuracy', accuracy_ph))


train_summary = tf.summary.merge(train_summaries)

start_time = time.time()

print("starting learning session")
print('fully connected layers: ' + str(len(fc_sizes)))
for i in range(len(fc_sizes)):
    print('fully connected layer ' + str(i + 1) + ' size: ' + str(fc_sizes[i]))
print("dropout probability: " + str(dropout))
print("initial weights seed: " + str(initial_weights_seed))
print("train amount: " + str(train_amount))
print("test amount: " + str(test_amount))
print("epochs: " + str(epochs))
print("data path: " + str(data_path))

total_summary_records = 500
summary_interval = int(max(iterations / total_summary_records, 1))

print("summary interval: " + str(summary_interval))

for i in range(iterations):

    if i % summary_interval == 0:
        
        #print("Minibatch Loss= " + "{:.6f}".format(c))        
        test_accuracy(i, iterations)

    if i % summary_interval == 0:
       s = sess.run(train_summary, feed_dict = { accuracy_ph: accuracy_value })
       train_writer.add_summary(s)

    #_, c, _, summary = sess.run([optimizer, cost, learning_rate, wc1_summary], feed_dict = {keep_prob: dropout} )
    #  _, _, summary = sess.run([optimizer, learning_rate, wc1_summary], feed_dict = {keep_prob: dropout} )
    _ = sess.run([optimizer], feed_dict = { dropout_ph: dropout } )
    #_, summary = sess.run([optimizer, wc1_summary], feed_dict = {keep_prob: dropout} )
    # _ = sess.run([optimizer], feed_dict = {keep_prob: dropout} )
    # print((i * 100) / iterations, "% done" )    

    '''
    array = sess.run(weights['wc2'])
    fname = 'conv' + str(i).zfill(9) + '.csv'
    numpy.savetxt(fname, array.flatten(), "%10.10f")
    '''
    
    '''
    array = sess.run(weights['out'])
    fname = 'out' + str(i).zfill(9) + '.csv'
    numpy.savetxt(fname, array.flatten(), "%10.10f")
    '''
    

'''
array = sess.run(weights['wd1'])
fname = 'wd1last.csv'
numpy.savetxt(fname, array.flatten(), "%10.10f")
'''
                                             
test_accuracy(iterations, iterations)

s = sess.run(train_summary, feed_dict = { accuracy_ph: accuracy_value })
train_writer.add_summary(s)

end_time = time.time()
passed = end_time - start_time

time_spent_summary = tf.summary.scalar('time spent, s', tf.constant(passed))
time_spent_summary_result = sess.run(time_spent_summary)
train_writer.add_summary(time_spent_summary_result)    

print("learning ended, total time spent: " + str(passed) + " s")

# save weights

print("saving weights...")

weights_summaries = []

for i in range(hidden_layers_n):
    wname = 'f' + str(i + 1) + '-weights'
    bname = 'f' + str(i + 1) + '-biases'
    weights_summaries.append(tf.summary.tensor_summary(wname, weights['wd'][i]))
    weights_summaries.append(tf.summary.tensor_summary(bname, biases['bd'][i]))
weights_summaries.append(tf.summary.tensor_summary('out-weights', weights['out']))
weights_summaries.append(tf.summary.tensor_summary('out-biases', biases['out']))

weights_summary = tf.summary.merge(weights_summaries)

weights_summary_result = sess.run(weights_summary)
train_writer.add_summary(weights_summary_result)


'''
import os, datetime
mydir = os.path.join(os.getcwd(), weights-datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(mydir)

array = sess.run(weights['wc1'])
fname = 'c1.csv'
numpy.savetxt(fname, array.flatten(), "%10.20f")
array = sess.run(biases['bc1'])
fname = 'c1-biases.csv'
numpy.savetxt(fname, array.flatten(), "%10.20f")

array = sess.run(weights['wc2'])
fname = 'c2.csv'
numpy.savetxt(fname, array.flatten(), "%10.20f")
array = sess.run(biases['bc2'])
fname = 'c2-biases.csv'
numpy.savetxt(fname, array.flatten(), "%10.20f")

for i in range(hidden_layers_n):
    fname = 'f' + str(i + 1) + '.csv'
    array = sess.run(weights['wd'][i])
    numpy.savetxt(name, array.flatten(), "%10.20f")
    fname = 'f' + str(i + 1) + '-biases.csv'
    array = sess.run(biases['bd'][i])
    numpy.savetxt(name, array.flatten(), "%10.20f")

array = sess.run(weights['out'])
fname = 'out.csv'
numpy.savetxt(fname, array.flatten(), "%10.20f")
array = sess.run(biases['out'])
fname = 'out-biases.csv'
numpy.savetxt(fname, array.flatten(), "%10.20f")
'''

coord.request_stop()
coord.join()

sess.close()
