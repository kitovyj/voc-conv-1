import sys
import numpy
import tensorflow as tf
import tf_visualization
import argparse
import time
import os
import re
import math
import glob
import string
import random
import model_persistency
import augment

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
parser.add_argument('--data-path', dest = 'data_path', default = './vocs_data/', help = 'the path where input data are stored')
parser.add_argument('--test-data-path', dest = 'test_data_path', default = None, help = 'the path where input test data are stored')
parser.add_argument('--test-amount', dest = 'test_amount', type = int, default = 500, help = 'number of test samples')
parser.add_argument('--summary-file', dest = 'summary_file', default = None, help = 'the summary file where the trained weights and network parameters are stored')
parser.add_argument('--regularization', dest = 'regularization_coeff', type = float, default = 100*5E-4, help = 'fully connected layers weights regularization')
parser.add_argument('--batch-normalization', action='store_true', dest='batch_normalization', help='if \'batch normalization\' is enabled')
parser.add_argument('--out-path', dest = 'out_path', default = './out/', help = 'the path where the classification results are stored')
parser.add_argument('--compare', dest = 'compare', default = False, action='store_true', help = 'compare results with labels')

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
out_path = args.out_path
do_compare = args.compare

image_width = 100
image_height = 100

#image_width = 28
#image_height = 28

# Network Parameters
n_input = image_width * image_height 
n_classes = 9 # Mtotal classes

batch_size = 1

dropout_ph = tf.placeholder(tf.float32)
is_training_ph = tf.placeholder(tf.bool)


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
   total_outputs = n_classes;
   var = 2. / (total_inputs)
   #var = 2. / (total_inputs + total_outputs)

   weights['out'] = tf.Variable(tf.truncated_normal([fc_sizes[-1], n_classes], stddev = math.sqrt(var), seed = initial_weights_seed))

   biases['out'] = tf.Variable(tf.constant(0.1, shape = [n_classes]))


if summary_file is not None:
   weights, biases, normalization_data, kernel_sizes, features, strides, max_pooling, fc_sizes = model_persistency.load_summary_file(summary_file)
   hidden_layers_n = len(weights['wd'])
   conv_layers_n = len(weights['wc'])

print("building classifier")

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
print("batch normalization: " + str(batch_normalization))
print("learning rate: " + str(learning_rate))
print("regularization coefficient: " + str(regularization_coeff))
print("initial weights seed: " + str(initial_weights_seed))
print("train amount: " + str(train_amount))
print("test amount: " + str(test_amount))
print("epochs: " + str(epochs))
print("data path: " + str(data_path))

def just_prepare(gray8):
    return augment.prepare(gray8, do_augment = False)

def prepare_and_augment(gray8):
    return augment.prepare(gray8, do_augment = True)

def input_data():

    #file_name_list = tf.train.match_filenames_once(data_path + '*.png')
    #files_amount = file_name_list.get_shape()

    file_name_list = glob.glob(data_path + '*r.png')

    file_name_list.sort()

    files_amount = len(file_name_list)

    # print(file_name_list[0])

    #print(files_amount)

    file_name_queue = tf.train.string_input_producer(file_name_list, num_epochs = 1, shuffle = False)

    png_file_name = file_name_queue.dequeue()

    #png_file_name = tf.Print(png_file_name, [png_file_name], message = "fn: ")

    #split = tf.string_split([png_file_name], delimiter = '.')
    #start = tf.gather(split.values, tf.constant(0))
    #start = tf.Print(start, [start], message = "start: ")
    #csv_file_name = tf.string_join([start, tf.constant('.csv')])

    #csv_file_name = png_file_name

    png_data = tf.read_file(png_file_name)
    data = tf.image.decode_png(png_data)

    data = tf.py_func(just_prepare, [data], [tf.float32])[0]

    data = tf.reshape(data, [-1])
    data = tf.to_float(data)

    if do_compare:
        csv_data = tf.read_file(csv_file_name)
        label_defaults = [[] for x in range(n_classes)]
        unpacked_labels = tf.decode_csv(csv_data, record_defaults = label_defaults)
        labels = tf.pack(unpacked_labels)
        return data, labels, png_file_name, files_amount
    else:
        return data, png_file_name, files_amount

if do_compare:

   print('do compare!')

   x, y, files, examples_amount = input_data()
   x.set_shape([image_height * image_width])
   y.set_shape([n_classes])

   x_batch, y_batch, files_batch = tf.train.batch([x, y, files], batch_size = batch_size)

else:

   print('do not compare...')

   x, files, examples_amount = input_data()
   x.set_shape([image_height * image_width])

   x_batch, files_batch = tf.train.batch([x, files], batch_size = batch_size)

print('input samples:', examples_amount)

x_ph = tf.placeholder(tf.float32, [None, image_height * image_width])

# Construct model
pred = tf.round(tf.sigmoid(conv_net(x_ph, weights, biases, normalization_data, dropout_ph, is_training_ph)))


y_ph = tf.placeholder(tf.float32, [None, n_classes])
pred_ph = tf.placeholder(tf.float32, [None, n_classes])

correct_pred = tf.equal(pred_ph, y_ph)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# the end of graph construction

sess = tf.Session()

# Initializing the variables

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
#init = tf.global_variables_initializer()
    
sess.run(init)

coord = tf.train.Coordinator()

threads = tf.train.start_queue_runners(sess = sess, coord = coord)

# todo : print out 'batch loss'

iterations = max(1, int(math.floor((examples_amount + batch_size - 1) / batch_size)))

print("iterations:", iterations)


#write const summaries

print("starting classification")

start_time = time.time()

total_summary_records = 500
summary_interval = int(max(iterations / total_summary_records, 1))


print("summary interval: " + str(summary_interval))

os.makedirs(out_path, exist_ok = True)

c = 0

accuracy_sum = 0.
accuracies_added = 0

for i in range(iterations):

    if i % summary_interval == 0:
        done = int((i * 100) / iterations)
        print(str(done) + "% done")

    #x_ = sess.run(x)

    if do_compare:
        x, y, files = sess.run([x_batch, y_batch, files_batch])
    else:
        x, files = sess.run([x_batch, files_batch])

    y_pred = sess.run(pred, { x_ph: x, is_training_ph: False, dropout_ph: 0.0 })

    # iterate over the rows

    #print(len(y))



    for r, f in zip(y_pred, files):

        source_fn = os.path.basename(f)
        name, ext = os.path.splitext(source_fn)

#        fname = 'ftest' + str(c).zfill(9) + '.csv'

        #print(name)
        fname = name.decode() + '.csv'
        flat = r.flatten()
        row = flat.reshape((1, -1))
        #print(row.shape)
        numpy.savetxt(out_path + '/f' + fname, row, fmt = '%i', delimiter = ',', newline='')
        c = c + 1

    if do_compare:

        accuracy_value = sess.run(accuracy, feed_dict = { y_ph: y, pred_ph: y_pred, is_training_ph: False, dropout_ph: 0.0 })
        accuracy_sum += accuracy_value
        accuracies_added += 1
        overall_accuracy = accuracy_sum / accuracies_added

        print('accuracy:', overall_accuracy)


end_time = time.time()
passed = end_time - start_time

print("classification ended, total time spent: " + str(passed) + " s")

coord.request_stop()
coord.join()

sess.close()
