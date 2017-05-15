from __future__ import print_function

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

parser = argparse.ArgumentParser()

parser.add_argument('--kernel-size', dest = 'kernel_size', type = int, default = 5)
parser.add_argument('--fc-sizes', dest = 'fc_sizes', type = int, nargs = '+', default = 1024, help = 'fully connected layer size')
parser.add_argument('--fc-num', dest = 'fc_num', type = int, default = 1, help = 'fully connected layers number')
parser.add_argument('--data-path', dest = 'data_path', default = './vocs_data2/', help = 'the path where the input data are stored')
parser.add_argument('--summary-file', dest = 'summary_file', help = 'the summary file where the trained weights and network parameters are stored')
parser.add_argument('--out-path', dest = 'out_path', default = './out/', help = 'the path where the classification results are stored')
parser.add_argument('--compare', dest = 'compare', default = False, action='store_true', help = 'compare results with labels')

args = parser.parse_args()

kernel_size = args.kernel_size
fc_sizes = args.fc_sizes

if not isinstance(fc_sizes, list):
   fc_sizes = [fc_sizes]

hidden_layers_n = args.fc_num
data_path = args.data_path
out_path = args.out_path
summary_file = args.summary_file
do_compare = args.compare

image_width = 100
image_height = 100

#image_width = 28
#image_height = 28

# Network Parameters
n_input = image_width * image_height 
n_classes = 9 # Mtotal classes

batch_size = 1

# Create some wrappers for simplicity
def conv2d(x, W, b, strides = 1):
    # Conv2D wrapper, with bias and relu activation
    print(W.get_shape())
    print(b.get_shape())
    x = tf.nn.conv2d(x, W, strides = [1, strides, strides, 1], padding = 'SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k = 2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Create model
def conv_net(x, weights, biases):
    # Reshape input picture
    x = tf.reshape(x, shape = [-1, image_width, image_height, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k = 2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k = 2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc = tf.reshape(conv2, [-1, weights['wd'][0].get_shape().as_list()[0]])

    for i in range(hidden_layers_n):
        fc = tf.add(tf.matmul(fc, weights['wd'][i]), biases['bd'][i])
        fc = tf.nn.relu(fc)

    # Output, class prediction
    out = tf.add(tf.matmul(fc, weights['out']), biases['out'])
    return out


biases = {
    'bc1': None,
    'bc2': None,
    'bd': [],
    'out': None
}


# Store layers weight & bias
weights = {
    # kernel_size x kernel_size conv
    'wc1': None,
    # kernel_size x kernel_size conv
    'wc2': None,
    # fully connected
    'wd': [],
     # n_classes outputs (class prediction)
    'out': None
}

def tensor_summary_value_to_variable(value):
    fb = numpy.frombuffer(v.tensor.tensor_content, dtype = numpy.float32)
    #print(type(v.tensor.tensor_shape))
    shape = []
    for d in v.tensor.tensor_shape.dim:
        shape.append(d.size)
    print(shape)
    #fb.reshape(reversed(shape))
    fb = fb.reshape(shape)
    #w = tf.Variable.from_proto(v)
    var = tf.Variable(fb)
    return var



for e in tf.train.summary_iterator(summary_file):
    #print(e)
    for v in e.summary.value:

        #print(v.tag)
        #print(v.node_name)

        loaded = True

        if v.tag == 'kernel_size':
           kernel_size = int(v.simple_value)
        elif v.tag == 'fully_connected_layers':
           hidden_layers_n = int(v.simple_value)
           fc_sizes = [None] * hidden_layers_n
           weights['wd'] = [None] * hidden_layers_n
           biases['bd'] = [None] * hidden_layers_n
        elif v.tag.startswith('fully_connected_layer_'):
           split = v.tag.split('_')
           index = int(split[3])
           fc_sizes[index - 1] = int(v.simple_value)
        elif v.node_name is not None:

           if v.node_name == 'c1-weights':
               w = tensor_summary_value_to_variable(v)
               weights['wc1'] = w
           elif v.node_name == 'c1-biases':
               b = tensor_summary_value_to_variable(v)
               biases['bc1'] = b
           elif v.node_name == 'c2-weights':
               w = tensor_summary_value_to_variable(v)
               weights['wc2'] = w
           elif v.node_name == 'c2-biases':
               b = tensor_summary_value_to_variable(v)
               biases['bc2'] = b
           elif v.node_name == 'out-weights':
               w = tensor_summary_value_to_variable(v)
               weights['out'] = w
           elif v.node_name == 'out-biases':
               b = tensor_summary_value_to_variable(v)
               biases['out'] = b
           elif re.match('f[0-9]+-weights', v.node_name) :
               split = v.node_name.split('-')
               num = int(split[0][1:])
               w = tensor_summary_value_to_variable(v)
               weights['wd'][num - 1] = w
           elif re.match('f[0-9]+-biases', v.node_name) :
               split = v.node_name.split('-')
               num = int(split[0][1:])
               b = tensor_summary_value_to_variable(v)
               biases['bd'][num - 1] = b
           else:
               loaded = False

        else:

            loaded = False

        if loaded:
           if (v.tag is not None) and (len(v.tag) > 0):
              print(v.tag + ' loaded')
           else:
              print(v.node_name + ' loaded')


print("building classifier")
print('fully connected layers: ' + str(len(fc_sizes)))
for i in range(len(fc_sizes)):
    print('fully connected layer ' + str(i + 1) + ' size: ' + str(fc_sizes[i]))
print("kernel size: " + str(kernel_size))
print("data path: " + str(data_path))
print("out path: " + str(out_path))


def input_data():

    #file_name_list = tf.train.match_filenames_once(data_path + '*.png')
    #files_amount = file_name_list.get_shape()

    file_name_list = glob.glob(data_path + '*.png')
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
pred = tf.round(tf.sigmoid(conv_net(x_ph, weights, biases)))


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

    y_pred = sess.run(pred, { x_ph: x })

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

        accuracy_value = sess.run(accuracy, feed_dict = { y_ph: y, pred_ph: y_pred })
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
