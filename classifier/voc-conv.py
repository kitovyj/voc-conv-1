from __future__ import print_function

'''

Original code:

https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py

See also

http://stackoverflow.com/questions/34340489/tensorflow-read-images-with-labels
http://stackoverflow.com/questions/37091899/how-to-actually-read-csv-data-in-tensorflow
https://gist.github.com/eerwitt/518b0c9564e500b4b50f
http://stackoverflow.com/questions/37504470/tensorflow-crashes-when-using-sess-run
http://learningtensorflow.com
http://openmachin.es/blog/tensorflow-mnist
https://freedomofkeima.com/blog/posts/flag-8-first-trial-to-image-processing-with-tensorflow

something interesting about TF

https://bamos.github.io/2016/08/09/deep-completion/

http://christopher5106.github.io/deep/learning/2015/11/11/tensorflow-google-deeplearning-library.html
https://github.com/TensorVision/TensorVision
https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/

https://ischlag.github.io/2016/06/03/simple-neural-network-in-tensorflow/

here is the best explanation of how tf works:
https://ischlag.github.io/2016/06/19/tensorflow-input-pipeline-example/

how to visualize weights:

http://stackoverflow.com/questions/33783672/how-can-i-visualize-the-weightsvariables-in-cnn-in-tensorflow
https://www.snip2code.com/Snippet/1104315/Tensorflow---visualize-convolutional-fea

softmax_cross_entropy_with_logits and sparce_softmax_cross_entropy_with_logits diference:
http://stackoverflow.com/questions/37312421/tensorflow-whats-the-difference-between-sparse-softmax-cross-entropy-with-logi

L1 and L2 regularizations explained:
https://www.quora.com/What-is-the-difference-between-L1-and-L2-regularization

independent and mutex classes:
https://www.quora.com/How-does-one-use-neural-networks-for-the-task-of-multi-class-label-classification

deep completion : https://bamos.github.io/2016/08/09/deep-completion/

'''

import sys
import numpy
import tensorflow as tf
import tf_visualization
import argparse
import time

parser = argparse.ArgumentParser()

parser.add_argument('--kernel-size', dest = 'kernel_size', type = int, default = 5)
parser.add_argument('--fc-sizes', dest = 'fc_sizes', type = int, nargs = '+', default = 1024, help = 'fully connected layer size')
parser.add_argument('--fc-num', dest = 'fc_num', type = int, default = 1, help = 'fully connected layers number')
parser.add_argument('--learning-rate', dest = 'learning_rate', type = float, default = 0.0001, help = 'learning rate')
parser.add_argument('--initial-weights-seed', dest = 'initial_weights_seed', type = int, default = None, help = 'initial weights seed')
parser.add_argument('--dropout', dest = 'dropout', type = float, default = 0.0, help = 'drop out probability')
parser.add_argument('--epochs', dest = 'epochs', type = int, default = 40, help = 'number of training epochs')
parser.add_argument('--train-amount', dest = 'train_amount', type = int, default = 150000, help = 'number of training samples')
parser.add_argument('--data-path', dest = 'data_path', default = './data/', help = 'the path where the input data are stored')
parser.add_argument('--test-amount', dest = 'test_amount', type = int, default = 500, help = 'number of test samples')

args = parser.parse_args()

kernel_size = args.kernel_size
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

image_width = 100
image_height = 100

#image_width = 28
#image_height = 28

# Network Parameters
n_input = image_width * image_height 
n_classes = 9 # Mtotal classes


batch_size = 64

eval_batch_size = n_classes * 100

# tf Graph input
#x = tf.placeholder(tf.float32, [None, n_input])
#y = tf.placeholder(tf.float32, [None, n_classes])

dropout_ph = tf.placeholder(tf.float32) #dropout (keep probability)
accuracy_ph = tf.placeholder(tf.float32) #dropout (keep probability)

# Create some wrappers for simplicity
def conv2d(x, W, b, strides = 1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides = [1, strides, strides, 1], padding = 'SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
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
        # Apply Dropout
        fc = tf.nn.dropout(fc, 1.0 - dropout)


    # Output, class prediction
    out = tf.add(tf.matmul(fc, weights['out']), biases['out'])
    return out


biases = {
    'bc1': tf.Variable(tf.zeros([32])),
    'bc2': tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32)),
    'bd': [],
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes]))
}


# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.truncated_normal([kernel_size, kernel_size, 1, 32], stddev=0.1, seed = initial_weights_seed)),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.truncated_normal([kernel_size, kernel_size, 32, 64], stddev=0.1, seed = initial_weights_seed)),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd': [],
     # 1024 inputs, n_classes outputs (class prediction)
    'out': tf.Variable(tf.truncated_normal([fc_sizes[-1], n_classes], stddev=0.1, seed = initial_weights_seed))
}


weights_copy = {
    'wc1': tf.Variable(weights['wc1'].initialized_value()),
    'wc2': tf.Variable(weights['wc2'].initialized_value()),
    'wd': [],
    'out': tf.Variable(weights['out'].initialized_value())
}

for i in range(hidden_layers_n):
  if i == 0:
     weights['wd'].append(tf.Variable(tf.truncated_normal([int((image_width / 4) * (image_height / 4) * 64), fc_sizes[i]], stddev=0.1, seed = initial_weights_seed)))
  else:
     weights['wd'].append(tf.Variable(tf.truncated_normal([fc_sizes[i - 1], fc_sizes[i]], stddev=0.1, seed = initial_weights_seed)))

  biases['bd'].append(tf.Variable(tf.constant(0.1, shape=[fc_sizes[i]])))
  weights_copy['wd'].append(tf.Variable(weights['wd'][i].initialized_value()))

  
def input_data(start_index, amount, shuffle):
    
#    data_folder = '/media/sf_vb-shared/data/'
    range_queue = tf.train.range_input_producer(amount, shuffle = shuffle)

    range_value = range_queue.dequeue()

#    if shuffle == False:
#    if shuffle == True
#    range_value = tf.Print(range_value, [range_value], message = "rv: ")            

                
    abs_index = tf.add(range_value, tf.constant(start_index))
    
    abs_index_str = tf.as_string(abs_index, width = 9, fill = '0')
    
    png_file_name = tf.string_join([tf.constant(data_path), tf.constant('data'), abs_index_str, tf.constant('.png')])
    csv_file_name = tf.string_join([tf.constant(data_path), tf.constant('data'), abs_index_str, tf.constant('.csv')])
    
#    if shuffle == False:
#    png_file_name = tf.Print(png_file_name, [png_file_name], message = "This is file name: ")
#    csv_file_name = tf.Print(csv_file_name, [csv_file_name], message = "This is file name: ")


    #filename_queue = tf.train.string_input_producer([csv_file_name])
    #filename_queue.enqueue([csv_file_name]);
    #reader = tf.TextLineReader()
    
    #_, csv_data = reader.read(filename_queue)
    csv_data = tf.read_file(csv_file_name)
    #csv_data = tf.slice(csv_data, [0], [string_length(csv_data) - 1])
 #   csv_data = tf.Print(csv_data, [csv_data], message = "This is csv_data: ")
    label_defaults = [[] for x in range(n_classes)]   
  #  csv_data = tf.Print(csv_data, [csv_data], message = "b4! ")
    unpacked_labels = tf.decode_csv(csv_data, record_defaults = label_defaults)
#    png_file_name = tf.Print(png_file_name, [png_file_name], message = "after ")
#    unpacked_labels = list(reversed(unpacked_labels))
#    unpacked_labels.pop()
    #unpacked_labels[4] = tf.constant(1, dtype = tf.float32);
    #unpacked_labels[5] = tf.constant(1, dtype = tf.float32);
    labels = tf.pack(unpacked_labels)
#    labels = tf.Print(labels, [labels], message = "These are labels: ")  
#    print(labels.get_shape())
        
    png_data = tf.read_file(png_file_name)    
    
    data = tf.image.decode_png(png_data)

    #data_shape = tf.shape(data);
    #data = tf.Print(data, [data_shape], message = "Data shape: ")            
    #data = tf.image.rgb_to_grayscale(data)
#    data = tf.image.resize_images(data, image_height, image_width)
    
    
    data = tf.reshape(data, [-1])    
    data = tf.to_float(data)
    
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
    wc1 = weights_change(weights['wc1'], weights_copy['wc1'])
    wc2 = weights_change(weights['wc2'], weights_copy['wc2'])

    for i in range(hidden_layers_n):
        wd = weights_change(weights['wd'][i], weights_copy['wd'][i])
        l.append(tf.summary.scalar('wd' + str(i + 1), wd))

    out = weights_change(weights['out'], weights_copy['out'])
    l.append(tf.summary.scalar('wc1', wc1))
    l.append(tf.summary.scalar('wc2', wc2))
    l.append(tf.summary.scalar('out', out))
    return tf.summary.merge(l)

x, y = input_data(0, train_amount, shuffle = True)

x.set_shape([image_height * image_width])
y.set_shape([n_classes])
#y = tf.reshape(y, [n_classes])

x_batch, y_batch = tf.train.batch([x, y], batch_size = batch_size)

# Construct model
pred = conv_net(x_batch, weights, biases, dropout_ph)

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
train_size = 15000

'''
learning_rate = tf.train.exponential_decay(
    0.001,                # Base learning rate.
    batch * batch_size,  # Current index into the dataset.
    train_size,          # Decay step.
    0.95,                # Decay rate.
    staircase = True)
'''

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#optimizer = tf.train.MomentumOptimizer(learning_rate, 0.1).minimize(cost, global_step=batch)

#try smaller values
#optimizer = tf.train.MomentumOptimizer(0.001, 0.9).minimize(cost)
#optimizer = tf.train.MomentumOptimizer(0.0001, 0.9).minimize(cost, global_step=batch)

#optimizer = tf.train.MomentumOptimizer(0.001, 0.9).minimize(cost, global_step=batch)

#optimizer = tf.train.MomentumOptimizer(0.001, 0.9).minimize(cost, global_step=batch)

#optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

# Define evaluation pipeline

x1, y1 = input_data(train_amount, eval_batch_size, shuffle = False)
x1.set_shape([image_height * image_width])
y1.set_shape([n_classes])

x1_batch, y1_batch = tf.train.batch([x1, y1], batch_size = eval_batch_size)
pred1 = tf.round(tf.sigmoid(conv_net(x1_batch, weights, biases, dropout_ph)))
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

grid = tf_visualization.put_kernels_on_color_grid (weights['wc1'], grid_Y = 4, grid_X = 8)
grid_orig = tf_visualization.put_kernels_on_color_grid (weights_copy['wc1'], grid_Y = 4, grid_X = 8)
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

iterations = max(1, int(train_amount / batch_size)) * epochs

'''
array = sess.run(weights['wd1'])
fname = 'wd1first.csv'
numpy.savetxt(fname, array.flatten(), "%10.10f")
'''


const_summaries = []

const_summaries.append(tf.summary.scalar('kernel size', tf.constant(kernel_size)))
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
train_summaries.append(tf.summary.image('conv1/features', grid, max_outputs = 1))
train_summaries.append(tf.summary.image('conv1orig', grid_orig, max_outputs = 1))
train_summaries.append(tf.summary.scalar('accuracy', accuracy_ph))


train_summary = tf.summary.merge(train_summaries)


start_time = time.time()

print("starting learning session")
print('fully connected layers: ' + str(len(fc_sizes)))
for i in range(len(fc_sizes)):
    print('fully connected layer ' + str(i + 1) + ' size: ' + str(fc_sizes[i]))
print("kernel size: " + str(kernel_size))
print("dropout probability: " + str(dropout))
print("initial weights seed: " + str(initial_weights_seed))
print("train amount: " + str(train_amount))
print("test amount: " + str(test_amount))
print("epochs: " + str(epochs))
print("data path: " + str(data_path))

total_summary_records = 10000
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

time_spent_summary = tf.summary.scalar('time spent', tf.constant(passed))
time_spent_summary_result = sess.run(time_spent_summary)
train_writer.add_summary(time_spent_summary_result)

print("learning ended, total time spent: " + str(passed) + " s")

# save weights

print("saving weights...")

weights_summaries = []

weights_summaries.append(tf.summary.tensor_summary('c1-weights', weights['wc1']))
weights_summaries.append(tf.summary.tensor_summary('c1-biases', biases['bc1']))
weights_summaries.append(tf.summary.tensor_summary('c2-weights', weights['wc2']))
weights_summaries.append(tf.summary.tensor_summary('c2-biases', biases['bc2']))
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

coord.request_stop()
coord.join()

sess.close()
