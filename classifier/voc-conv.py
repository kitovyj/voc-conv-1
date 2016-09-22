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


'''

import tensorflow as tf

# Parameters
learning_rate = 0.001

image_width = 100
image_height = 100

# Network Parameters
n_input = image_width * image_height 
n_classes = 2 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units


train_amount = 90000

epochs = 1

batch_size = 200
eval_batch_size = 200

# tf Graph input
#x = tf.placeholder(tf.float32, [None, n_input])
#y = tf.placeholder(tf.float32, [None, n_classes])

keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

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
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    #'wc1': tf.Variable(tf.random_normal([12, 12, 1, 32])),
    #'wc1': tf.Variable(tf.zeros([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    #'wc2': tf.Variable(tf.random_normal([12, 12, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    #'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    'wd1': tf.Variable(tf.random_normal([int((image_width / 4) * (image_height / 4) * 64), 1024])),
    # 1024 inputs, n_classes outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
    
def input_data(start_index, amount, shuffle):
    
    data_folder = '/media/sf_vb-shared/data/'
        
    folder_map = tf.constant(['a', 'b'])
    label_map = tf.constant([ [0.0, 1.0], [1.0, 0.0] ])
    
    range_queue = tf.train.range_input_producer(amount, shuffle = shuffle)

    range_value = range_queue.dequeue()

#    if shuffle == False:
#    if shuffle == True:
#       range_value = tf.Print(range_value, [range_value], message = "rv: ")            

        
    per_class = int(amount / n_classes)
    class_index = tf.div(range_value, tf.constant(per_class))
               
    label = tf.gather(label_map, class_index)
    folder = tf.gather(folder_map, class_index)
    
    relative_index = tf.mod(range_value, tf.constant(per_class))
        
    abs_index = tf.add(relative_index, tf.constant(start_index))
    
    abs_index_str = tf.as_string(abs_index, width = 9, fill = '0')
    
    file_name = tf.string_join([tf.constant(data_folder), folder, tf.constant('/data'), abs_index_str, tf.constant('.png')])
    
    #file_name = tf.Print(file_name, [file_name], message = "This is file name: ")
        
    raw_data = tf.read_file(file_name)    
    
    data = tf.image.decode_png(raw_data)

    #data_shape = tf.shape(data);
    #data = tf.Print(data, [data_shape], message = "Data shape: ")            
    #data = tf.image.rgb_to_grayscale(data)
    #data = tf.image.resize_images(data, image_height, image_width)
    
    
    data = tf.reshape(data, [-1])    
    data = tf.to_float(data)
    
    return data, label

def put_kernels_on_grid (kernel, grid_Y, grid_X, pad = 1):
    
    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.

    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)
    
    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    '''
    
    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)
    
    #x_min = tf.Print(x_min, [x_min], message = "x_min: ")            
    #x_max = tf.Print(x_max, [x_max], message = "x_max: ")
    
    kernel1 = (kernel - x_min) / (x_max - x_min)
    
    # pad X and Y
    x1 = tf.pad(kernel1, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad
    
    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, channels])) #3
    
    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, channels])) #3
    
    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 1]
    #x_min = tf.reduce_min(x7)
    #x_max = tf.reduce_max(x7)
    
    #x_min = tf.Print(x_min, [x_min], message = "x_min: ")            
    #x_max = tf.Print(x_max, [x_max], message = "x_max: ")
    
    #x8 = (x7 - x_min) / (x_max - x_min)

    #x8 = tf.Print(x8, [x8], message = "x8: ")

    # scale to [0, 255] and convert to uint8
    return tf.image.convert_image_dtype(x7, dtype = tf.uint8)

def put_averaged_kernels_on_grid (kernel, grid_Y, grid_X, pad = 1):

    print(kernel.get_shape())
        
    averaged = tf.reduce_mean(kernel, 2, keep_dims = True)

    shape = tf.shape(averaged);
    averaged = tf.Print(averaged, [shape], message = "shape: ")            
        
    x_min = tf.reduce_min(averaged)
    x_max = tf.reduce_max(averaged)
    
    kernel1 = (averaged - x_min) / (x_max - x_min)
    
    # pad X and Y
    x1 = tf.pad(kernel1, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad
    
    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, channels])) #3
    
    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, channels])) #3
    
    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 255] and convert to uint8
    return tf.image.convert_image_dtype(x7, dtype = tf.uint8)

            
x, y = input_data(0, train_amount, shuffle = True)

x.set_shape([image_height * image_width])
y.set_shape([n_classes])

x_batch, y_batch = tf.train.batch([x, y], batch_size = batch_size)

# Construct model
pred = conv_net(x_batch, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y_batch))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

# Define evaluation pipeline

x1, y1 = input_data(int(train_amount / n_classes), eval_batch_size, shuffle = False)
x1.set_shape([image_height * image_width])
y1.set_shape([n_classes])

x1_batch, y1_batch = tf.train.batch([x1, y1], batch_size = eval_batch_size)
pred1 = conv_net(x1_batch, weights, biases, keep_prob)
correct_pred = tf.equal(tf.argmax(pred1, 1), tf.argmax(y1_batch, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def test_accuracy():
    acc = sess.run(accuracy, feed_dict = {keep_prob: 1.0} )    
    print("Testing Accuracy:", acc )    


#grid = put_kernels_on_grid (weights['wc1'], grid_Y = 4, grid_X = 8)
grid = put_kernels_on_grid (weights['wc2'], grid_Y = 8, grid_X = 8)

# the end of graph construction

sess = tf.Session()

train_writer = tf.train.SummaryWriter('./train',  sess.graph)

# Initializing the variables
init = tf.initialize_all_variables()
    
sess.run(init)

coord = tf.train.Coordinator()

threads = tf.train.start_queue_runners(sess = sess, coord = coord)

# todo : print out 'batch loss'

iterations = max(1, int(train_amount / batch_size)) * epochs

for i in range(iterations):

    wc1_summary = tf.image_summary('conv1/features'+ str(i), grid, max_images = 1)

    _, summary = sess.run([optimizer, wc1_summary], feed_dict = {keep_prob: dropout} )
    print((i * 100) / iterations, "% done" )    
    # if i % 10 == 0:
    test_accuracy()
        
    train_writer.add_summary(summary)
                    
test_accuracy()
    
coord.request_stop()
coord.join()

sess.close()