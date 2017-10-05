import sys
import numpy
import argparse
import time
import re
import augment
import scipy.misc
import random
import tensorflow as tf
import math

def tensor_summary_value_to_numpy(value):
    fb = numpy.frombuffer(value.tensor.tensor_content, dtype = numpy.float32)

    shape = []
    for d in value.tensor.tensor_shape.dim:
        shape.append(d.size)
    #fb.reshape(reversed(shape))
    fb = fb.reshape(shape)
    return fb

def tensor_summary_value_to_variable(value):
    fb = tensor_summary_value_to_numpy(value)

    #w = tf.Variable.from_proto(v)
    var = tf.Variable(fb)
    return var
    
def load_summary_file(summary_file):

   biases = { 'bc': [], 'bd': [], 'out': None }
   weights = { 'wc': [], 'wd': [], 'out': None }
   normalization_data = { 'nc': [], 'nd': [] }

   ge = tf.train.summary_iterator(summary_file)

   for e in ge:
       #print(e)
       #gc.collect()

       for v in e.summary.value:

           #gc.collect()

           #print("tag is " + v.tag)
           #print("node name is" + v.node_name)

           if v.node_name == "":
              v.node_name = v.tag

           #v.node_name = v.tag
           #print(v.node_name)
           loaded = True

           if v.tag == 'kernel_size':
              kernel_size = int(v.simple_value)
           elif v.tag == 'fully_connected_layers':
                hidden_layers_n = int(v.simple_value)
                fc_sizes = [None] * hidden_layers_n
                
                weights['wd'] = [None] * hidden_layers_n
                biases['bd'] = [None] * hidden_layers_n

                for i in range(hidden_layers_n):
                    normalization_data['nd'].append([None, None, None, None])
                
           elif v.tag.startswith('fully_connected_layer_'):
                split = v.tag.split('_')
                index = int(split[3])
                fc_sizes[index - 1] = int(v.simple_value)

           elif v.tag == 'convolutional_layers':
                conv_layers_n = int(v.simple_value)
                kernel_sizes = [None] * conv_layers_n
                features = [None] * conv_layers_n
                strides = [None] * conv_layers_n
                max_pooling = [None] * conv_layers_n
                weights['wc'] = [None] * conv_layers_n
                biases['bc'] = [None] * conv_layers_n

                for i in range(conv_layers_n):
                    normalization_data['nc'].append([None, None, None, None])

           elif v.tag.startswith('convolutional_layer_kernel_size_'):
                split = v.tag.split('_')
                index = int(split[-1])
                kernel_sizes[index - 1] = int(v.simple_value)

           elif v.tag.startswith('convolutional_layer_features_'):
                split = v.tag.split('_')
                index = int(split[-1])
                features[index - 1] = int(v.simple_value)

           elif v.tag.startswith('convolutional_layer_max_pooling_'):
                split = v.tag.split('_')
                index = int(split[-1])
                max_pooling[index - 1] = int(v.simple_value)

           elif v.tag.startswith('convolutional_layer_strides_'):
                split = v.tag.split('_')
                index = int(split[-1])
                strides[index - 1] = int(v.simple_value)

           elif v.node_name is not None:

                if v.node_name == 'out-weights':
                   w = tensor_summary_value_to_variable(v)
                   weights['out'] = w
                elif v.node_name == 'out-biases':
                   b = tensor_summary_value_to_variable(v)
                   biases['out'] = b
                elif re.match('c[0-9]+-weights', v.node_name) :
                   split = v.node_name.split('-')
                   num = int(split[0][1:])
                   print("loading convolutional layer " + str(num) + " weights")
                   w = tensor_summary_value_to_variable(v)
                   weights['wc'][num - 1] = w
                elif re.match('c[0-9]+-biases', v.node_name) :
                   split = v.node_name.split('-')
                   num = int(split[0][1:])
                   b = tensor_summary_value_to_variable(v)
                   biases['bc'][num - 1] = b
                elif re.match('f[0-9]+-weights', v.node_name) :
                   split = v.node_name.split('-')
                   num = int(split[0][1:])
                   print("loading fully connected layer " + str(num) + " weights")
                   w = tensor_summary_value_to_variable(v)
                   weights['wd'][num - 1] = w
                elif re.match('f[0-9]+-biases', v.node_name) :
                   split = v.node_name.split('-')
                   num = int(split[0][1:])
                   b = tensor_summary_value_to_variable(v)
                   biases['bd'][num - 1] = b
                elif re.match('nc[0-9]+-mean', v.node_name) :
                   split = v.node_name.split('-')
                   num = int(split[0][2:])
                   print("loading mean value for conv. layer " + str(num))
                   n = tensor_summary_value_to_numpy(v)
                   normalization_data['nc'][num - 1][0] = n
                elif re.match('nc[0-9]+-var', v.node_name) :
                   split = v.node_name.split('-')
                   num = int(split[0][2:])
                   n = tensor_summary_value_to_numpy(v)
                   normalization_data['nc'][num - 1][1] = n
                elif re.match('nc[0-9]+-beta', v.node_name) :
                   split = v.node_name.split('-')
                   num = int(split[0][2:])
                   n = tensor_summary_value_to_numpy(v)
                   normalization_data['nc'][num - 1][2] = n
                elif re.match('nc[0-9]+-gamma', v.node_name) :
                   split = v.node_name.split('-')
                   num = int(split[0][2:])
                   n = tensor_summary_value_to_numpy(v)
                   normalization_data['nc'][num - 1][3] = n
                elif re.match('nd[0-9]+-mean', v.node_name) :
                   split = v.node_name.split('-')
                   num = int(split[0][2:])
                   n = tensor_summary_value_to_numpy(v)
                   normalization_data['nd'][num - 1][0] = n
                elif re.match('nd[0-9]+-var', v.node_name) :
                   split = v.node_name.split('-')
                   num = int(split[0][2:])
                   n = tensor_summary_value_to_numpy(v)
                   normalization_data['nd'][num - 1][1] = n
                elif re.match('nd[0-9]+-beta', v.node_name) :
                   split = v.node_name.split('-')
                   num = int(split[0][2:])
                   n = tensor_summary_value_to_numpy(v)
                   normalization_data['nd'][num - 1][2] = n
                elif re.match('nd[0-9]+-gamma', v.node_name) :
                   split = v.node_name.split('-')
                   num = int(split[0][2:])
                   n = tensor_summary_value_to_numpy(v)
                   normalization_data['nd'][num - 1][3] = n
                else:
                   loaded = False

           else:

                loaded = False

           if loaded:
              if (v.tag is not None) and (len(v.tag) > 0):
                 print(v.tag + ' loaded')
              else:
                 print(v.node_name + ' loaded')
   e = None
   ge = None

   return weights, biases, normalization_data, kernel_sizes, features, strides, max_pooling, fc_sizes


def save_weights_to_summary(weights_summaries, weights, biases, normalization_data):

    for i in range(len(weights['wc'])):
        wname = 'c' + str(i + 1) + '-weights'
        bname = 'c' + str(i + 1) + '-biases'
        weights_summaries.append(tf.summary.tensor_summary(wname, weights['wc'][i]))
        weights_summaries.append(tf.summary.tensor_summary(bname, biases['bc'][i]))

    for i in range(len(weights['wd'])):
        wname = 'f' + str(i + 1) + '-weights'
        bname = 'f' + str(i + 1) + '-biases'
        weights_summaries.append(tf.summary.tensor_summary(wname, weights['wd'][i]))
        weights_summaries.append(tf.summary.tensor_summary(bname, biases['bd'][i]))

    weights_summaries.append(tf.summary.tensor_summary('out-weights', weights['out']))
    weights_summaries.append(tf.summary.tensor_summary('out-biases', biases['out']))

    for i in range(len(normalization_data['nc'])):
        mname = 'nc' + str(i + 1) + '-mean'
        vname = 'nc' + str(i + 1) + '-var'
        bname = 'nc' + str(i + 1) + '-beta'
        gname = 'nc' + str(i + 1) + '-gamma'
        weights_summaries.append(tf.summary.tensor_summary(mname, normalization_data['nc'][i][0]))
        weights_summaries.append(tf.summary.tensor_summary(vname, normalization_data['nc'][i][1]))
        weights_summaries.append(tf.summary.tensor_summary(bname, normalization_data['nc'][i][2]))
        weights_summaries.append(tf.summary.tensor_summary(gname, normalization_data['nc'][i][3]))

    for i in range(len(normalization_data['nd'])):
        mname = 'nd' + str(i + 1) + '-mean'
        vname = 'nd' + str(i + 1) + '-var'
        bname = 'nd' + str(i + 1) + '-beta'
        gname = 'nd' + str(i + 1) + '-gamma'
        weights_summaries.append(tf.summary.tensor_summary(mname, normalization_data['nd'][i][0]))
        weights_summaries.append(tf.summary.tensor_summary(vname, normalization_data['nd'][i][1]))
        weights_summaries.append(tf.summary.tensor_summary(bname, normalization_data['nd'][i][2]))
        weights_summaries.append(tf.summary.tensor_summary(gname, normalization_data['nd'][i][3]))

    
