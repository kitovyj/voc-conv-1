import numpy
import tensorflow as tf

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


def put_kernels_on_color_grid (kernel, grid_Y, grid_X, pad = 1):
    
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
    
    zeros = tf.zeros(kernel.get_shape())

    # separate negative and positive values
    positive_kernel = tf.maximum(kernel, zeros)
    negative_kernel = tf.neg(tf.minimum(kernel, zeros))
    
    # normalize values    
    kmin = tf.reduce_min(positive_kernel)
    kmax = tf.reduce_max(positive_kernel)        
    positive = (positive_kernel - kmin) / (kmax - kmin)

    kmin = tf.reduce_min(negative_kernel)
    kmax = tf.reduce_max(negative_kernel)        
    negative = (negative_kernel - kmin) / (kmax - kmin)
            
    kernel1 = tf.concat(tf.constant(2), [positive, zeros, negative]);
    
    # pad X and Y
    x1 = tf.pad(kernel1, tf.constant( [[pad, pad], [pad, pad], [0, 0], [0, 0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad
    
    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, channels]))
    
    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, channels]))
    
    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 255] and convert to uint8
    return tf.image.convert_image_dtype(x7, dtype = tf.uint8)


def put_averaged_kernels_on_grid (kernel, grid_Y, grid_X, pad = 1):

    print(kernel.get_shape())
    print("something\n")
        
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


def put_fully_connected_on_grid (weights, grid_Y, grid_X, pad = 3):
        
        
    x_min = tf.reduce_min(weights)
    x_max = tf.reduce_max(weights)
    
    weights1 = (weights - x_min) / (x_max - x_min)

    weights1 = tf.reshape(weights1, tf.pack([8*32, 8*32, 1, 25*25])) 
    
    # pad X and Y
    x1 = tf.pad(weights1, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = weights1.get_shape()[0] + 2 * pad
    X = weights1.get_shape()[1] + 2 * pad
    
    channels = weights1.get_shape()[2]

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

def put_fully_connected_on_color_grid (weights, grid_Y, grid_X, pad = 0):

    weights1 = tf.reshape(weights, tf.pack([8*32, 8*32, 1, 25*25])) 

    zeros = tf.zeros(weights1.get_shape())

    # separate negative and positive values
    positive_kernel = tf.maximum(weights1, zeros)
    negative_kernel = tf.neg(tf.minimum(weights1, zeros))
    
    # normalize values    
    kmin = tf.reduce_min(positive_kernel)
    kmax = tf.reduce_max(positive_kernel)        
    positive = (positive_kernel - kmin) / (kmax - kmin)

    kmin = tf.reduce_min(negative_kernel)
    kmax = tf.reduce_max(negative_kernel)        
    negative = (negative_kernel - kmin) / (kmax - kmin)
            
    weights2 = tf.concat(tf.constant(2), [positive, zeros, negative]);

    # pad X and Y
    x1 = tf.pad(weights2, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = weights2.get_shape()[0] + 2 * pad
    X = weights2.get_shape()[1] + 2 * pad
    
    channels = weights2.get_shape()[2]

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

def put_averaged_kernels_on_color_grid (kernel, grid_Y, grid_X, pad = 1):

    #print(kernel.get_shape())
    #print("something\n")
        
    averaged = tf.reduce_mean(kernel, 2, keep_dims = True)

    #shape = tf.shape(averaged);
    #averaged = tf.Print(averaged, [shape], message = "shape: ")            

    zeros = tf.zeros(averaged.get_shape())

    # separate negative and positive values
    positive_kernel = tf.maximum(averaged, zeros)
    negative_kernel = tf.neg(tf.minimum(averaged, zeros))
    
    # normalize values    
    kmin = tf.reduce_min(positive_kernel)
    kmax = tf.reduce_max(positive_kernel)        
    positive = (positive_kernel - kmin) / (kmax - kmin)

    kmin = tf.reduce_min(negative_kernel)
    kmax = tf.reduce_max(negative_kernel)        
    negative = (negative_kernel - kmin) / (kmax - kmin)
            
    kernel1 = tf.concat(tf.constant(2), [positive, zeros, negative]);

    # pad X and Y
    x1 = tf.pad(kernel1, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad
    
    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, channels]))
    
    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, channels]))
    
    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 255] and convert to uint8
    return tf.image.convert_image_dtype(x7, dtype = tf.uint8)

