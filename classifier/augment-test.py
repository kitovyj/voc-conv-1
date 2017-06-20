import augment
import scipy.misc
from scipy.ndimage import zoom
import time
import numpy

def rgb2gray(rgb):
    r = numpy.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    r[r > 255] = 255
    return r

image_width = 256
image_height = 256

file_name = 'augtest.jpg'

im_src = scipy.misc.imread(file_name, mode = 'RGB')

total_time = 0

st = time.time()

for i in range(64):

    #im = im_src
    im = augment.augment(im_src)



    #im = rgb2gray(im)

    #im = im.astype(numpy.uint8)

    shape = im.shape
    mx = max(shape[0], shape[1])
    resized = numpy.zeros((mx, mx, 3))
    resized[0:shape[0], 0:shape[1]] = im

    resized = scipy.misc.imresize(resized, (image_width, image_height), interp='nearest')


    scipy.misc.imsave('out' + str(i) + '.png', resized)

et = time.time()
total_time += et - st

print('total time: ' + str(total_time))
