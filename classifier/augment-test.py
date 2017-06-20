import augment
import scipy.misc
from scipy.ndimage import zoom
import time
import numpy

def rgb2gray(rgb):
    r = numpy.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    r[r > 255] = 255
    return r

image_width = 100
image_height = 100

file_name = 'augtest.png'

im_src = scipy.misc.imread(file_name, mode = 'L')

total_time = 0

st = time.time()

for i in range(64):

    #im = im_src
    im = augment.augment(im_src)

    #im = im_src



    #im = rgb2gray(im)

    #im = im.astype(numpy.uint8)

    shape = im.shape
    #mx = max(shape[0], shape[1])



    resized = numpy.zeros((233, 100))

    max_width = min(shape[1], 100)

    resized[0:233, 0:max_width] = im




    #resized[0:shape[0], 0:shape[1]] = im

    resized = scipy.misc.imresize(resized, (image_width, image_height), interp='nearest')


    scipy.misc.imsave('out' + str(i) + '.png', resized)

et = time.time()
total_time += et - st

print('total time: ' + str(total_time))
