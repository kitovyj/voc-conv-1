import sys
import numpy
import numpy as np
import argparse
import time
import os
import datetime
import glob
import random
import scipy.misc
from scipy.ndimage import zoom
import skimage
import math

def rgb2gray(rgb):
    return numpy.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def random_color_aug_coeff():
    aug_range = 0.5
    c = 1.0 + aug_range - 2 * aug_range * random.random();
    return c


def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # width and height of the zoomed image
    zh = int(numpy.round(zoom_factor * h))
    zw = int(numpy.round(zoom_factor * w))

    # for multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # zooming out
    if zoom_factor < 1:
        # bounding box of the clip region within the output array
        top = (h - zh) // 2
        left = (w - zw) // 2
        # zero-padding
        out = numpy.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # zooming in
    elif zoom_factor > 1:
        # bounding box of the clip region within the input array
        top = (zh - h) // 2
        left = (zw - w) // 2
        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)
        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # if zoom_factor == 1, just return the input array
    else:
        out = img
    return out

def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result

def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5) + 1
    x2 = int(image_center[0] + width * 0.5) - 1
    y1 = int(image_center[1] - height * 0.5) + 1
    y2 = int(image_center[1] + height * 0.5) - 1

    return image[y1:y2, x1:x2]

def augment(gray8):

#    gray8 = numpy.squeeze(gray8)
    return np.asarray(gray8.astype(np.float32))

    print('augment')

    image_width = 100
    image_height = 100

    # add noise

    # gray8 = skimage.util.random_noise(gray8, mode = 's&p')
    gray8 = skimage.util.random_noise(gray8, mode = 'gaussian')

    # augment volume

    rc = random_color_aug_coeff()

    gray8 = gray8.astype(np.float32)
    gray8[:] *= rc

    shape = gray8.shape

    resized = numpy.zeros((233, 100), dtype = np.float32)

    max_width = min(shape[1], 100)

    resized[0:233, 0:max_width] = gray8[:, 0:max_width]

    resized = scipy.misc.imresize(resized, (image_width, image_height), interp='nearest')

    #resized = resized[:, None]

    resized = np.asarray(resized)

    print('augment1')

    return resized

def prepare(gray8, do_augment):

    global _i

    #time.sleep(1.0)

    #return numpy.asarray(gray8.astype(numpy.float32))

    #return gray8

    #print('augment')

    gray8 = numpy.squeeze(gray8)

    image_width = 100
    image_height = 100

    if do_augment:

       # gray8 = skimage.util.random_noise(gray8, mode = 's&p')


       gray8 = gray8.astype(numpy.float32)

       gray8 /= 255.

       max_pad_coeff = 0.1
       max_pad = int(gray8.shape[1] * max_pad_coeff)

       padded = numpy.zeros((gray8.shape[0], gray8.shape[1] + 2*max_pad), dtype = gray8.dtype)

       # converts to float
       padded = skimage.util.random_noise(padded, mode = 'gaussian', var = 0.01)

       padded[:, max_pad:(max_pad + gray8.shape[1])] = gray8

       gray8 = padded

       left = int((random.random() * 2 * max_pad))
       right = gray8.shape[1] - int((random.random() * 2 * max_pad))

       gray8 = gray8[:, left:right]

       # change volume

       rc = random_color_aug_coeff()

       gray8 *= rc

       gray8[gray8 > 1.0] = 1.0

       # add noise

       # converts to 0..1, float

       gray8 = skimage.util.random_noise(gray8, mode = 'gaussian', var = 0.01 * random.random())

       gray8[gray8 > 1.0] = 1.0

       gray8 *= 255

    else:

       gray8 = gray8.astype(numpy.float32)


    shape = gray8.shape

    resized = numpy.zeros((233, 100), dtype = numpy.float32)

    max_width = min(shape[1], 100)

    resized[0:233, 0:max_width] = gray8[:, 0:max_width]

    


    # resize rescales the image if it's not uint8!

    resized = resized.astype(numpy.uint8)
    resized = scipy.misc.imresize(resized, (image_height, image_width), interp='nearest')

    '''
    if do_augment:
       fn = 'aout'
    else:
       fn = 'out'
    _i = _i + 1
    if _i > 20:
       _i = 0
    u8 = resized.astype(numpy.uint8)
    scipy.misc.toimage(u8).save(fn + str(_i) + '.png')
    '''

    #print(resized.shape)

    #resized = numpy.asarray(resized)

    #print('augment1')

    #return gray8

    resized = resized.astype(numpy.float32)
    resized = resized - resized.mean()

    resized = resized[:, None]

    return resized
