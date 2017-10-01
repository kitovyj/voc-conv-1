import sys
import tensorflow as tf

path = sys.argv[1]
 
print('reading...')

i = 0

for e in tf.train.summary_iterator(path):
    #sz = sys.getsizeof(e)
    #print(str(sz))
    #print(e)

    for v in e.summary.value:
        if v.tag.startswith('conv'):
            content = v.image.encoded_image_string
            fname = 'conv' + str(i).zfill(9) + '.png'
            with open(fname, 'wb') as f:
                f.write(content)    
                i = i + 1

