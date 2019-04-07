import tensorflow as tf

def conv_cond_concat(x, y):
    #Concatenate conditioning vector on feature map axis.
    x_shapes = x.shape
    y_shapes = y.shape

    return tf.concat([x, y*tf.zeros([x_shapes[0], x_shapes[1], x_shapes[2] , y_shapes[3]])], 3)
