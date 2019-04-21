import tensorflow as tf
import math

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

def conv_cond_concat(x, y):
    #Concatenate conditioning vector on feature map axis.
    x_shapes = x.shape
    y_shapes = y.shape

    return tf.concat([x, y*tf.zeros([x_shapes[0], x_shapes[1], x_shapes[2] , y_shapes[3]])], 3)

def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name='deconv2d'):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

            # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        return deconv

def fully_connected(input_, output_size, scope=None, stddev=0.02, bias_start=0.0):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        try:
            matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
        except ValueError as err:
            msg = "NOTE: Usually, this is due to an issue with the image dimensions.  Did you correctly set '--crop' or '--input_height' or '--output_height'?"
            err.args = err.args + (msg,)
            raise
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))

        return tf.matmul(input_, matrix) + bias

def batch_norm(input , scope="scope" , train=True):
    return tf.contrib.layers.batch_norm(input , epsilon=1e-5, decay=0.9 , scale=True, scope=scope , is_training=train , updates_collections=None)
