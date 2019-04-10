import os, cv2, sys
from utils import *
from ops import conv_cond_concat
import tensorflow as tf
import numpy as np

sample_size = 1000
sample_dir = 'samples/'
output_size = 128
channel = 3
learning_rate = 0.0002
batch_size = 16
max_epochs = 5000
celebA = CelebA(output_size, channel, sample_size, batch_size)
z_dim = 100
y_dim = celebA.y_dim
version = 'face_gen_v2'
log_dir = '/tmp/tensorflow_cgan/'+version
model_dir = 'model/'

def train():

    with tf.variable_scope('input'):
        images = tf.placeholder('float', shape=[batch_size,output_size, output_size, channel], name='real_images')
        z = tf.placeholder('float', shape=[batch_size,z_dim], name='noise_vec')
        y = tf.placeholder('float', shape=[batch_size,y_dim], name='condition_vec')
        phase = tf.placeholder('bool', name='phase')

    fake_images = generator(z, y, phase)

    real_result, real_logits = discriminator(images, y, phase)
    fake_result, fake_logits = discriminator(fake_images, y, phase, reuse=True)

    d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_result), logits=fake_logits))

    d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_result), logits=real_logits))

    g_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_result), logits=fake_logits))

    d_loss = d_real_loss + d_fake_loss
    g_loss = g_fake_loss

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]
    trainer_d = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5).minimize(d_loss, var_list=d_vars)
    trainer_g = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5).minimize(g_loss, var_list=g_vars)

    d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]

    batch_num = sample_size // batch_size
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    start_epoch = 1

    if not os.path.exists(model_dir+version):
        os.makedirs(model_dir+version)
    else:
        saver.restore(sess, model_dir+version+'/'+version+'.ckpt')
        with open(model_dir+version+'/epoch.txt', 'r') as ep:
            start_epoch = int(ep.read()) + 1

    print('Sample Size: {}'.format(sample_size))
    print('Batch Size: {} - Batches per Epoch: {} - Max Epochs: {}'.format(batch_size, batch_num, max_epochs))
    print('Starting training...')
    sample_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, z_dim]).astype(np.float32)
    _, sample_labels = celebA.get_next_batch(0)
    for i in range(start_epoch,max_epochs+1):
        print('Epoch {}/{}'.format(i,max_epochs))
        for j in range(batch_num):
            d_iters = 5
            g_iters = 1

            train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, z_dim]).astype(np.float32)
            train_images, real_labels = celebA.get_next_batch(i-1)

            for k in range(d_iters):
                print('\rBatch {}/{} - Itter - {}'.format(j+1,batch_num, k),end='')
                sess.run(d_clip)

                _, dLoss = sess.run([trainer_d, d_loss], feed_dict={z: train_noise, images: train_images, y: real_labels, phase: True})
            for k in range(g_iters):
                _, gLoss = sess.run([trainer_g, g_loss], feed_dict={z: train_noise, y: real_labels, phase: True})
        print('')
        if i%50 == 0:
            saver.save(sess,model_dir+version+'/'+version+'.ckpt')
            with open(model_dir+version+'/epoch.txt', 'w') as ep:
                ep.write(str(i))
            print('Model Saved | train:[{}] | d_loss: {:.2f} | g_loss: {:.2f}'.format(i, dLoss, gLoss))

        if i%5 == 0:
            if not os.path.exists(sample_dir+version):
                os.makedirs(sample_dir+version)
            imgtest = sess.run(fake_images, feed_dict={z: sample_noise, y: sample_labels, phase: False})
            #print(imgtest.shape)
            #print(imgtest[0])
            #cv2.imshow('frame',imgtest[0])
            #if cv2.waitKey(1) & 0xFF == ord('q'):
                #break
            imgtest = imgtest * 255.0
            save_images(imgtest, [4,4], sample_dir+version+'/epoch_'+str(i)+'.jpg')

            print('Sample Saved | train:[{}] | d_loss: {:.2f} | g_loss: {:.2f}'.format(i, dLoss, gLoss))

    coord.request_stop()
    coord.join(threads)

def test(version):

    if os.path.exists(model_dir+version+'/'+version+'.ckpt'):
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)

            saver.restore(sess, model_dir+version+'/'+version+'.ckpt')
            sample_z = np.random.uniform(1, -1, size=[batch_size, z_dim])
            description = input('Enter a description --> ')
            description_vec = text_to_vector(description)

            output = sess.run(fake_images, feed_dict={z: sample_z, y: description_vec})

            save_images(output, [4,4], './{}/test_'+version+'/test{:02d}_{:04d}.png'.format(sample_dir, 0, 0))

            image = cv2.imread('./{}/test_'+version+'/test{:02d}_{:04d}.png'.format(sample_dir, 0, 0))

            cv2.imshow('test', image)
            cv2.waitKey(0)

            print('Test Finished')
    else:
        print('ERROR - [Model {} not found]'.format(version))

def generator(z, y, phase):

    f4, f8, f16, f32, f64 = 512, 256, 128, 64, 32
    s = 8

    with tf.variable_scope('gen') as scope:
        yb = tf.reshape(y, shape=[batch_size, 1, 1, y_dim])
        z = tf.concat([z, y], 1)

        weight = tf.get_variable('w', shape=[111, s * s * f4], dtype='float', initializer=tf.truncated_normal_initializer(stddev=0.02))
        bias = tf.get_variable('b', shape=[f4 * s * s], dtype='float', initializer=tf.constant_initializer(0.0))
        flat = tf.add(tf.matmul(z, weight), bias, name='flat')

        conv1 = tf.reshape(flat, shape=[batch_size, s, s, f4], name='conv1')
        conv1 = tf.contrib.layers.batch_norm(conv1, is_training=phase, epsilon=1e-5, decay=0.9, updates_collections=None, scope='bn1')
        conv1 = tf.nn.leaky_relu(conv1, name='act1')

        #Concat label to tensor
        conv1 = conv_cond_concat(conv1, yb)

        conv2 = tf.layers.conv2d_transpose(conv1, f8, kernel_size=[5,5], strides=[2,2], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv2')
        conv2 = tf.contrib.layers.batch_norm(conv2, is_training=phase, epsilon=1e-5, decay=0.9, updates_collections=None, scope='bn2')
        conv2 = tf.nn.leaky_relu(conv2, name='act2')

        conv2 = conv_cond_concat(conv2, yb)

        conv3 = tf.layers.conv2d_transpose(conv2, f16, kernel_size=[5,5], strides=[2,2], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv3')
        conv3 = tf.contrib.layers.batch_norm(conv3, is_training=phase, epsilon=1e-5, decay=0.9, updates_collections=None, scope='bn3')
        conv3 = tf.nn.leaky_relu(conv2, name='act3')

        conv3 = conv_cond_concat(conv3, yb)

        conv4 = tf.layers.conv2d_transpose(conv3, f32, kernel_size=[5,5], strides=[2,2], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv4')
        conv4 = tf.contrib.layers.batch_norm(conv4, is_training=phase, epsilon=1e-5, decay=0.9, updates_collections=None, scope='bn4')
        conv4 = tf.nn.leaky_relu(conv4, name='act4')

        conv4 = conv_cond_concat(conv4, yb)

        conv5 = tf.layers.conv2d_transpose(conv4, f64, kernel_size=[5,5], strides=[2,2], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv5')
        conv5 = tf.contrib.layers.batch_norm(conv5, is_training=phase, epsilon=1e-5, decay=0.9, updates_collections=None, scope='bn5')
        conv5 = tf.nn.leaky_relu(conv5, name='act5')

        conv5 = conv_cond_concat(conv5, yb)

        conv6 = tf.layers.conv2d_transpose(conv5, channel, kernel_size=[5,5], strides=[2,2], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv6')

        conv6 = tf.nn.tanh(conv6, name='act6')
        return conv6

def discriminator(images, y, phase, reuse=False):

    f2, f4, f8, f16 = 64, 128, 256, 512
    with tf.variable_scope('dis') as scope:

        if reuse == True:
            scope.reuse_variables()

        #Data shape is (128, 128, 3)
        yb = tf.reshape(y, shape=[batch_size, 1, 1, y_dim])

        #Concat label to tensor
        concat_data = conv_cond_concat(images,yb)

        conv1 = tf.layers.conv2d(concat_data, f2, kernel_size=[5,5], strides=[2,2], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv1')
        conv1 = tf.contrib.layers.batch_norm(conv1, is_training=phase, epsilon=1e-5, decay=0.9, updates_collections=None, scope='bn1')
        conv1 = tf.nn.leaky_relu(conv1, name='act1')

        conv1 = conv_cond_concat(conv1, yb)

        conv2 = tf.layers.conv2d(conv1, f4, kernel_size=[5,5], strides=[2,2], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv2')
        conv2 = tf.contrib.layers.batch_norm(conv2, is_training=phase, epsilon=1e-5, decay=0.9, updates_collections=None, scope='bn2')
        conv2 = tf.nn.leaky_relu(conv2, name='act2')

        conv2 = conv_cond_concat(conv2, yb)

        conv3 = tf.layers.conv2d(conv2, f8, kernel_size=[5,5], strides=[2,2], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv3')
        conv3 = tf.contrib.layers.batch_norm(conv3, is_training=phase, epsilon=1e-5, decay=0.9, updates_collections=None, scope='bn3')
        conv3 = tf.nn.leaky_relu(conv3, name='act3')

        conv3 = conv_cond_concat(conv3, yb)

        conv4 = tf.layers.conv2d(conv3, f16, kernel_size=[5,5], strides=[2,2], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv4')
        conv4 = tf.contrib.layers.batch_norm(conv4, is_training=phase, epsilon=1e-5, decay=0.9, updates_collections=None, scope='bn4')
        conv4 = tf.nn.leaky_relu(conv4, name='act4')

        dim = int(np.prod(conv4.get_shape()[1:]))
        flat = tf.reshape(conv4, shape=[-1,dim], name='flat')
        flat = tf.concat([flat, y], 1)

        weight = tf.get_variable('w', shape=[flat.shape[-1], 1], dtype='float', initializer=tf.truncated_normal_initializer(stddev=0.02))
        bias = tf.get_variable('b', shape=[1], dtype='float', initializer=tf.constant_initializer(0.0))
        logits = tf.add(tf.matmul(flat, weight), bias, name='logits')

        return tf.nn.sigmoid(logits), logits

if __name__ == "__main__":
    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'test':
        version = sys.argv[2]
        test(version)
