import os, cv2, sys
from utils import *
from ops import *
import tensorflow as tf
import numpy as np

class CGAN(object):

    def __init__(self):
        self.sample_size = 5000
        self.output_size = 64
        self.channel = 3
        self.learning_rate = 0.0002
        self.batch_size = 64
        self.max_epochs = 5000
        self.celebA = CelebA(self.output_size, self.channel, self.sample_size, self.batch_size)
        self.z_dim = 100
        self.y_dim = self.celebA.y_dim
        self.version = 'face_gen_v7'
        self.log_dir = '/tmp/tensorflow_cgan/'+self.version
        self.model_dir = 'model/'
        self.sample_dir = 'samples/'
        self.test_dir = 'test/'

        self.images = tf.placeholder('float', shape=[self.batch_size,self.output_size, self.output_size, self.channel], name='real_images')
        self.z = tf.placeholder('float', shape=[self.batch_size,self.z_dim], name='noise_vec')
        self.y = tf.placeholder('float', shape=[self.batch_size,self.y_dim], name='condition_vec')
        self.phase = tf.placeholder('bool', name='self.phase')

    def build_model(self):

        self.fake_images = self.generator(self.z, self.y, self.phase)

        real_result, real_logits = self.discriminator(self.images, self.y, self.phase)
        
        fake_result, fake_logits = self.discriminator(self.fake_images, self.y, self.phase, reuse=True)

        d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_result), logits=fake_logits))

        d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_result), logits=real_logits))

        g_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_result), logits=fake_logits))

        self.d_loss = d_real_loss + d_fake_loss
        self.g_loss = g_fake_loss

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'dis' in var.name]
        self.g_vars = [var for var in t_vars if 'gen' in var.name]

        #self.d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.d_vars]

        self.saver = tf.train.Saver()

    def train(self):

        self.celebA.load_data()

        trainer_d = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
        trainer_g = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)

        batch_num = self.sample_size // self.batch_size

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            start_epoch = 1

            if os.path.exists(self.model_dir+self.version):
                self.saver.restore(sess, self.model_dir+self.version+'/'+self.version+'.ckpt')
                with open(self.model_dir+self.version+'/epoch.txt', 'r') as ep:
                    start_epoch = int(ep.read()) + 1

            print('\nVersion: {}'.format(self.version))
            print('Sample Size: {}'.format(self.sample_size))
            print('Max Epochs: {}'.format(self.max_epochs))
            print('Batch Size: {}'.format(self.batch_size))
            print('Batches per Epoch: {}'.format(batch_num))
            print('Starting training...\n')
            sample_noise = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.z_dim]).astype(np.float32)
            _, sample_labels = self.celebA.get_next_batch(0)
            for i in range(start_epoch,self.max_epochs+1):
                for j in range(batch_num):
                    train_noise = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.z_dim]).astype(np.float32)
                    train_images, real_labels = self.celebA.get_next_batch(i-1)

                    print('\rEpoch {}/{} - Batch {}/{}'.format(i, self.max_epochs, j+1, batch_num),end='')

                    _, dLoss = sess.run([trainer_d, self.d_loss], feed_dict={self.z: train_noise, self.images: train_images, self.y: real_labels, self.phase: True})

                    #Update generator twice to avoid discriminator convergence
                    _, gLoss = sess.run([trainer_g, self.g_loss], feed_dict={self.z: train_noise, self.y: real_labels, self.phase: True})
                    _, gLoss = sess.run([trainer_g, self.g_loss], feed_dict={self.z: train_noise, self.y: real_labels, self.phase: True})

                print('')
                if i%50 == 0:
                    self.saver.save(sess,self.model_dir+self.version+'/'+self.version+'.ckpt')
                    with open(self.model_dir+self.version+'/epoch.txt', 'w') as ep:
                        ep.write(str(i))
                    print('Model Saved | train:[{}] | self.d_loss: {:.2f} | self.g_loss: {:.2f}'.format(i, dLoss, gLoss))

                if i%5 == 0:
                    if not os.path.exists(self.sample_dir+self.version):
                        os.makedirs(self.sample_dir+self.version)
                    imgtest = sess.run(self.fake_images, feed_dict={self.z: sample_noise, self.y: sample_labels, self.phase: False})
                    #print(imgtest.shape)
                    #print(imgtest[0])
                    #cv2.imshow('frame',imgtest[0])
                    #if cv2.waitKey(1) & 0xFF == ord('q'):
                        #break
                    imgtest = imgtest * 255.0
                    save_images(imgtest, [8,8], self.sample_dir+self.version+'/epoch_'+str(i)+'.jpg')

                    print('Sample Saved [epoch_{}.jpg]'.format(i))

            coord.request_stop()
            coord.join(threads)

    def test(self, version):

        path = self.model_dir+version
        if os.path.exists(path):
            init = tf.initialize_all_variables()
            with tf.Session() as sess:
                sess.run(init)

                self.saver.restore(sess, path+'/'+version+'.ckpt')
                sample_z = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.z_dim]).astype(np.float32)
                description = input('Enter a description --> ').lower()
                description_vec = self.celebA.text_to_vector(description)

                output = sess.run(self.fake_images, feed_dict={self.z: sample_z, self.y: description_vec, self.phase: False})

                output = output * 255.0
                if not os.path.exists(self.test_dir+version):
                    os.makedirs(self.test_dir+self.version)

                save_images(output, [8,8], self.test_dir+version+'/{}.jpg'.format(description.replace(' ','_')))

                image = cv2.imread(self.test_dir+version+'/{}.jpg'.format(description.replace(' ','_')))

                cv2.imshow('test', image)
                if cv2.waitKey(0) and 0xFF == ord('q'):
                    cv2.destroyAllWindows()

                print('Test Finished')
        else:
            print('ERROR - [Model {} not found] - Path {}'.format(self.version, path))

    def generator(self, z, y, phase):

        f4, f8, f16, f32 = 512, 256, 128, 64
        s = 4

        with tf.variable_scope('gen') as scope:
            yb = tf.reshape(y, shape=[self.batch_size, 1, 1, self.y_dim])
            z = tf.concat([z, y], 1)

            weight = tf.get_variable('w', shape=[self.z_dim+self.y_dim, s * s * f4], dtype='float', initializer=tf.truncated_normal_initializer(stddev=0.02))
            bias = tf.get_variable('b', shape=[f4 * s * s], dtype='float', initializer=tf.constant_initializer(0.0))
            flat = tf.add(tf.matmul(z, weight), bias, name='flat')

            conv1 = tf.reshape(flat, shape=[self.batch_size, s, s, f4], name='conv1')
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
            conv3 = tf.nn.leaky_relu(conv3, name='act3')

            conv3 = conv_cond_concat(conv3, yb)

            conv4 = tf.layers.conv2d_transpose(conv3, f32, kernel_size=[5,5], strides=[2,2], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv4')
            conv4 = tf.contrib.layers.batch_norm(conv4, is_training=phase, epsilon=1e-5, decay=0.9, updates_collections=None, scope='bn4')
            conv4 = tf.nn.leaky_relu(conv4, name='act4')

            conv4 = conv_cond_concat(conv4, yb)

            conv5 = tf.layers.conv2d_transpose(conv4, self.channel, kernel_size=[5,5], strides=[2,2], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv6')

            conv5 = tf.nn.tanh(conv5, name='act6')
            return conv5

    def discriminator(self, images, y, phase, reuse=False):

        f2, f4, f8, f16 = 64, 128, 256, 512
        with tf.variable_scope('dis') as scope:

            if reuse == True:
                scope.reuse_variables()

            #Data shape is (128, 128, 3)
            yb = tf.reshape(y, shape=[self.batch_size, 1, 1, self.y_dim])

            #Concat label to tensor
            concat_data = conv_cond_concat(images, yb)

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
    cgan = CGAN()
    cgan.build_model()
    if sys.argv[1] == 'train':
        cgan.train()
    elif sys.argv[1] == 'test':
        version = sys.argv[2]
        cgan.test(version)
