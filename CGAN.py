import os, cv2, sys
from utils import *
from ops import *
import tensorflow as tf
import numpy as np

class CGAN(object):

    def __init__(self):
        self.sample_size = 20000
        self.output_size = 64
        self.crop = True
        self.channel = 3
        self.learning_rate = 0.0002
        self.batch_size = 64
        self.max_epochs = 1000
        self.save_samples = 5
        self.save_model = 10
        self.celebA = CelebA(self.output_size, self.channel, self.sample_size, self.batch_size, self.crop)
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

    def build_model(self):

        self.fake_images = self.generator(self.z, self.y)

        self.gen_sampler = self.sampler(self.z, self.y)

        real_result, real_logits = self.discriminator(self.images, self.y)

        fake_result, fake_logits = self.discriminator(self.fake_images, self.y, reuse=True)

        d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_result), logits=fake_logits))

        d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_result), logits=real_logits))

        self.d_loss = d_real_loss + d_fake_loss

        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_result), logits=fake_logits))

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'dis' in var.name]
        self.g_vars = [var for var in t_vars if 'gen' in var.name]

        self.d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.d_vars]

        self.saver = tf.train.Saver()

    def train(self):

        self.celebA.load_data()

        trainer_d = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=self.d_vars)
        trainer_g = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=self.g_vars)

        batch_num = self.sample_size // self.batch_size

        with tf.Session() as sess:

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            start_epoch = 1

            if os.path.exists(self.model_dir+self.version):
                self.saver.restore(sess, self.model_dir+self.version+'/'+self.version+'.ckpt')
                with open(self.model_dir+self.version+'/epoch.txt', 'r') as ep:
                    start_epoch = int(ep.read()) + 1

            print('\nVersion: {}'.format(self.version))
            print('Crop: {}'.format(self.crop))
            print('Sample Size: {}'.format(self.sample_size))
            print('Max Epochs: {}'.format(self.max_epochs))
            print('Batch Size: {}'.format(self.batch_size))
            print('Batches per Epoch: {}'.format(batch_num))
            print('Starting training...\n')
            sample_noise = np.random.uniform(-1, 1, size=[self.batch_size, self.z_dim]).astype(np.float32)
            _, sample_labels = self.celebA.get_next_batch(0)
            for i in range(start_epoch,self.max_epochs+1):
                for j in range(batch_num):
                    train_noise = np.random.uniform(-1, 1, size=[self.batch_size, self.z_dim]).astype(np.float32)
                    train_images, real_labels = self.celebA.get_next_batch(i-1)

                    d_itters = 1
                    for _ in range(d_itters):
                        _, dLoss = sess.run([trainer_d, self.d_loss], feed_dict={self.z: train_noise, self.images: train_images, self.y: real_labels})
                        sess.run(self.d_clip)

                    g_itters = 1
                    for _ in range(g_itters):
                        _, gLoss = sess.run([trainer_g, self.g_loss], feed_dict={self.z: train_noise, self.y: real_labels})

                    print('\rEpoch {}/{} - Batch {}/{} - D_loss {:.2f} - G_loss {:.2f}'.format(i, self.max_epochs, j+1, batch_num, dLoss, gLoss),end='')

                print('')
                if i%self.save_model == 0:
                    self.saver.save(sess,self.model_dir+self.version+'/'+self.version+'.ckpt')
                    with open(self.model_dir+self.version+'/epoch.txt', 'w') as ep:
                        ep.write(str(i))
                    print('Model Saved | Epoch:[{}] | D_loss:[{:.2f}] | G_loss:[{:.2f}]'.format(i, dLoss, gLoss))

                if i%self.save_samples == 0:
                    if not os.path.exists(self.sample_dir+self.version):
                        os.makedirs(self.sample_dir+self.version)
                    imgtest = sess.run(self.gen_sampler, feed_dict={self.z: sample_noise,  self.y: sample_labels})
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

    def test(self):

        path = self.model_dir+self.version
        if os.path.exists(path):
            init = tf.initialize_all_variables()
            with tf.Session() as sess:
                sess.run(init)

                self.saver.restore(sess, path+'/'+self.version+'.ckpt')
                sample_z = np.random.uniform(-1, 1, size=[self.batch_size, self.z_dim]).astype(np.float32)
                description = input('Enter a description --> ').lower()
                description_vec = self.celebA.text_to_vector(description)

                output = sess.run(self.fake_images, feed_dict={self.z: sample_z, self.y: description_vec, self.phase: False})

                output = output * 255.0
                if not os.path.exists(self.test_dir+self.version):
                    os.makedirs(self.test_dir+self.version)

                save_images(output, [8,8], self.test_dir+self.version+'/{}.jpg'.format(description.replace(' ','_')))

                image = cv2.imread(self.test_dir+self.version+'/{}.jpg'.format(description.replace(' ','_')))

                cv2.imshow('test', image)
                if cv2.waitKey(0) and 0xFF == ord('q'):
                    cv2.destroyAllWindows()

                print('Test Finished')
        else:
            print('ERROR - [Model {} not found] - Path {}'.format(self.version, path))

    def discriminator(self, images, y, reuse=False):

        with tf.variable_scope('dis') as scope:
            k = 64

            if reuse == True:
                scope.reuse_variables()

            #Data shape is (128, 128, 3)
            yb = tf.reshape(y, shape=[self.batch_size, 1, 1, self.y_dim])

            #Concat label to tensor
            concat_data = conv_cond_concat(images, yb)

            conv1 = conv2d(concat_data, k, name='d_conv1')
            conv1 = tf.nn.leaky_relu(conv1, name='d_act1')

            conv1 = conv_cond_concat(conv1, yb)

            conv2 = conv2d(conv1, k*2, name='d_conv2')
            conv2 = batch_norm(conv2, scope='d_bn2')
            conv2 = tf.nn.leaky_relu(conv2, name='d_act2')

            conv2 = conv_cond_concat(conv2, yb)

            conv3 = conv2d(conv2, k*4, name='d_conv3')
            conv3 = batch_norm(conv3, scope='d_bn3')
            conv3 = tf.nn.leaky_relu(conv3, name='d_act3')

            conv3 = conv_cond_concat(conv3, yb)

            conv4 = conv2d(conv3, k*8, name='d_conv4')
            conv4 = batch_norm(conv4, scope='d_bn4')
            conv4 = tf.nn.leaky_relu(conv4, name='d_act4')

            flat = fully_connected(tf.reshape(conv4, [self.batch_size, -1]), 1, 'd_full')

            return tf.nn.sigmoid(flat), flat

    def generator(self, z, y):

        with tf.variable_scope('gen') as scope:

            k = 64
            s_h, s_w = self.output_size, self.output_size
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            yb = tf.reshape(y, shape=[self.batch_size, 1, 1, self.y_dim])
            z = tf.concat([z, y], 1)

            flat = fully_connected(z, k*8*s_h16*s_w16, 'g_flat', with_w=False)

            conv1 = tf.reshape(flat, shape=[self.batch_size, s_h16, s_h16, k*8], name='g_conv1')
            conv1 = batch_norm(conv1, scope='g_bn1')
            conv1 = tf.nn.relu(conv1, name='g_act1')

            #Concat label to tensor
            conv1 = conv_cond_concat(conv1, yb)

            conv2 = deconv2d(conv1, [self.batch_size, s_h8, s_w8, k*4], name='g_conv2', with_w=False)
            conv2 = batch_norm(conv2, scope='g_bn2')
            conv2 = tf.nn.relu(conv2, name='g_act2')

            conv2 = conv_cond_concat(conv2, yb)

            conv3 = deconv2d(conv2, [self.batch_size, s_h4, s_w4, k*2], name='g_conv3', with_w=False)
            conv3 = batch_norm(conv3,  scope='g_bn3')
            conv3 = tf.nn.relu(conv3, name='g_act3')

            conv3 = conv_cond_concat(conv3, yb)

            conv4 = deconv2d(conv3, [self.batch_size, s_h2, s_w2, k], name='g_conv4', with_w=False)
            conv4 = batch_norm(conv4, scope='g_bn4')
            conv4 = tf.nn.relu(conv4, name='g_act4')

            conv4 = conv_cond_concat(conv4, yb)

            conv5 = deconv2d(conv4, [self.batch_size, s_h, s_w, self.channel], name='g_conv5', with_w=False)

            conv5 = tf.nn.tanh(conv5, name='g_act5')
            return conv5

    def sampler(self, z, y):

        with tf.variable_scope('gen') as scope:
            scope.reuse_variables()

            k = 64
            s_h, s_w = self.output_size, self.output_size
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            yb = tf.reshape(y, shape=[self.batch_size, 1, 1, self.y_dim])
            z = tf.concat([z, y], 1)

            flat = fully_connected(z, k*8*s_h16*s_w16, 'g_flat', with_w=False)

            conv1 = tf.reshape(flat, shape=[self.batch_size, s_h16, s_h16, k*8], name='g_conv1')
            conv1 = batch_norm(conv1, scope='g_bn1', train=False)
            conv1 = tf.nn.relu(conv1, name='g_act1')

            #Concat label to tensor
            conv1 = conv_cond_concat(conv1, yb)

            conv2 = deconv2d(conv1, [self.batch_size, s_h8, s_w8, k*4], name='g_conv2', with_w=False)
            conv2 = batch_norm(conv2, scope='g_bn2', train=False)
            conv2 = tf.nn.relu(conv2, name='g_act2')

            conv2 = conv_cond_concat(conv2, yb)

            conv3 = deconv2d(conv2, [self.batch_size, s_h4, s_w4, k*2], name='g_conv3', with_w=False)
            conv3 = batch_norm(conv3,  scope='g_bn3', train=False)
            conv3 = tf.nn.relu(conv3, name='g_act3')

            conv3 = conv_cond_concat(conv3, yb)

            conv4 = deconv2d(conv3, [self.batch_size, s_h2, s_w2, k], name='g_conv4', with_w=False)
            conv4 = batch_norm(conv4, scope='g_bn4', train=False)
            conv4 = tf.nn.relu(conv4, name='g_act4')

            conv4 = conv_cond_concat(conv4, yb)

            conv5 = deconv2d(conv4, [self.batch_size, s_h, s_w, self.channel], name='g_conv5', with_w=False)

            conv5 = tf.nn.tanh(conv5, name='g_act5')
            return conv5

if __name__ == "__main__":
    cgan = CGAN()
    cgan.build_model()
    if sys.argv[1] == 'train':
        cgan.train()
    elif sys.argv[1] == 'test':
        cgan.test()
