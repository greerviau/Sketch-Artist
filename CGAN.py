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
        self.max_epochs = 50
        self.d_itters = 1
        self.g_itters = 2
        self.save_samples = 1
        self.save_model = 10
        self.celebA = CelebA(self.output_size, self.channel, self.sample_size, self.batch_size, self.crop)
        self.z_dim = 100
        self.y_dim = self.celebA.y_dim
        self.version = 'face_gen_FINAL_test'
        self.log_dir = '/tmp/tensorflow_cgan/'+self.version
        self.model_dir = 'model/'
        self.sample_dir = 'samples/'
        self.test_dir = 'test/'

        self.real_images = tf.placeholder('float', shape=[self.batch_size,self.output_size, self.output_size, self.channel], name='real_images')
        self.z = tf.placeholder('float', shape=[self.batch_size,self.z_dim], name='noise_vec')
        self.y = tf.placeholder('float', shape=[self.batch_size,self.y_dim], name='condition_vec')

    def build_model(self):

        self.fake_images = self.generator(self.z, self.y)

        self.gen_sampler = self.sampler(self.z, self.y)

        real_result, real_logits = self.discriminator(self.real_images, self.y)

        fake_result, fake_logits = self.discriminator(self.fake_images, self.y, reuse=True)

        d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_result), logits=fake_logits))

        d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_result), logits=real_logits))

        self.d_loss = d_real_loss + d_fake_loss

        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_result), logits=fake_logits))

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'dis' in var.name]
        self.g_vars = [var for var in t_vars if 'gen' in var.name]

        self.saver = tf.train.Saver()

    def train(self):

        self.celebA.load_data()

        trainer_d = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
        trainer_g = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)

        batch_num = self.sample_size // self.batch_size

        with tf.Session() as sess:

            print('\nVersion: {}'.format(self.version))
            print('Crop: {}'.format(self.crop))
            print('Sample Size: {}'.format(self.sample_size))
            print('Max Epochs: {}'.format(self.max_epochs))
            print('Batch Size: {}'.format(self.batch_size))
            print('Batches per Epoch: {}'.format(batch_num))
            print('Starting training...\n')

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            start_epoch = 1
            sample_noise = None

            if os.path.exists(self.model_dir+self.version):
                self.saver.restore(sess, self.model_dir+self.version+'/'+self.version+'.ckpt')
                with open(self.model_dir+self.version+'/epoch.txt', 'r') as ep:
                    start_epoch = int(ep.read()) + 1
                    sample_noise = np.load(self.model_dir+self.version+'/sample_noise.npy')
            else:
                sample_noise = np.random.uniform(-1, 1, size=[self.batch_size, self.z_dim]).astype(np.float32)

            _, sample_labels = self.celebA.get_next_batch(0)
            #print(sample_labels)
            for epoch in range(start_epoch,self.max_epochs+1):

                dLoss_avg = []
                gLoss_avg = []

                for batch in range(batch_num):
                    train_noise = np.random.uniform(-1, 1, size=[self.batch_size, self.z_dim]).astype(np.float32)
                    train_images, real_labels = self.celebA.get_next_batch(batch)

                    for d in range(self.d_itters):
                        _, dLoss = sess.run([trainer_d, self.d_loss], feed_dict={self.z: train_noise, self.real_images: train_images, self.y: real_labels})
                        dLoss_avg.append(dLoss)

                    for g in range(self.g_itters):
                        _, gLoss = sess.run([trainer_g, self.g_loss], feed_dict={self.z: train_noise, self.y: real_labels})
                        gLoss_avg.append(gLoss)


                    print('\rEpoch {}/{} - Batch {}/{} - D_loss {:.3f} - G_loss {:.3f}   '.format(epoch, self.max_epochs, batch+1, batch_num, avg(dLoss_avg), avg(gLoss_avg)), end='')

                print('')
                if epoch%self.save_model == 0:
                    self.saver.save(sess,self.model_dir+self.version+'/'+self.version+'.ckpt')
                    with open(self.model_dir+self.version+'/epoch.txt', 'w') as ep:
                        ep.write(str(epoch))
                    np.save(self.model_dir+self.version+'/sample_noise.npy', sample_noise)
                    print('Model Saved | Epoch:[{}] | D_loss:[{:.2f}] | G_loss:[{:.2f}]'.format(epoch, dLoss, gLoss))

                if epoch%self.save_samples == 0:
                    if not os.path.exists(self.sample_dir+self.version):
                        os.makedirs(self.sample_dir+self.version)
                    imgtest = sess.run(self.gen_sampler, feed_dict={self.z: sample_noise,  self.y: sample_labels})
                    #print(imgtest.shape)
                    #print(imgtest[0])
                    #cv2.imshow('frame',imgtest[0])
                    #if cv2.waitKey(1) & 0xFF == ord('q'):
                        #break
                    imgtest = imgtest * 255.0
                    save_images(imgtest, [8,8], self.sample_dir+self.version+'/epoch_'+str(epoch)+'.jpg')

                    print('Sample Saved [epoch_{}.jpg]'.format(epoch))

            coord.request_stop()
            coord.join(threads)

    def test(self):

        path = self.model_dir+self.version
        if os.path.exists(path):
            with tf.Session() as sess:
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                self.saver.restore(sess, path+'/'+self.version+'.ckpt')
                sample_z = np.random.uniform(-1, 1, size=[self.batch_size, self.z_dim]).astype(np.float32)
                description = input('Enter a description --> ').lower()
                description_vec = self.celebA.text_to_vector(description)
                #print(description_vec)

                output = sess.run(self.gen_sampler, feed_dict={self.z: sample_z, self.y: description_vec})

                output = output * 255.0
                if not os.path.exists(self.test_dir+self.version):
                    os.makedirs(self.test_dir+self.version)

                save_images(output, [8,8], self.test_dir+self.version+'/{}.jpg'.format(description.replace(' ','_')))

                image = cv2.imread(self.test_dir+self.version+'/{}.jpg'.format(description.replace(' ','_')))

                cv2.imshow('test', image)
                if cv2.waitKey(0) and 0xFF == ord('q'):
                    cv2.destroyAllWindows()

                print('Test Finished')

                coord.request_stop()
                coord.join(threads)
        else:
            print('ERROR - [Model {} not found] - Path {}'.format(self.version, path))

    def discriminator(self, image, y, reuse=False):

        with tf.variable_scope('dis') as scope:
            k = 64

            if reuse == True:
                scope.reuse_variables()

            #Data shape is (128, 128, 3)
            yb = tf.reshape(y, shape=[self.batch_size, 1, 1, self.y_dim])

            conv1 = conv2d(image, k, name='d_conv1')
            conv1 = batch_norm(conv1, scope='d_conv1_bn')
            conv1 = tf.nn.leaky_relu(conv1, name='d_conv1_act')

            conv2 = conv2d(conv1, k*2, name='d_conv2')
            conv2 = batch_norm(conv2, scope='d_conv2_bn')
            conv2 = tf.nn.leaky_relu(conv2, name='d_conv2_act')

            conv3 = conv2d(conv2, k*4, name='d_conv3')
            conv3 = batch_norm(conv3, scope='d_conv3_bn')
            conv3 = tf.nn.leaky_relu(conv3, name='d_conv3_act')

            conv4 = conv2d(conv3, k*8, name='d_conv4')
            conv4 = batch_norm(conv4, scope='d_conv4_bn')
            conv4 = tf.nn.leaky_relu(conv4, name='d_conv4_act')

            flat = tf.reshape(conv4, [self.batch_size, -1])
            flat = tf.concat([flat, y] ,1)

            full1 = fully_connected(flat, 1024, 'd_full1')
            full1 = tf.nn.relu(full1, name='d_full1_act')

            full2 = fully_connected(full1, 1, 'd_full2')

            return tf.nn.sigmoid(full2, name='d_full2_act'), full2

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

            full1 = fully_connected(z, k*8*s_h16*s_w16, 'g_full1')
            full1 = tf.nn.relu(full1, name='g_full1_act1')
            full1 = batch_norm(full1, scope='g_full1_bn')
            full1 = tf.nn.leaky_relu(full1, name='g_full1_act2')

            conv1 = tf.reshape(full1, shape=[self.batch_size, s_h16, s_h16, k*8], name='g_conv1')

            conv2 = deconv2d(conv1, [self.batch_size, s_h8, s_w8, k*4], name='g_conv2')
            conv2 = batch_norm(conv2, scope='g_conv2_bn')
            conv2 = tf.nn.leaky_relu(conv2, name='g_conv2_act')

            conv3 = deconv2d(conv2, [self.batch_size, s_h4, s_w4, k*2], name='g_conv3')
            conv3 = batch_norm(conv3,  scope='g_conv3_bn')
            conv3 = tf.nn.leaky_relu(conv3, name='g_conv3_act')

            conv4 = deconv2d(conv3, [self.batch_size, s_h2, s_w2, k], name='g_conv4')
            conv4 = batch_norm(conv4, scope='g_conv4_bn')
            conv4 = tf.nn.leaky_relu(conv4, name='g_conv4_act')

            conv5 = deconv2d(conv4, [self.batch_size, s_h, s_w, self.channel], name='g_conv5')

            conv5 = tf.nn.tanh(conv5, name='g_conv5_act')

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

            full1 = fully_connected(z, k*8*s_h16*s_w16, 'g_full1')
            full1 = tf.nn.relu(full1, name='g_full1_act1')
            full1 = batch_norm(full1, scope='g_full1_bn')
            full1 = tf.nn.leaky_relu(full1, name='g_full1_act2')

            conv1 = tf.reshape(full1, shape=[self.batch_size, s_h16, s_h16, k*8], name='g_conv1')

            conv2 = deconv2d(conv1, [self.batch_size, s_h8, s_w8, k*4], name='g_conv2')
            conv2 = batch_norm(conv2, scope='g_conv2_bn')
            conv2 = tf.nn.leaky_relu(conv2, name='g_conv2_act')

            conv3 = deconv2d(conv2, [self.batch_size, s_h4, s_w4, k*2], name='g_conv3')
            conv3 = batch_norm(conv3,  scope='g_conv3_bn')
            conv3 = tf.nn.leaky_relu(conv3, name='g_conv3_act')

            conv4 = deconv2d(conv3, [self.batch_size, s_h2, s_w2, k], name='g_conv4')
            conv4 = batch_norm(conv4, scope='g_conv4_bn')
            conv4 = tf.nn.leaky_relu(conv4, name='g_conv4_act')

            conv5 = deconv2d(conv4, [self.batch_size, s_h, s_w, self.channel], name='g_conv5')

            conv5 = tf.nn.tanh(conv5, name='g_conv5_act')
            return conv5

if __name__ == "__main__":
    cgan = CGAN()
    cgan.build_model()
    if sys.argv[1] == 'train':
        cgan.train()
    elif sys.argv[1] == 'test':
        cgan.test()
