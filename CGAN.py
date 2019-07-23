import os, cv2, sys
from utils import *
from ops import *
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tkinter import *
from PIL import ImageTk, Image

class CGAN(object):

    def __init__(self):
        self.sample_size = 20000
        self.output_size = 64
        self.crop = True
        self.filter = True
        self.channel = 3
        self.learning_rate = 0.0002
        self.batch_size = 64
        self.max_epochs = 100
        self.d_itters = 1
        self.g_itters = 1
        self.save_mode = 2  #1 = every epoch    2 = every 5 batches
        self.save_model = 10
        self.celebA = CelebA(self.output_size, self.channel, self.sample_size, self.batch_size, self.crop, self.filter)
        self.z_dim = 100
        self.y_dim = self.celebA.y_dim
        self.version = 'face_gen_per_batch_filtered_4'
        self.log_dir = '/tmp/tensorflow_cgan/'+self.version
        self.model_dir = 'model/'
        self.sample_dir = 'samples/'
        self.test_dir = 'test/'
        self.sequence_dir = 'image_sequence/'

        self.real_images = tf.placeholder('float', shape=[self.batch_size,self.output_size, self.output_size, self.channel], name='real_images')
        self.z = tf.placeholder('float', shape=[self.batch_size,self.z_dim], name='noise_vec')
        self.y = tf.placeholder('float', shape=[self.batch_size,self.y_dim], name='condition_vec')

    def build_model(self):

        self.fake_images, self.rec_prediction = self.generator(self.z, self.y)

        self.gen_sampler = self.sampler(self.z, self.y)

        real_result, real_logits = self.discriminator(self.real_images, self.y)

        fake_result, fake_logits = self.discriminator(self.fake_images, self.y, reuse=True)

        d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_result), logits=fake_logits))

        d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_result), logits=real_logits))

        self.d_loss = d_real_loss + d_fake_loss

        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_result), logits=fake_logits))

        self.z_loss = tf.reduce_mean(tf.square(tf.concat([self.z, self.y], 1) - self.rec_prediction), name='z_prediction_loss')

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'dis' in var.name]
        self.g_vars = [var for var in t_vars if 'gen' in var.name]

        self.saver = tf.train.Saver()

    def train(self):

        trainer_d = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
        trainer_g = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)
        trainer_z = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(self.z_loss, var_list=self.g_vars)

        batch_num = self.sample_size // self.batch_size

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            start_epoch = 1
            sample_noise = None
            sample_labels = None

            if os.path.exists(self.model_dir+self.version):
                self.saver.restore(sess, self.model_dir+self.version+'/'+self.version+'.ckpt')
                with open(self.model_dir+self.version+'/epoch.txt', 'r') as ep:
                    start_epoch = int(ep.read()) + 1
                self.celebA.load(self.model_dir+self.version)
                sample_noise = np.load(self.model_dir+self.version+'/sample_noise.npy')
                sample_labels = np.load(self.model_dir+self.version+'/sample_labels.npy')
                print('\n===CHECKPOINT RESTORED===')
            else:
                os.makedirs(self.model_dir+self.version)
                self.celebA.load_data()
                self.celebA.save(self.model_dir+self.version)
                sample_noise = np.random.uniform(-1, 1, size=[self.batch_size, self.z_dim]).astype(np.float32)
                np.save(self.model_dir+self.version+'/sample_noise.npy', sample_noise)
                _, sample_labels = self.celebA.get_next_batch(0)
                np.save(self.model_dir+self.version+'/sample_labels.npy', sample_labels)

            #print(sample_labels)
            print('\n===HYPER PARAMS===')
            print('Version: {}'.format(self.version))
            print('Crop: {}'.format(self.crop))
            print('Filter: {}'.format(self.filter))
            print('Sample Size: {}'.format(self.sample_size))
            print('Max Epochs: {}'.format(self.max_epochs))
            print('Batch Size: {}'.format(self.batch_size))
            print('Batches per Epoch: {}'.format(batch_num))
            print('Starting training...\n')

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
                        sess.run([trainer_z], feed_dict={self.z: train_noise, self.y: real_labels})
                        gLoss_avg.append(gLoss)

                    print('\rEpoch {}/{} - Batch {}/{} - D_loss {:.3f} - G_loss {:.3f}   '.format(epoch, self.max_epochs, batch+1, batch_num, avg(dLoss_avg), avg(gLoss_avg)), end='')

                    if self.save_mode == 2 and batch%5 == 0:
                        if not os.path.exists(self.sample_dir+self.version):
                            os.makedirs(self.sample_dir+self.version)
                        imgtest = sess.run(self.gen_sampler, feed_dict={self.z: sample_noise,  self.y: sample_labels})
                        imgtest = imgtest * 255.0
                        save_images(imgtest, [8,8], self.sample_dir+self.version+'/epoch_'+str(epoch)+'_batch_'+str(batch)+'.jpg')

                print('')
                if epoch%self.save_model == 0:
                    self.saver.save(sess,self.model_dir+self.version+'/'+self.version+'.ckpt')
                    with open(self.model_dir+self.version+'/epoch.txt', 'w') as ep:
                        ep.write(str(epoch))
                    print('Model Saved | Epoch:[{}] | D_loss:[{:.2f}] | G_loss:[{:.2f}]'.format(epoch, avg(dLoss_avg), avg(gLoss_avg)))

                if self.save_mode == 1 and epoch%1 == 0:
                    if not os.path.exists(self.sample_dir+self.version):
                        os.makedirs(self.sample_dir+self.version)
                    imgtest = sess.run(self.gen_sampler, feed_dict={self.z: sample_noise,  self.y: sample_labels})
                    imgtest = imgtest * 255.0
                    save_images(imgtest, [8,8], self.sample_dir+self.version+'/epoch_'+str(epoch)+'.jpg')

                    print('Sample Saved [epoch_{}.jpg]'.format(epoch))

    def test(self):

        path = self.model_dir+self.version

        if os.path.exists(path):

            with tf.Session() as sess:

                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                self.saver.restore(sess, path+'/'+self.version+'.ckpt')

                def enter_button(ent):
                    description = ent.get().lower()
                    if description != '':
                        sample_z = np.random.uniform(-1, 1, size=[self.batch_size, self.z_dim]).astype(np.float32)
                        #description = input('Enter a description --> ').lower()
                        description_vec = self.celebA.text_to_vector(description)
                        #print(description_vec)

                        output = sess.run(self.gen_sampler, feed_dict={self.z: sample_z, self.y: description_vec})

                        output = output * 255.0
                        if not os.path.exists(self.test_dir+self.version):
                            os.makedirs(self.test_dir+self.version)

                        save_images(output, [8,8], self.test_dir+self.version+'/{}.jpg'.format(description.replace(' ','_')))

                        image = ImageTk.PhotoImage(Image.open(self.test_dir+self.version+'/{}.jpg'.format(description.replace(' ','_'))))

                        ent.delete(0, 'end')

                        panel.configure(image=image)
                        panel.image = image
                    else:
                        print('No Description Given')

                window = Tk()
                window.title('Sketch Artist')
                window.configure(background='grey')

                img = ImageTk.PhotoImage(Image.new('RGB', (512, 512)))
                panel = Label(window, image = img)
                panel.pack(side = 'top')

                but = Button(window, text='Generate Faces', command=lambda:enter_button(ent))
                but.pack(side = 'bottom')

                ent = Entry(window, width=50)
                ent.pack(side = 'bottom')

                mainloop()
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
            full1_act = tf.nn.leaky_relu(full1, name='g_full1_act2')

            conv1 = tf.reshape(full1_act, shape=[self.batch_size, s_h16, s_h16, k*8], name='g_conv1')

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

            #Auto encoder to predict noise
            z_pred = fully_connected(full1, self.z_dim+self.y_dim, 'z_full')
            z_pred = tf.nn.tanh(z_pred, name='z_full_act')

            return conv5, z_pred

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
            full1_act = tf.nn.leaky_relu(full1, name='g_full1_act2')

            conv1 = tf.reshape(full1_act, shape=[self.batch_size, s_h16, s_h16, k*8], name='g_conv1')

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

    def to_image_sequence(self):
        samples_dir = self.sample_dir+self.version
        dir = self.sequence_dir+self.version
        if not os.path.exists(dir):
        	os.makedirs(dir)
        count = 0
        for i in range(1,101):
            for j in range(0,311,5):
                img_name = 'epoch_{}_batch_{}.jpg'.format(i,j)
                print(img_name)
                img = cv2.imread(os.path.join(samples_dir,img_name))
                cv2.imwrite(os.path.join(dir,"frame_{:05d}.jpg".format(count)),img)
                count+=1

if __name__ == "__main__":
    cgan = CGAN()
    cgan.build_model()
    if sys.argv[1] == 'train':
        cgan.train()
    elif sys.argv[1] == 'test':
        cgan.test()
    elif sys.argv[1] == 'format':
        cgan.to_image_sequence()
