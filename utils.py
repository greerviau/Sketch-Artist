import os, cv2, random
import numpy as np
import tensorflow as tf
import pandas as pd

class CelebA(object):

    def __init__(self, op_size, channel, sample_size, batch_size, crop, filter, data_dir='E:/USB Backup/Data/celeba/'):

        self.dataname = 'CelebA'
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.crop = crop
        self.filter = filter
        self.dims = op_size*op_size
        self.shape = [op_size,op_size,channel]
        self.image_size = op_size
        self.data_dir = data_dir
        self.y_dim = 5
        self.data_file = 'list_attr_celeba.csv'

    def load_data(self):

        images_dir = os.path.join(self.data_dir,'img_align_celeba')

        cur_dir = os.getcwd()

        X = []
        y = []

        faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        data = pd.read_csv(os.path.join(self.data_dir,self.data_file))

        i = 0
        count = 0
        print('\n===LOADING DATA===')
        while count < self.sample_size:
            img = data['image_id'][i]
            print('\rLoading: {} - Loaded: {}'.format(img, count), end='')
            image = cv2.imread(os.path.join(images_dir,img))
            if self.crop:
                h, w, c = image.shape
                #crop 4/6ths of the image
                cr_h = h//6
                cr_w = w//6
                crop_image = image[cr_h:h-cr_h,cr_w:w-cr_w]
                image = crop_image
            image = cv2.resize(image, (self.image_size, self.image_size))
            face = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
            if type(face) is np.ndarray:
                features = np.zeros(self.y_dim)
                features[0] = int(data['Black_Hair'][i])        #Black hair
                features[1] = int(data['Brown_Hair'][i])        #Brown hair
                features[2] = int(data['Blond_Hair'][i])        #Blonde hair
                features[3] = int(data['Male'][i])              #Male
                features[4] = int(data['No_Beard'][i]) * -1     #Beard (invert because in dataset, positive 1 represents no beard)
                if sum([1 for i in features[:3] if i == 1]) == 1:
                    X.append(image)
                    y.append(features)
                    count+=1
            i+=1

        print('\n\n===DATA STATS===')
        print('Black Hair: ', sum([ 1 for i in y if i[0] == 1]))
        print('Brown Hair: ', sum([ 1 for i in y if i[1] == 1]))
        print('Blonde Hair: ', sum([ 1 for i in y if i[2] == 1]))
        print('Male: ', sum([ 1 for i in y if i[3] == 1]))
        print('Beard: ', sum([ 1 for i in y if i[4] == 1]))

        X = np.array(X)
        y = np.array(y)

        seed = 547

        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

        self.data = X / 255.
        self.data_y = y

    def get_next_batch(self, iter_num):
        ro_num = self.sample_size // self.batch_size - 1

        if iter_num % ro_num == 0:
            length = len(self.data)
            perm = np.arange(length)
            np.random.shuffle(perm)
            self.data = np.array(self.data)
            self.data = self.data[perm]
            self.data_y = np.array(self.data_y)
            self.data_y = self.data_y[perm]

        return self.data[int(iter_num % ro_num) * self.batch_size: int(iter_num % ro_num + 1) * self.batch_size], self.data_y[int(iter_num % ro_num) * self.batch_size: int(iter_num % ro_num + 1) * self.batch_size]

    def text_to_vector(self, text):
        text = text.lower()
        key_words = ['black hair',
                    'brown_hair',
                    'blonde hair',
                    'male',
                    'beard']
        vec = np.ones(self.y_dim)*-1
        for i, key in enumerate(key_words, 0):
            if key in text:
                vec[i] = 1
        #print(vec)
        batch_vector = np.tile(vec,(self.batch_size,1))
        return batch_vector

    def save(self, dir):
        np.save(dir+'/data.npy', self.data)
        np.save(dir+'/data_y.npy', self.data_y)

    def load(self, dir):
        self.data = np.load(dir+'/data.npy')
        self.data_y = np.load(dir+'/data_y.npy')

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter must have dimensions: HxW or HxWx3 or HxWx4')

def inverse_transform(images):
    return (images+1.)/2.

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return cv2.imwrite(path, image)

def avg(list):
    return sum(list)/len(list)
