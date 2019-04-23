import os, csv, cv2, random
import numpy as np
import tensorflow as tf

class CelebA(object):

    def __init__(self, op_size, channel, sample_size, batch_size, crop, data_dir='D:/Data/celeba/'):

        self.dataname = 'CelebA'
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.crop = crop
        self.dims = op_size*op_size
        self.shape = [op_size,op_size,channel]
        self.image_size = op_size
        self.data_dir = data_dir
        self.y_dim = 6

    def load_data(self):
        cur_dir = os.getcwd()

        X = []
        y = []

        with open(os.path.join(self.data_dir,'list_attr_celeba.csv')) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            data = []
            for row in readCSV:
                data.append(row)
            #print(data[0])
            del data[0]
            images_dir = os.path.join(self.data_dir,'img_align_celeba')
            for i in range(self.sample_size):
                img = data[i][0]
                print('\rLoading: {}'.format(img), end='')
                image = cv2.imread(os.path.join(images_dir,img))
                if self.crop:
                    h, w, c = image.shape
                    #crop 4/6ths of the image
                    cr_h = h//6
                    cr_w = w//6
                    crop_image = image[cr_h:h-cr_h,cr_w:w-cr_w]
                    image = crop_image
                image = cv2.resize(image, (self.image_size, self.image_size))
                X.append(image)
                features = np.zeros(self.y_dim)
                #features[0] = int(data[i][5])        #Bald
                features[0] = int(data[i][9])        #Black hair
                features[1] = int(data[i][10])       #Blond hair
                features[2] = int(data[i][12])       #Brown hair
                #features[3] = int(data[i][18])       #Gray hair
                features[3] = int(data[i][16])       #Glasses
                #features[5] = int(data[i][17])       #Goatee
                features[4] = int(data[i][21])       #Male
                #features[7] = int(data[i][23])       #Mustache
                features[5] = int(data[i][25]) * -1  #Beard (invert because in dataset, positive 1 represents no beard)
                #features[9] = int(data[i][27])     #Pale skin
                y.append(features)

        print('')
        X = np.array(X)
        y = np.array(y)

        seed = 547

        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

        self.data = X / 255.
        self.data_y = y
        #print(self.data_y[0])
        #cv2.imshow('frame',self.data[0])
        #if cv2.waitKey(0) & 0xFF == ord('q'):
            #cv2.destroyAllWindows()

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
        key_words = [#'bald',
                    'black hair',
                    'blond hair',
                    'brown hair',
                    #'gray hair',
                    'glasses',
                    #'goatee',
                    'male',
                    #'mustache',
                    'beard',
                    #'white'
                    ]
        vec = np.ones(self.y_dim)*-1
        for i, key in enumerate(key_words, 0):
            if key in text:
                vec[i] = 1.
        #print(vec)
        batch_vector = np.tile(vec,(self.batch_size,1))
        return batch_vector

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
