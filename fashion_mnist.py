import numpy as np
import matplotlib.pyplot as plt

import utils.mnist_reader as mnist_reader
from utils import mnist_reader
XM_train, YM_train = mnist_reader.load_mnist('data-FM/fashion', kind='train')
XM_test, YM_test = mnist_reader.load_mnist('data-FM/fashion', kind='t10k')

# divide training data to two parts, one for training, and the other is for validation, then, normalized the datas.
inds = np.arange(len(XM_train))
np.random.shuffle(inds)
train_inds = inds[:50000]
valid_inds = inds[50000:]

x_train = np.matrix(XM_train[train_inds,:]).T
# xt_mean = np.mean(x_train)
# xt_std = np.std(x_train)
# x_train = (x_train-xt_mean)/xt_std
y_train = np.matrix(YM_train[train_inds])
print('x_train:',np.shape(x_train))
print('y_train:',np.shape(y_train))

x_valid = np.matrix(XM_train[valid_inds,:]).T
# xv_mean = np.mean(x_valid)
# xv_std = np.std(x_valid)
# x_valid = (x_valid-xv_mean)/xv_std
y_valid = np.matrix(YM_train[valid_inds])
print('x_valid:',np.shape(x_valid))
print('y_valid:',np.shape(y_valid))

x_test = np.matrix(XM_test).T
# xte_mean = np.mean(x_test)
# xte_std = np.std(x_test)
# x_test = (x_test-xte_mean)/xte_std
y_test = np.matrix(YM_test)
print('x_test:',np.shape(x_test))
print('y_test:',np.shape(y_test))


'''
# Update by Jeanne
# 2018-12-02
# input: label_listid: int/[y_test[i, 0], y_test.shape: 10000, 1]
# output: the Name of the label
# Test: getLabelName(y_test[12,0])
'''
def getLabelName(label_listid):
    switcher = {
        0 : 'T-shirt/top',
        1 : 'Trouser',
        2 : 'Pullover',
        3 : 'Dress',
        4 : 'Coat',
        5 : 'Sandal',
        6 : 'Shirt',
        7 : 'Sneaker',
        8 : 'Bag',
        9 : 'Ankle boot',
        10 : 'Unkown'
    }
    return switcher[label_listid]

'''
# Update by Jeanne
# 2018-12-02
# showImageRandom
# input: data set, label set
# output: Randomly show the image with reshape 28*28, show the label name 
# Test: showImageRandom(x_train, y_train)
'''
def showImageRandom(data, label):
    x_image = np.zeros((28,28))
    i = np.random.randint(data.shape[0])
    x_image = (data[i,:]).reshape(28,28)
    plt.imshow(x_image, cmap='gray')
    print(getLabelName(label[i,0]))
    
'''
# Update by Jeanne
# 2018-12-02
# showImageWithId
# input: image index, data set, label set
# output: Show specific image with reshape 28*28, show the label name 
# Test: showImageWithId(233, x_train, y_train)
'''
def showImageWithId(idx, data, label):
    x_image = np.zeros((28,28))
    x_image = (data[idx,:]).reshape(28,28)
    plt.imshow(x_image, cmap='gray')
    print(getLabelName(label[idx,0]))