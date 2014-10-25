import numpy as np;
import pylab
from PIL import Image
import pylab
from matplotlib import pyplot as plt


def read_mnist():
    trainData = read_data(type = 'mnistdb', file = '/home/zephyr/workspace/machine_learning/res/data/mnist/train-images.idx3-ubyte')
    trainLabel = read_label(type = 'mnistdb', file = '/home/zephyr/workspace/machine_learning/res/data/mnist/train-labels.idx1-ubyte')
    testData = read_data(type = 'mnistdb', file = '/home/zephyr/workspace/machine_learning/res/data/mnist/t10k-images.idx3-ubyte')
    testLabel = read_label(type = 'mnistdb', file = '/home/zephyr/workspace/machine_learning/res/data/mnist/t10k-labels.idx1-ubyte')
    trainLabel = trainLabel.flatten().T
    testLabel = testLabel.flatten().T
    #index = 0
    #while (index < 0):
    #    number = np.random.random() * 60000
    #    number = number / 1
    #    plot = plt.imshow(data[number])
    #    print (label[number])
    #    plt.show()
    #    index += 1
    return (trainData, trainLabel, testData, testLabel)

def testWithImage():
    img = Image.open(open('../res/data/randomimages/3wolfmoon.jpg'))
    img = np.asarray(img, dtype='float64') / 256
    img_ = img.swapaxes(0, 2).swapaxes(1, 2).reshape(1, 3, 639, 516)
    filtered_img = cp.Convolute(img_)
    # plot original image and first and second components of output
    pylab.subplot(1, 3, 1); pylab.axis('off'); pylab.imshow(img)
    pylab.gray();
    # recall that the convOp output (filtered image) is actually a "minibatch",
    # of size 1 here, so we take index 0 in the first dimension:
    pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(filtered_img[0, 0, :, :])
    print filtered_img.shape
    filtered_img = cp.max_pool2d(filtered_img,(2,2))
    print filtered_img.shape
    pylab.subplot(1, 3, 3); pylab.axis('off'); pylab.imshow(filtered_img[0, 0, :, :])
    pylab.show()

def read_data(type, file):
    if (type == 'mnistdb'):
        return read_mnistdb(file)

def read_label(type, file):
    if (type == 'mnistdb'):
        return read_label_mnistdb(file)

def read_mnistdb(file):
    fd = open(file, 'rb')
    datatype = 'i'
    size = 1
    i4_be = np.dtype('>i4')
    [magic, nimages, nrows, ncolumns] =  np.fromfile(fd,i4_be,4)
    size =   nimages * nrows * ncolumns
    ui_be = np.dtype('u1')
    read_data =  np.fromfile(fd,ui_be,size)
    shape = (nimages, nrows, ncolumns)
    data = np.reshape(read_data,shape)
    return data

def read_label_mnistdb(file):
    fd = open(file, 'rb')
    datatype = 'i'
    size = 1
    i4_be = np.dtype('>i4')
    [magic, nimages] =  np.fromfile(fd,i4_be,2)
    size =   nimages 
    ui_be = np.dtype('u1')
    read_data =  np.fromfile(fd,ui_be,size)
    shape = (nimages, 1)
    data = np.reshape(read_data,shape)
    return data

