import reader as rd
import numpy as np
import cnn as cn
from math import exp
from matplotlib import pyplot as plt
import neural_networks as nn
import cnn_evaluate_parameters as eval


def backPropagationTest():
    numFilters = 2;
    filterDim = 9;
    poolDim = 5;
    (images, labels, testImages, testlables) = rd.read_mnist() 
    images = (images)/255.0
    images = images[1:10,:,:]
    labels = labels[1:10]

    print images.shape
    print labels.shape
    neural_net = nn.NeuralNetworks()
    imageDim = 28
    nclasses = 10

    (cLayer,pLayer) = neural_net.init(numFilters,filterDim,nclasses,imageDim,poolDim)
    weights = [pLayer.weights.ravel(),cLayer.weights.ravel(), pLayer.bias.ravel(), cLayer.bias.ravel()]
    (cost, grad) = eval.evaluate (images, labels, cLayer, pLayer, poolDim)
    #x = eval.evaluate(image,label,cLayer,pLayer,poolDim)

    numgrad = eval.computeNumericalGradient(eval.evaluate,images,labels,cLayer,pLayer,poolDim)
    print "Current Gradiant"
    print grad.shape
    print numgrad
    print((numgrad[0] - grad[1][0]))

def convolutionTest():
    (images, labels, testImages, testlables) = rd.read_mnist() 

    numFeatureMap = 100;
    imageDim = 28;
    filterDim = 8;
    inputFeatureNum = 0;
    numInputFeatureMap = 1
    
    cnn = cn.ConvolutionNN()
    #-------------------------------------------------------------
    # Testing convolution


    images = np.reshape(images, (images.shape[0],1,imageDim,imageDim))
    W = np.random.random((numFeatureMap,numInputFeatureMap,filterDim,filterDim))
    b = np.random.random((numFeatureMap))

    convTestImages = images[0:8, :, :, :]

    convolvedFeatures = cn.ConvolutionNN().convolve(convTestImages, W, b)


    for i in range(0,1000): 
        filterNum = np.random.randint(1, numFeatureMap);
        imageNum = np.random.randint(1, 8);
        imageRow = np.random.randint(1, imageDim - filterDim)
        imageCol = np.random.randint(1, imageDim - filterDim)

        patch = convTestImages[imageNum, inputFeatureNum, imageRow:imageRow + filterDim, imageCol:imageCol + filterDim];
        w = W[filterNum,inputFeatureNum,:,:]
        feature = np.sum(np.sum(patch.dot(W[filterNum,inputFeatureNum,:,:]))+b[filterNum])
        feature = 1.0/(1+np.exp(-feature))

        if abs(feature - convolvedFeatures[ imageNum,filterNum,imageRow, imageCol]) > 1e-9:
            print('Convolved feature does not match test feature\n');
            print('Filter Number    : %d\n', filterNum);
            print('Image Number      : %d\n', imageNum);
            print('Image Row         : %d\n', imageRow);
            print('Image Column      : %d\n', imageCol);
            print('Convolved feature : %0.5f\n', convolvedFeatures[imageNum, filterNum, imageRow, imageCol])
            print('Test feature : %0.5f\n', feature)
            print('Convolved feature does not match test feature');
       

    print('Congratulations! Your convolution code passed the test.');


    #--------------------------------------------------------------------
    testMatrix = np.arange(1,65).reshape(8, 8)
    expectedMatrix = np.array([np.mean(np.mean(testMatrix[0:4, 0:4])), np.mean(np.mean(testMatrix[0:4, 4:8])),
                  np.mean(np.mean(testMatrix[4:8, 0:4])), np.mean(np.mean(testMatrix[4:8, 4:8])) ])
    testMatrix = np.reshape(testMatrix, (1, 1, 8, 8))
        
    pooledFeatures = cnn.maxpool(testMatrix,4).flatten()

    if  not np.array_equal(pooledFeatures, expectedMatrix):
        print('Pooling incorrect');
        print('Expected');
        print(expectedMatrix);
        print('Got');
        print(pooledFeatures);
    else:
        print('Congratulations! Your pooling code passed the test.');
    


if (__name__ == '__main__'):
    backPropagationTest()

