import numpy as np;
from math import sqrt
import reader as rd;
import cnn_evaluate_parameters as ce
import neural_networks as nn
import cnn as cn

import cnn 


class NeuralNetworks(object):

    def init(self,numFilters,filterDim,numClasses,imgDim,poolDim):

        #Initial Settings

        cnn = cn.ConvolutionNN()
        cnn.initialize_params(numFilters,1,filterDim)

        outputDim = imgDim - filterDim + 1
        outputDim = outputDim/poolDim

        hiddenLayerSize = outputDim * outputDim  * numFilters;

        regressionLayer  = NeuralNetworks()
        regressionLayer.initialize_params(hiddenLayerSize, numClasses)
        return (cnn,regressionLayer)
    
    
    #Initializes a single layer network with hiddenLayerSize * numClasses parameters
    def initialize_params(self,hiddenLayerSize, numClasses):

        r  = sqrt(6) / sqrt(numClasses+hiddenLayerSize+1);
        self.weights = np.random.random((hiddenLayerSize,numClasses)) * 2 * r - r
        self.bias = np.random.random(numClasses)

    # Forward Propagate with input and already trained weights
    # uses softmax for multiclass trained network
    def feed_forward(self, input):

        weighted_value = np.exp( input.dot(self.weights) + self.bias)
        print weighted_value
        weighted_value = (weighted_value.T/ weighted_value.sum(axis = 1)).T
        return weighted_value




