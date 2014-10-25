import numpy as np;
from math import sqrt
import reader as rd;
import cnn_evaluate_parameters as ce
import neural_networks as nn
import cnn as cn

import cnn 

# Contains useful function to initialize  a network and core class for neural network
# Currently used to Initialize two layer convolution network with one c/p layer and one mutliclass logistic regressor
class NeuralNetworks(object):

    def init(self,numFilters,filterDim,numClasses,imgDim,poolDim):

        #Initial Settings

        outputDim = imgDim - filterDim + 1
        outputDim = outputDim/poolDim

        hiddenLayerSize = outputDim * outputDim  * numFilters;

        # Layer 1
        cnn = cn.ConvolutionNN()
        cnn.initialize_params(numFilters,1,filterDim)

        # Layer 2
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
        weighted_value = (weighted_value.T/ weighted_value.sum(axis = 1)).T
        return weighted_value

    def propagate_network (self, input, cLayer, pLayer, poolDim):

      # Extracting information
      nInput = input.shape[0]
      # Reshaping input as (NumImages,FeatureMap,Height,Width)
      input = np.reshape(input,(input.shape[0],1,input.shape[1],input.shape[2]))
      ## Forward Propagate to compute cost
      convolvedOutput = cLayer.convolve(input)
      sampledOutput = cLayer.pool(convolvedOutput,poolDim)

      #Crude way to straighten out the sampled output for fully connected nn classifier
      dim = 1
      for m in sampledOutput[0].shape:
        dim = dim * m

      classifierInput = np.reshape(sampledOutput,(input.shape[0],dim))
      computedOutput = pLayer.feed_forward(classifierInput)
      return computedOutput

    def calculate_misclassification(self, result, target):

        # Create sparse indication vector for class labels
      result = np.argmax(result,axis = 1)
      return np.sum(result == target)

      # Computing the cost from cross entrophy function



