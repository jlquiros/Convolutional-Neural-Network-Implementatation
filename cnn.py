import numpy as np
from skimage import measure
import scipy
from scipy import signal

###########################################################
class ConvolutionNN(object):
###########################################################

    # input and W are 4Dimensional elelemts
    # input is of form (numImages, numInputFeatureMap, imageHeight, imageWidth)
    # W is of shape (numOutputFeatureMap, numInputFeatureMap, filterHeight, fiterWidth)
    # b is of shape (numOutputFeatureMap, 1)
    def convolve(self, input):

        # Extracting useful info
        W = self.weights
        b = self.bias
        numImages = input.shape[0]
        imageDim = input.shape[2:]
        filterDim = W.shape[2:]
        numFeatureMap = W.shape[0]
        numInputFeatureMap = W.shape[1]
        convDim = np.subtract(imageDim,filterDim) + 1

        convolvedFeatures = np.zeros((numImages, numFeatureMap, convDim[0], convDim[1]))


        for imageNum in range(0, numImages):
            for outFeatureNum in range(0, numFeatureMap):
                convolvedImage = np.zeros((convDim[0],convDim[1]))
                for inpFeatureNum in range(0, numInputFeatureMap):
                    convolvedImage = convolvedImage + scipy.signal.convolve2d(input[imageNum][inpFeatureNum], np.rot90(W[outFeatureNum][inpFeatureNum],2), 'valid') 
                convolvedImage = self.sigmoid(convolvedImage + b[outFeatureNum])
                convolvedFeatures[imageNum][outFeatureNum] = convolvedImage
        return convolvedFeatures

    # input and delta are 4Dimensional elelemts
    # input is of form (numImages, numInputFeatureMap, imageHeight, imageWidth)
    # delta is of shape ((numImages , numOuputFeatureMap, filterHeight, fiterWidth)
    def convolve_error(self, input, delta):

        # Extracting useful info
        numImages = input.shape[0]
        imageDim = input.shape[2]
        numFeatureMap = delta.shape[1]
        filterDim = delta.shape[2]
        convDim = imageDim - filterDim + 1

        convolvedFeatures = np.zeros((numImages, numFeatureMap, convDim, convDim))


        # convolving error-filter mask  with input image that made the error calculation for given filter
        # this gives the gradient of filter parameters for all the input images
        for imageNum in range(0, numImages):
            for outFeatureNum in range(0, numFeatureMap):
                convolvedImage = np.zeros((convDim, convDim))
                convolvedImage = scipy.signal.convolve2d(input[imageNum][0], np.rot90(delta[imageNum][outFeatureNum],2), 'valid') 
                convolvedFeatures[imageNum][outFeatureNum] = convolvedImage
        
        # resultant convolved feature is three dimensional filter masks gradient for every input image
        return convolvedFeatures

    # pooling with np.mean reduces the dimension of input by dim
    def pool(self, input, dim):

        shape = (1, 1, dim, dim)
        return  measure.block_reduce(input, block_size=shape, func=np.mean)

    def sigmoid(self, input):

        return (1.0/(1 + np.exp(-input)))

    def initialize_params(self,numFeatureMap, numInputFeatureMap, filterDim):

        self.weights = np.random.random((numFeatureMap, numInputFeatureMap, filterDim, filterDim))
        self.bias = np.random.random((numFeatureMap))


