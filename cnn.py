import numpy as np
from skimage import measure
import scipy
from scipy import signal

###########################################################
class ConvolutionNN(object):
    ####################################################
    def convolve(self, input, W, b):

        # input and W are 4Dimensional elelemts
        # input is of form (numImages, numInputFeatureMap, imageHeight, imageWidth)
        # W is of shape (numOutputFeatureMap, numInputFeatureMap, filterHeight, fiterWidth)
        # b is of shape (numOutputFeatureMap, 1)

        # Extracting useful metadata
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
                    convolvedImage = convolvedImage + scipy.signal.convolve2d(input[imageNum][inpFeatureNum], W[outFeatureNum][inpFeatureNum], 'valid') 
                convolvedImage = self.sigmoid(convolvedImage + b[outFeatureNum])
                convolvedFeatures[imageNum][outFeatureNum] = convolvedImage
        return convolvedFeatures


    def maxpool(self, input, dim):
        shape = (1, 1, dim, dim)
        return  measure.block_reduce(input, block_size=shape, func=np.mean)
        #dim dimensions to be reduced

    def sigmoid(self, input):
        return (1.0/(1 + np.exp(-input)))

