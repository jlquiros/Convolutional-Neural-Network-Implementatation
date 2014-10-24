import numpy as np;
from sklearn import preprocessing
import cnn as cn

## Used to cross verify whether back propagation works good
def computeNumericalGradient(evaluator, images, labels, cLayer, pLayer, poolDim):


    numgrad = np.zeros(cLayer.weights.shape)
    epsilon = 1e-4

    result = np.zeros(2)

    #pLayer.weights[0][0]  += epsilon
    cLayer.weights[0][0][0][0]  += epsilon
    #cLayer.bias[0] += epsilon
    #pLayer.bias[0]  += epsilon
    (pos, gradp) = evaluator(images, labels, cLayer, pLayer, poolDim)
    #pLayer.bias[0]  -= 2 * epsilon
    cLayer.weights[0][0][0][0]  -= 2 * epsilon
    #cLayer.bias[0] -= 2 * epsilon
    (neg, gradn) = evaluator(images, labels, cLayer, pLayer, poolDim)
    result[0] = (pos-neg)/(2* epsilon)
    cLayer.weights += epsilon

    cLayer.weights[0][0][1][1] += epsilon
    #pLayer.bias[0]  += epsilon
    (pos, gradp) = evaluator(images, labels, cLayer, pLayer, poolDim)
    #pLayer.bias[0]  -= 2 * epsilon
    #cLayer.weights[0][0][0][0]  -= 2 * epsilon
    cLayer.weights[0][0][1][1] -= 2 * epsilon
    (neg, gradn) = evaluator(images, labels, cLayer, pLayer, poolDim)
    result[1] = (pos-neg)/(2* epsilon)
    cLayer.weights += epsilon

    return result 
    
# Returns cost and gradiant for given training set, and 
# convolution layer cLayer and fully connected network pLayer with
# pooling of poolDim 
def evaluate (input, target, cLayer, pLayer, poolDim):

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
  computedOutput = pLayer.propagate(classifierInput)

  # Create sparse indication vector for class labels
  lb = preprocessing.LabelBinarizer()
  lb.fit(range(0,10))
  trueOutput = lb.transform(target[0:len(target)])

  # Computing the cost from cross entrophy function
  cost = np.multiply(trueOutput,np.log(computedOutput))
  cost = -1 * np.sum(cost)
  cost = cost/input.shape[0]

  
  #######################################################################

  # Gradiant of Error with respect to output activation ak
  deltaO = (computedOutput - trueOutput)

  # computing pre output layer error gradiant. It should be same as the no of input units to NN classifier given an image 
  deltaPO = np.dot(deltaO,pLayer.weights.T)

  # computing gradiant with respect the only weight layer present; 
  # Dimensions are same as parameters of the layer
  Wd_grad = np.dot(classifierInput.T,deltaO) 
  Wd_grad = Wd_grad /nInput
  assert(Wd_grad.shape == pLayer.weights.shape)

  # Bias is simply the error at that layer
  bd_grad = np.sum(deltaO,axis = 0)/len(deltaO)

  # Reconstructing error in 2d to aid in propagation through conv and pool layer
  reConPooledDelta = np.reshape(deltaPO,(sampledOutput.shape))

  # Recreating the required after convolution error mask for feature mask 
  # Propagating error thru pooling
  reConDelta = np.kron(reConPooledDelta,np.ones((poolDim,poolDim)))/(poolDim * poolDim)

  # Error gradient with respect to outer layer of convolution is product of error above it and differentiated activation
  convolvedOutput_ = 1 - convolvedOutput
  diConvolutionOutput = np.zeros(convolvedOutput.shape) 
  deltaCon = np.zeros(convolvedOutput.shape) 
  bc_grad = np.zeros(convolvedOutput.shape[1])
  for i in range(0,convolvedOutput.shape[0]):
    for j in range(0,convolvedOutput.shape[1]):
      diConvolutionOutput[i][j] = convolvedOutput[i][j] * convolvedOutput_[i][j]
      deltaCon[i][j] = diConvolutionOutput[i][j] * reConDelta[i][j]
      bc_grad[j] += np.sum(np.sum(deltaCon[i][j]))
  
  # Bias for convolution layer is calculated by just adding errors of its outer layer for a given mask
  bc_grad = bc_grad/(nInput)

  # propagating error to the input layer by convoluting with input image
  reConvolutedValue =  cLayer.convolved(input, deltaCon)
  Wc_grad  =  np.sum(reConvolutedValue, axis = 0)/nInput
  Wc_grad = np.reshape(Wc_grad, cLayer.weights.shape)

  grad = [Wd_grad.ravel(),Wc_grad.ravel(),bd_grad.ravel(),bc_grad.ravel()]
  return (cost,grad)

