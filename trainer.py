import numpy as np

# evaluator - function that computes the cost and gradient given the layers
# layers - [cLayer, pLayer] 
# Options
# *epoch - number of epoch
# *minibatch - size of minibatch
# alpha - initial learning rate
# momentum - defaults to 0.9
#
# Returns
# [cLayer, pLayer] optimized layers
# 

def SGD(evaluator, layers, data, labels, options):

    if 'minibatch' in options and 'epoch' in options and 'alpha' in options:
        pass
    else:
        print("Missing Required options minibatch, epoch and alpha")
        exit()

    nepoch = options['epoch']
    alpha = options['alpha']
    mbatchsz = options['minibatch']
    nImages = labels.shape[0]
    momentum = 0.5
    momIncrease = 20
    momUpdate = 0.9
    poolDim = 2

    (cLayer, pLayer) = layers


    if momentum  in options:
        momUpadate = options['momentum']


    velocity_gC = np.zeros(cLayer.weights.shape)
    velocity_gP = np.zeros(pLayer.weights.shape)
    velocity_bC = np.zeros(cLayer.bias.shape)
    velocity_bP = np.zeros(pLayer.bias.shape)

    alpha = 0.01

    for iloop in range(0,nepoch):

        rp = np.random.permutation(nImages)

        iImages = 0
        it = 0
        iBatch = 0
        while (iBatch < nImages):

            if it == momIncrease:
                momentum = momUpdate
            
            iBatchN = min(iBatch + mbatchsz - 1,nImages)

            mbData = data[rp[range(iBatch, iBatchN)],:,:]
            mbLabel = labels[rp[range(iBatch, iBatchN)]]


            (cost, grad) = evaluator(mbData,mbLabel,cLayer,pLayer,poolDim)

            velocity_gC = momentum * velocity_gC + alpha * (grad['gC'])
            velocity_gP = momentum * velocity_gP + alpha * (grad['gP'])
            velocity_bC = momentum * velocity_bC + alpha * (grad['gBc'])
            velocity_bP = momentum * velocity_bP + alpha * (grad['gBp'])


            cLayer.weights = cLayer.weights - velocity_gC
            pLayer.weights = pLayer.weights - velocity_gP
            cLayer.bias = cLayer.bias - velocity_bC
            pLayer.bias = pLayer.bias - velocity_bP

            print('epoch, iteration, cost ',iloop," ",it," ",cost)

            iBatch = iBatchN

            it = it + 1
        
        alpha = alpha / 2.0

    return (cLayer, pLayer)















