#!/anaconda/envs/torvnet python3

import sys
import os
import argparse

import numpy as np

import train

basePath = os.getcwd()

params = dict()
params['DataManagerParams'] = dict()
params['ModelParams'] = dict()

#  params of the algorithm
params['ModelParams']['sigma'] = 15 # used to produce randomly deformed images in data augmentation

# params['ModelParams']['device'] = 0
# params['ModelParams']['snapshot'] = 0

# params['ModelParams']['task'] = 'promise12'
params['ModelParams']['task'] = 'nci-isbi-2013'

params['ModelParams']['dirTrainImage'] = os.path.join(basePath,'dataset/imagesTr')  # if 'dirTest' is empty, denotes 'path to a dataset that will later be split into trainSet and testSet. Otherwise, denotes just trainSet.
params['ModelParams']['dirTrainLabel'] = os.path.join(basePath,'dataset/labelsTr')

# updated for 'nci-isbi-2013'
params['ModelParams']['dirTestImage'] = os.path.join(basePath,'dataset/imagesTs') # path to test images
params['ModelParams']['dirTestLabel'] = os.path.join(basePath,'dataset/labelsTs') # path to test labels

# no need to do this now as there is separate test set
# params['ModelParams']['testProp'] = 0.2  # if 'dirTestImage' or 'dirTestLabel' is empty, split 'dirTrainImage' and 'dirTrainLabel' into train and test

# no images with no labels provided - not needed for this run
params['ModelParams']['dirInferImage'] = '' # used for inference, usually no labels provided.

params['ModelParams']['dirResult'] = os.path.join(basePath, 'results'+'_'+params['ModelParams']['task'])  # where we need to save the results (relative to the base path)

# seems like snapshots aren't saved while training, but still a model is saved - should be looked into -> when?
# ????
# params['ModelParams']['dirSnapshots'] = os.path.join(basePath,'Models/MRI_cinque_snapshots/')  # where to save the models while training

params['ModelParams']['nProc'] = 4  # the number of threads to do data augmentation


# params of the DataManager

# these two are passed separately as arguments below
# params['DataManagerParams']['dstRes'] = np.asarray([1,1,1.5],dtype=float)
# params['DataManagerParams']['VolSize'] = np.asarray([128, 128, 64],dtype=int)

params['DataManagerParams']['normDir'] = False  # it rotates the volume according to its transformation in the mhd file. Not reccommended.

# print('\n+preset parameters:\n' + str(params))


#  parse sys.argv
parser = argparse.ArgumentParser()

# https://www.youtube.com/watch?v=QKCxAJsLWxQ
parser.add_argument('--numcontrolpoints', type=int, default=2) # for B-spline free-form deformation？？？ what are the method details? By Chao.

parser.add_argument('--testProp', type=float, default=0.2) # if 'dirTestImage' or 'dirTestLabel' is empty, split 'dirTrainImage' and 'dirTrainLabel' into train and test

# this is output spacing
parser.add_argument('--dstRes', type=str, default='[1, 1, 1.5]')

# this is output size
parser.add_argument('--VolSize', type=str, default='[128, 128, 64]')

parser.add_argument('--batchsize', type=int, default=2)

# around 30,000 - 50,000 for actual dataset. After that it starts overfitting for promise12
# 10 is set for testing. Real should be like 50,000
parser.add_argument('--numIterations', type=int, default=30000) # the number of iterations, used by https://github.com/faustomilletari/VNet, as only one Epoch run.

parser.add_argument('--baseLR', type=float, default=0.0001) # the learning rate, initial one
parser.add_argument('--momentum', type=float, default=0.99)
parser.add_argument('--weight_decay', '--wd', default=1e-8, type=float,
                    metavar='W', help='weight decay (default: 1e-8)')
parser.add_argument('--stepsize', type=int, default=20000)
parser.add_argument('--gamma', type=float, default=0.1)

parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--opt', type=str, default='adam',
                    choices=('sgd', 'adam', 'rmsprop'))

parser.add_argument('--dice', action='store_true', default=True)
parser.add_argument('--gpu_ids', type=int, default=1) # what if multiple gpu ids? use list? by Chao.
parser.add_argument('--nEpochs', type=int, default=1) # line "dataQueue_tmp = dataQueue" in train.py is not working for epoch=2 and so on. Why? By Chao.
parser.add_argument('--xLabel', type=str, default='Iteration', help='x-axis label for training performance transition curve, accepts "Epoch" or "Iteration"')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

# if resuming training, pass path to stored checkpoint
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--no-cuda', action='store_true', default=False)

args = parser.parse_args()

# print('\n+sys arguments:\n' + str(args))

#  load dataset, train, test(i.e. output predicted mask for test data in .mhd)
# NOTE: Uncomment later
train.main(params, args)


# NOTE: TESTING AREA

# NOTE: Trained models to be used for evaluation
# results/vnet.base.promise12.20190710_1045/vnet_epoch1_checkpoint.pth.tar
# results/vnet.base.nci-isbi-2013.20191005_0155/vnet_epoch1_checkpoint.pth.tar