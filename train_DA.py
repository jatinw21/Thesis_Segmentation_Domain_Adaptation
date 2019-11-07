#!/usr/bin/env python3
import os
from sys import exit
from os.path import splitext
from random import sample
import time
from multiprocessing import Process, Queue
import pdb

import shutil
import setproctitle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import lossFuncs
import utils as utils

import vnet_DA as vnet
import DataManager as DM
import DicomManager as DCM

import customDataset

### NEW ###
import random
from sys import exit
# import torch.backends.cudnn as cudnn

source_dataset_name = 'promise12'
target_dataset_name = 'nci-isbi-2013'
source_image_root = os.path.join('dataset')
target_image_root = os.path.join('dataset_nci')
model_root = 'models'
cuda = True
basePath = os.getcwd()
# cudnn.benchmark = True

# NOTE: review these below - They're all set in main for us, not here
# lr = 1e-3
# batch_size = 128
# image_size = 28
# n_epoch = 100


### NEW ###

iteration_dice_average = 0
iteration_err_average = 0

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()


def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


def save_checkpoint(state, path, prefix, filename='checkpoint.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)


def train_test_split(images, labels, test_proportion):
    # images and labels are both dict().
    # pdb.set_trace()
    # NOTE: '_segmentation' appended to file name is only true for promise12
    # but because only promise12 needs train_test split, not nci - no need to change
    keys = list(images.keys())
    size = len(keys)
    test_keys = sample(keys, int(test_proportion*size))
    test_images = {i: images[i] for i in keys if i in test_keys}
    test_labels = {i+'_segmentation': labels[i+'_segmentation']
                   for i in keys if i in test_keys}  # require customization
    train_images = {i: images[i] for i in keys if i not in test_keys}
    train_labels = {i+'_segmentation': labels[i+'_segmentation']
                    for i in keys if i not in test_keys}  # require customization
    return train_images, train_labels, test_images, test_labels


def dataAugmentation(params, args, dataQueue, numpyImages, numpyGT):

    nr_iter = args.numIterations  # params['ModelParams']['numIterations']
    batchsize = args.batchsize  # params['ModelParams']['batchsize']
    task = params['ModelParams']['task']

    # pdb.set_trace()
    keysIMG = list(numpyImages.keys())

    nr_iter_dataAug = nr_iter*batchsize
    np.random.seed(1)
    whichDataList = np.random.randint(len(keysIMG), size=int(
        nr_iter_dataAug/params['ModelParams']['nProc']))
    np.random.seed(11)
    whichDataForMatchingList = np.random.randint(
        len(keysIMG), size=int(nr_iter_dataAug/params['ModelParams']['nProc']))

    for whichData, whichDataForMatching in zip(whichDataList, whichDataForMatchingList):

        currImgKey = keysIMG[whichData]
        # require customization. This is for PROMISE12 data.
        
        if task == 'nci-isbi-2013':
            # NOTE: the training set labels for promise12 have same name as the actual file
            # + _truth is for test set
            currGtKey = keysIMG[whichData]
        else:
            currGtKey = keysIMG[whichData] + '_segmentation'

        # print("keysIMG type:{}\nkeysIMG:{}".format(type(keysIMG),str(keysIMG)))
        # print("whichData:{}".format(whichData))
        # pdb.set_trace()
        # currImgKey = keysIMG[whichData]
        # currGtKey = keysIMG[whichData] # for MSD data.

        # data agugumentation through hist matching across different examples...
        ImgKeyMatching = keysIMG[whichDataForMatching]

        defImg = numpyImages[currImgKey]
        defLab = numpyGT[currGtKey]

        # why do histogram matching for all images? By Chao.
        defImg = utils.hist_match(defImg, numpyImages[ImgKeyMatching])

        # do not apply deformations always, just sometimes
        if(np.random.rand(1)[0] > 0.5):
            defImg, defLab = utils.produceRandomlyDeformedImage(
                defImg, defLab, args.numcontrolpoints, params['ModelParams']['sigma'])

        dataQueue.put(tuple((defImg, defLab)))


def adjust_opt(optAlg, optimizer, iteration):
    if optAlg == 'sgd':
        if epoch < 150:
            lr = 1e-1
        elif epoch == 150:
            lr = 1e-2
        elif epoch == 225:
            lr = 1e-3
        else:
            return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def train_dice(args, epoch, iteration, model, trainLoader, optimizer, trainF):
    global iteration_dice_average

    model.train()
    nProcessed = 0
    batch_size = len(trainLoader.dataset)

    
    for batch_idx, output in enumerate(trainLoader):
        # NOTE: placeholder for alpha
        alpha = 0.001

        # data shape [batch_size, channels, z, y, x], output shape [batch_size, z, y, x]
        data, target = output
        # pdb.set_trace()
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data = Variable(data)
        target = Variable(target)
        # data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output, domain_output = model(data, alpha)  # output shape[batch_size, 2, z*y*x]
        # print("data shape:{}\noutput shape:{}\ntarget shape:{}".format(data.shape, output.shape, target.shape))
        loss = lossFuncs.dice_loss(output, target)
        # make_graph.make_dot(os.path.join(resultDir, 'promise_net_graph.dot'), loss)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        # loss.data[0] is sum of dice coefficient over a mini-batch. By Chao.
        diceOvBatch = loss.data[0]/batch_size
        iteration_dice_average += diceOvBatch
        err = 100.*(1. - diceOvBatch)

    num_it_average = 10
    if np.mod(iteration, num_it_average) == 0:
        iteration_dice_average /= num_it_average
        print(f'\nFor training: epoch: {epoch} iteration: {iteration} \tdice_coefficient over last {num_it_average} : {iteration_dice_average:.4f}\n')
        iteration_dice_average = 0
        # print('\nFor trainning: epoch: {} iteration: {} \tdice_coefficient over batch: {:.4f}\tError: {:.4f}\n'.format(
        #     epoch, iteration, diceOvBatch, err))

    return diceOvBatch, err

def train_dice_DA(args, epoch, iteration, model, trainLoader, optimizer, trainF, trainLoader_target, nr_iter):
    global iteration_dice_average
    global iteration_err_average

    loss_domain = torch.nn.NLLLoss()

    model.train()
    nProcessed = 0

    len_dataloader = min(len(trainLoader), len(trainLoader_target))
    data_source_iter = iter(trainLoader)
    data_target_iter = iter(trainLoader_target)

    batch_size = len(trainLoader.dataset)
    
    # print('Lengths of trainloader source and target', len(trainLoader), len(trainLoader_target)) # 1 1
    # print('Length of dataloader (min)', len_dataloader) # 1
    # print('batch size', batch_size) # 2

    i = 0

    while i < len_dataloader:
        # NOTE: changing epoch to iteration in the formula below,
        # because for this epoch is always 1, but iteration is what changes
        # REVIEW: this formula with iteration now
        p = float(i + iteration * len_dataloader) / (nr_iter) / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        # print(alpha)

        # print(p) # 0.5
        # print(alpha) # 0.9866142981514305 for nr_ier = 2 while testing
        
        # BUG: remove later
        # nr_iter = 30000
        # p = float(i + iteration * len_dataloader) / nr_iter / len_dataloader
        # alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # print(p) # 3.3333333333333335e-05
        # print(alpha) # 0.0001666666651234383
        # NOTE: seems like alpha starts off really small and then gets bigger
        # Not sure how that really works

        
    

    
    # for batch_idx, output in enumerate(trainLoader):
    #     # NOTE: placeholder for alpha
    #     alpha = 0.001

        # data shape [batch_size, channels, z, y, x], output shape [batch_size, z, y, x]
        # data, target = output
        data, target = data_source_iter.next()
        domain_label = torch.zeros(batch_size)
        domain_label = domain_label.long()

        # NOTE: why is there an extra 1 dimension below for data??
        # print(data.shape) # torch.Size([2, 1, 64, 128, 128])
        # print(target.shape) # torch.Size([2, 64, 128, 128])
        # print(domain_label.shape) # torch.Size([2])
        # print(domain_label) # tensor([0, 0])

        
        # batch_size = len(target)
        # pdb.set_trace()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
            domain_label = domain_label.cuda()
            loss_domain = loss_domain.cuda()

        data = Variable(data)
        target = Variable(target)
        # data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output, domain_output = model(data, alpha)  # output shape[batch_size, 2, z*y*x]
        # print('domain_output', domain_output)

        # print(output.shape) # torch.Size([2, 2, 1048576])
        # print(domain_output.shape) # torch.Size([4, 2])
        # print(domain_output) # tensor([[-0.9123, -0.5135],
        # [-0.5528, -0.8565],
        # [-0.9123, -0.5135],
        # [-0.5528, -0.8565]], device='cuda:0', grad_fn=<GatherBackward>)


        # print("data shape:{}\noutput shape:{}\ntarget shape:{}".format(data.shape, output.shape, target.shape))
        loss = lossFuncs.dice_loss(output, target)
        # make_graph.make_dot(os.path.join(resultDir, 'promise_net_graph.dot'), loss)
        loss.backward(retain_graph=True)
        
        # print(type(loss)) # <class 'torch.Tensor'>
        # print(loss.shape) # torch.Size([1])
        # print(loss) # tensor([0.1000], device='cuda:0', grad_fn=<DiceLoss>)
        # NOTE: need to split it in half because there is repitition
        domain_output = domain_output.split((2))[0]
        err_s_domain = loss_domain(domain_output, domain_label)

        # print(err_s_domain)

        # NOTE: training model using target data
        data_target = data_target_iter.next()
        t_img, _ = data_target
        
        # batch_size = len(t_img)
        domain_label = torch.ones(batch_size)
        domain_label = domain_label.long()
        # batch_size = len(target)
        # pdb.set_trace()
        if args.cuda:
            t_img = t_img.cuda()
            domain_label = domain_label.cuda()

        t_img = Variable(t_img)
        _, domain_output = model(t_img, alpha)
        # print('domain_output', domain_output)

        # NOTE: need to split it in half because there is repitition
        domain_output = domain_output.split((2))[0]
        # https://stackoverflow.com/questions/53994625/how-can-i-process-multi-loss-in-pytorch
        err_t_domain = loss_domain(domain_output, domain_label)
        
        err = err_t_domain + err_s_domain
        err.backward()
        # print(err_t_domain.item(), err_s_domain.item())
        # print(err)
        
        optimizer.step()

        # print(loss.data[0])
        # print(err)
        # print(err.data)
        

        
        nProcessed += len(data)
        
        # loss.data[0] is sum of dice coefficient over a mini-batch. By Chao.
        diceOvBatch = loss.data[0]/batch_size
        errOvBatch = err.item()/batch_size
        iteration_dice_average += diceOvBatch
        iteration_err_average += errOvBatch
        # NOTE: This was just diceOvBatch compleemnt and with %age. Now it's the real full error
        # err = 100.*(1. - diceOvBatch)
        i += 1

    num_it_average = 10
    if np.mod(iteration, num_it_average) == 0:
        iteration_dice_average /= num_it_average
        iteration_err_average /= num_it_average
        print(f'\nFor trainning: epoch: {epoch} iteration: {iteration} \tdice_coefficient over last {num_it_average} : {iteration_dice_average:.4f} and DA error : {iteration_err_average:.4f}')
        iteration_dice_average = 0
        iteration_err_average = 0
        # print('\nFor trainning: epoch: {} iteration: {} \tdice_coefficient over batch: {:.4f}\tError: {:.4f}\n'.format(
        #     epoch, iteration, diceOvBatch, err))

    return diceOvBatch, errOvBatch

def test_dice(dataManager, args, epoch, model, testLoader, testF, resultDir):
    '''
    :param dataManager: contains self.sitkImages which is a dict of test sitk images or all sitk images including test sitk images.
    :param args:
    :param epoch:
    :param model:
    :param testLoader:
    :param testF: path to file recording test results.
    :return:
    '''
    model.eval()
    test_dice = 0
    incorrect = 0
    # assume single GPU/batch_size =1
    # pdb.set_trace()
    for batch_idx, data in enumerate(testLoader):
        # NOTE: placeholder for alpha
        alpha = 0.001

        data, target, id = data
        # print("testing with {}".format(id[0]))
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = Variable(data)
        target = Variable(target)
        output, domain_output = model(data, alpha)
        dice = lossFuncs.dice_loss(output, target).data[0]
        test_dice += dice
        incorrect += (1. - dice)

        # pdb.set_trace()
        _, _, z, y, x = data.shape  # need to squeeze to shape of 3-d. by Chao.
        output = output[0, ...]  # assume batch_size = 1
        _, output = output.max(0)
        output = output.view(z, y, x)
        output = output.cpu()
        # In numpy, an array is indexed in the opposite order (z,y,x)  while sitk will generate the sitk image in (x,y,z). (refer: http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/01_Image_Basics.html)
        output = output.numpy()
        # change to simpleITK order (x, y, z)
        output = np.transpose(output, [2, 1, 0])
        # pdb.set_trace()
        print("save predicted label for test{}".format(id[0]))
        dataManager.writeResultsFromNumpyLabel(output, id[0], '_tested_epoch{}'.format(
            epoch), '.mhd', resultDir)  # require customization
        testF.write('{},{},{},{}\n'.format(epoch, id[0], dice, 1-dice))

    nTotal = len(testLoader)
    test_dice /= nTotal  # loss function already averages over batch size
    err = 100.*incorrect/nTotal
    # if np.mod(iteration, 10) == 0:
    #     print('\nFor testing: iteration:{}\tAverage Dice Coeff: {:.4f}\tError:{:.4f}\n'.format(iteration, test_dice, err))

    #### added later ##### - Jatin
    print('\nFor testing: Average Dice Coeff: {:.4f}\tError:{:.4f}\n'.format(
        test_dice, err))
    #### added later #####

    # testF.write('{},{},{}\n'.format('avarage', test_dice, err))
    testF.flush()

def test_dice_DA(dataManager, args, epoch, model, testLoader, testF, resultDir):
    '''
    :param dataManager: contains self.sitkImages which is a dict of test sitk images or all sitk images including test sitk images.
    :param args:
    :param epoch:
    :param model:
    :param testLoader:
    :param testF: path to file recording test results.
    :return:
    '''
    model.eval()
    test_dice = 0
    incorrect = 0
    # assume single GPU/batch_size =1
    # pdb.set_trace()
    for batch_idx, data in enumerate(testLoader):
        # NOTE: placeholder for alpha
        alpha = 0.001

        data, target, id = data
        # print("testing with {}".format(id[0]))
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = Variable(data)
        target = Variable(target)
        output, domain_output = model(data, alpha)
        dice = lossFuncs.dice_loss(output, target).data[0]
        test_dice += dice
        incorrect += (1. - dice)

        # pdb.set_trace()
        _, _, z, y, x = data.shape  # need to squeeze to shape of 3-d. by Chao.
        output = output[0, ...]  # assume batch_size = 1
        _, output = output.max(0)
        output = output.view(z, y, x)
        output = output.cpu()
        # In numpy, an array is indexed in the opposite order (z,y,x)  while sitk will generate the sitk image in (x,y,z). (refer: http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/01_Image_Basics.html)
        output = output.numpy()
        # change to simpleITK order (x, y, z)
        output = np.transpose(output, [2, 1, 0])
        # pdb.set_trace()
        print("save predicted label for test{}".format(id[0]))
        dataManager.writeResultsFromNumpyLabel(output, id[0], '_tested_epoch{}'.format(
            epoch), '.mhd', resultDir)  # require customization
        testF.write('{},{},{},{}\n'.format(epoch, id[0], dice, 1-dice))

    nTotal = len(testLoader)
    test_dice /= nTotal  # loss function already averages over batch size
    err = 100.*incorrect/nTotal
    # if np.mod(iteration, 10) == 0:
    #     print('\nFor testing: iteration:{}\tAverage Dice Coeff: {:.4f}\tError:{:.4f}\n'.format(iteration, test_dice, err))

    #### added later ##### - Jatin
    print('\nFor testing: Average Dice Coeff: {:.4f}\tError:{:.4f}\n'.format(
        test_dice, err))
    #### added later #####

    # testF.write('{},{},{}\n'.format('avarage', test_dice, err))
    testF.flush()


def inference(dataManager, args, loader, model, resultDir):
    model.eval()
    # assume single GPU / batch size 1
    # pdb.set_trace()
    for batch_idx, data in enumerate(loader):
        # NOTE: placeholder for alpha
        alpha = 0.001

        data, id = data
        # pdb.set_trace()
        # convert names to batch tensor
        if args.cuda:
            data.pin_memory()
            data = data.cuda()
        with torch.no_grad():
            data = Variable(data)
        output, domain_output = model(data, alpha)

        _, _, z, y, x = data.shape  # need to subset shape of 3-d. by Chao.
        output = output[0, ...]  # assume batch_size=1
        _, output = output.max(0)
        output = output.view(z, y, x)
        output = output.cpu()
        # In numpy, an array is indexed in the opposite order (z,y,x)  while sitk will generate the sitk image in (x,y,z). (refer: http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/01_Image_Basics.html)
        output = output.numpy()
        # change to simpleITK order (x, y, z)
        output = np.transpose(output, [2, 1, 0])
        # pdb.set_trace()
        print("save predicted label for inference {}".format(id[0]))
        dataManager.writeResultsFromNumpyLabel(
            output, id[0], '_inferred', '.mhd', resultDir)  # require customization

def create_transforms():
    '''Create transforms, which is the same right now for 4 different use cases'''
    # https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.functional.to_tensor
    # Convert a PIL Image or numpy.ndarray to tensor
    return transforms.Compose([transforms.ToTensor()]), transforms.Compose([transforms.ToTensor()]), \
           transforms.Compose([transforms.ToTensor()]), transforms.Compose([transforms.ToTensor()])

## main method
def main(params, args):
    manual_seed = args.seed
    random.seed(manual_seed)

    ###############      NOTE: COMMON PART STARTS        ##################
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L139
    # best_prec1 sort of can be seen here in above link as best_acc1.
    # This is used to keep track of best_acc1 achieved yet in the checkpoints
    best_prec1 = 100.  # accuracy? by Chao
    epochs = args.nEpochs
    nr_iter = args.numIterations  # params['ModelParams']['numIterations']
    batch_size = args.batchsize  # params['ModelParams']['batchsize']
    task = params['ModelParams']['task']

    # for every run, a folder is created and this is how it gets its name
    resultDir = 'results/vnet.base.{}.{}'.format(task, datestr())

    # https://becominghuman.ai/this-thing-called-weight-decay-a7cd4bcfccab
    weight_decay = args.weight_decay

    # https://pypi.org/project/setproctitle/
    # The setproctitle module allows a process to change its title (as displayed by system tools such as ps and top).
    # set title of the current process
    setproctitle.setproctitle(resultDir)

    # https://docs.python.org/3/library/shutil.html#shutil.rmtree
    if os.path.exists(resultDir):
        shutil.rmtree(resultDir)
    os.makedirs(resultDir, exist_ok=True)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # https://discuss.pytorch.org/t/what-is-manual-seed/5939/4
    # You just need to call torch.manual_seed(seed), and it will set the seed of the random number generator to a fixed value,
    # so that when you call for example torch.rand(2), the results will be reproducible.
    torch.manual_seed(manual_seed)
    if args.cuda:
        torch.cuda.manual_seed(manual_seed)

    print("build vnet")
    model = vnet.VNet(elu=False, nll=False)
    
    gpu_ids = args.gpu_ids
    # torch.cuda.set_device(gpu_ids) # why do I have to add this line? It seems the below line is useless to apply GPU devices. By Chao.
    # model = nn.parallel.DataParallel(model, device_ids=[gpu_ids])
    model = nn.parallel.DataParallel(model)

    ###############      NOTE: COMMON PART ENDS       ##################

    if not args.testonly:
        # either resume model training - in which case, pass the path to checkpoint
        # or declare initial weights
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                
                # A state_dict is simply a Python dictionary object that maps each layer to its parameter tensor.
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.evaluate, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        else:

            # https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
            # https://discuss.pytorch.org/t/parameters-initialisation/20001
            model.apply(weights_init)

        train = train_dice_DA
        test = test_dice

        print('  + Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
        if args.cuda:
            model = model.cuda()

        # get transforms
        train_transform_source, train_transform_target, test_transform_source, test_transform_target = create_transforms()

        # setting optimiser from argument
        if args.opt == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.baseLR,
                                momentum=args.momentum, weight_decay=weight_decay)  # params['ModelParams']['baseLR']
        elif args.opt == 'adam':
            optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
        elif args.opt == 'rmsprop':
            optimizer = optim.RMSprop(
                model.parameters(), weight_decay=weight_decay)

        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

        # pdb.set_trace()
        DataManagerParams = {
            'dstRes': np.asarray(eval(args.dstRes), dtype=float),
            'VolSize': np.asarray(eval(args.VolSize), dtype=int),
            'normDir': params['DataManagerParams']['normDir']
        }

        # NOTE: We need to get the training data managers for both and test for the promise12 one
    
        # NOTE: This is for the nci dataset, that will be used as target. We just need training images without labels
        dataManagerTrain = DCM.DataManager(os.path.join(basePath,'dataset_nci/imagesTr'),
                                        os.path.join(basePath,'dataset_nci/labelsTr'),
                                        params['ModelParams']['dirResult'],
                                        DataManagerParams)
        dataManagerTrain.loadTrainingData()  # required
        target_images = dataManagerTrain.getNumpyImages()
        target_labels = dataManagerTrain.getNumpyGT()

        # print(len(target_images)) # 60
        # print(len(target_labels)) # 60
        # print(type(target_images['Prostate3T-01-0022'])) # <class 'numpy.ndarray'>
        # print(target_images['Prostate3T-01-0022'].shape) # (128, 128, 64)
        # print(target_labels['Prostate3T-01-0022'].shape) # (128, 128, 64)


        # NOTE: This is the promise12 dataset. We need both train and test for them.
        # We'll get all 50 images that we have labels for and then do train/test split
        dataManager = DM.DataManager(params['ModelParams']['dirTrainImage'],
                                    params['ModelParams']['dirTrainLabel'],
                                    params['ModelParams']['dirResult'],
                                    DataManagerParams)
        dataManager.loadTrainingData()  # required
        source_images = dataManager.getNumpyImages()
        source_labels = dataManager.getNumpyGT()
        # pdb.set_trace()

        # print(len(source_images)) # 50
        # print(len(source_labels)) # 50
        # print(type(source_images['Case00'])) # <class 'numpy.ndarray'>
        # print(source_images['Case00'].shape) # (128, 128, 64)
        # print(source_labels['Case00_segmentation'].shape) # (128, 128, 64)


        train_images, train_labels, test_images, test_labels = train_test_split(source_images, source_labels, args.testProp)
        testSet = customDataset.customDataset(
            mode='test', images=test_images, task=task, GT=test_labels, transform=test_transform_source
        )
        testLoader = DataLoader(testSet, batch_size=1, shuffle=True, **kwargs)

        # print(type(testLoader)) # <class 'torch.utils.data.dataloader.DataLoader'>
        # print(len(testLoader)) # 10
        # print(type(train_images)) # <class 'dict'>
        # print(len(train_images)) # 40

        dataManager_toTestFunc = dataManager

        ### For train_images and train_labels, starting data augmentation and loading augmented data with multiprocessing
        dataQueue = Queue(30)  # max 30 images in queue?
        dataPreparation = [None] * params['ModelParams']['nProc']

        # processes creation
        for proc in range(0, params['ModelParams']['nProc']):
            # the dataAugmentation processes put the augmented training images in the dataQueue
            dataPreparation[proc] = Process(target=dataAugmentation,
                                            args=(params, args, dataQueue, train_images, train_labels))
            dataPreparation[proc].daemon = True
            dataPreparation[proc].start()

        batchData = np.zeros((batch_size, DataManagerParams['VolSize'][0],
                            DataManagerParams['VolSize'][1],
                            DataManagerParams['VolSize'][2]), dtype=float)
        batchLabel = np.zeros((batch_size, DataManagerParams['VolSize'][0],
                            DataManagerParams['VolSize'][1],
                            DataManagerParams['VolSize'][2]), dtype=float)


        ### For train_images and train_labels, starting data augmentation and loading augmented data with multiprocessing
        dataQueue_target = Queue(30)  # max 30 images in queue?
        dataPreparation_target = [None] * params['ModelParams']['nProc']

        # NOTE: same, but for target -> processes creation
        for proc in range(0, params['ModelParams']['nProc']):
            # the dataAugmentation processes put the augmented training images in the dataQueue
            dataPreparation_target[proc] = Process(target=dataAugmentation,
                                            args=(params, args, dataQueue_target, source_images, source_labels))
            dataPreparation_target[proc].daemon = True
            dataPreparation_target[proc].start()

        batchData_target = np.zeros((batch_size, DataManagerParams['VolSize'][0],
                            DataManagerParams['VolSize'][1],
                            DataManagerParams['VolSize'][2]), dtype=float)
        batchLabel_target = np.zeros((batch_size, DataManagerParams['VolSize'][0],
                            DataManagerParams['VolSize'][1],
                            DataManagerParams['VolSize'][2]), dtype=float)
        

        trainF = open(os.path.join(resultDir, 'train.csv'), 'w')
        testF = open(os.path.join(resultDir, 'test.csv'), 'w')

        print('cuda available?:', torch.cuda.is_available())

        print('Total epochs:', epochs)

        for epoch in range(1, epochs+1):
            # not working from epoch = 2 and so on. why??? By Chao.
            dataQueue_tmp = dataQueue
            dataQueue_target_tmp = dataQueue_target
            diceOvBatch = 0
            err = 0

            print(f'Startin first of total iterations ({nr_iter})')
            for iteration in range(1, nr_iter + 1):
                # adjust_opt(args.opt, optimizer, iteration+)
                if args.opt == 'sgd':
                    if np.mod(iteration, args.stepsize) == 0:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= args.gamma


                for i in range(batch_size):
                    # print(dataQueue_tmp.qsize()) # 4, then 3 in next iter
                    [defImg, defLab] = dataQueue_tmp.get()
                    # print(type(defImg)) # <class 'numpy.ndarray'>
                    # print(dataQueue_tmp.qsize()) # 3, then 2 in next iter
                    # print(defImg.shape) # (128, 128, 64)
                    # print(defLab.shape) # (128, 128, 64)

                    batchData[i, :, :, :] = defImg.astype(dtype=np.float32)
                    batchLabel[i, :, :, :] = (
                        defLab > 0.5).astype(dtype=np.float32)

                    # print(batchData.shape) # (2, 128, 128, 64)
                    # print(batchLabel.shape) # (2, 128, 128, 64)
                    

                    # NOTE: same, but for target
                    [defImg_target, defLab_target] = dataQueue_target_tmp.get()

                    batchData_target[i, :, :, :] = defImg_target.astype(dtype=np.float32)
                    batchLabel_target[i, :, :, :] = (
                        defLab_target > 0.5).astype(dtype=np.float32)

                    # print(batchData.shape) # (2, 128, 128, 64)
                    # print(batchLabel.shape) # (2, 128, 128, 64)
                

                trainSet = customDataset.customDataset(mode='train', images=batchData, GT=batchLabel,
                                                    task='promise12',
                                                    transform=train_transform_source)
                trainLoader = DataLoader(
                    trainSet, batch_size=batch_size, shuffle=True, **kwargs)


                # NOTE: same, but for target
                trainSet_target = customDataset.customDataset(mode='train', images=batchData_target, GT=batchLabel_target,
                                                    task='nci-isbi-2013',
                                                    transform=train_transform_target)
                trainLoader_target = DataLoader(
                    trainSet_target, batch_size=batch_size, shuffle=True, **kwargs)

                # print('Batch size:', batch_size)

                # print(len(trainLoader)) # 1. NOTE: Shouldn't these be equal to 2 i.e. the batch_size
                # probably this is 1, because it's 1 of the shape (2, 128, 128, 64). 
                # The batchsize is already included in dim 1
                # NOTE: found soln -> len(trainLoader.dataset) gives 2 for batch_size
                # print(len(trainLoader_target)) # 1

                

                # diceOvBatch_tmp, err_tmp = train(
                #     args, epoch, iteration, model, trainLoader, optimizer, trainF)

                diceOvBatch_tmp, err_tmp = train(
                    args, epoch, iteration, model, trainLoader, optimizer, trainF, trainLoader_target, nr_iter)

                if args.xLabel == 'Iteration':
                    trainF.write('{},{},{}\n'.format(
                        iteration, diceOvBatch_tmp, err_tmp))
                    trainF.flush()
                elif args.xLabel == 'Epoch':
                    diceOvBatch += diceOvBatch_tmp
                    err += err_tmp
            if args.xLabel == 'Epoch':
                trainF.write('{},{},{}\n'.format(
                    epoch, diceOvBatch/nr_iter, err/nr_iter))
                trainF.flush()

            if np.mod(epoch, epochs) == 0:  # default to set last epoch to save checkpoint
                save_checkpoint({'epoch': epoch,
                                'state_dict': model.state_dict(),
                                'best_prec1': best_prec1}, path=resultDir, prefix="vnet_epoch{}".format(epoch))
            if epoch == epochs and testLoader:
                # by Chao.
                test(dataManager_toTestFunc, args, epoch,
                    model, testLoader, testF, resultDir)

        os.system('./plot.py {} {} &'.format(args.xLabel, resultDir))

        trainF.close()
        testF.close()

        # inference, i.e. output predicted mask for test data
        # if params['ModelParams']['dirInferImage'] != '':
        #     print("loading inference data")
        #     dataManagerInfer = DM.DataManager(params['ModelParams']['dirInferImage'], None,
        #                                     params['ModelParams']['dirResult'],
        #                                     DataManagerParams)
        #     # required.  Create .loadInferData??? by Chao.
        #     dataManagerInfer.loadInferData()
        #     numpyImages = dataManagerInfer.getNumpyImages()

        #     inferSet = customDataset.customDataset(
        #         mode='infer', images=numpyImages, task=task, GT=None, transform=test_transform_source)

        #     inferLoader = DataLoader(inferSet, batch_size=1, shuffle=True, **kwargs)

        #     inference(dataManagerInfer, args, inferLoader, model, resultDir)
    else:
        print(f"Only running testing on the test dataset of '{params['ModelParams']['task']}' using DA model saved at '{args.testonly}'")

        # BUG: Initially this will work only for the case of using trained model of promise12 on nci, because nci has a clearer test data with labels,
        # the accuracy of which would be easier to compare
        
        # REVIEW: All experimental below
        assert not args.resume, "Cannot resume training when only testing. Remone one of the resume or testonly flags"
        
        model_path = args.testonly
        
        # load model
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            
            # A state_dict is simply a Python dictionary object that maps each layer to its parameter tensor.
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(model_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            exit()

        test = test_dice_DA

        print('  + Number of params: {}'.format(
            sum([p.data.nelement() for p in model.parameters()])))
        if args.cuda:
            model = model.cuda()

        test_transform_source = transforms.Compose([
            transforms.ToTensor()
        ])
        test_transform_target = transforms.Compose([
            transforms.ToTensor()
        ])

        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

        # pdb.set_trace()
        DataManagerParams = {
            'dstRes': np.asarray(eval(args.dstRes), dtype=float),
            'VolSize': np.asarray(eval(args.VolSize), dtype=int),
            'normDir': params['DataManagerParams']['normDir']
        }

        # if exists, means test files are given.
        if params['ModelParams']['dirTestImage']:
            print("\nloading test set")
            dataManagerTest = DCM.DataManager(params['ModelParams']['dirTestImage'], params['ModelParams']['dirTestLabel'],
                                            params['ModelParams']['dirResult'],
                                            DataManagerParams)
            dataManagerTest.loadTestingData()  # required
            test_images = dataManagerTest.getNumpyImages()
            test_labels = dataManagerTest.getNumpyGT()
            
            
            testSet = customDataset.customDataset(
                mode='test',
                images=test_images,
                GT=test_labels,

                task=task,
                
                # test_transform_source is using pytorch transform, just to convert ndarray to a tensor
                # REVIEW: shouldn't we be setting both transform and GT_transform?
                # to remind - the transformation is just converting it to tensors
                transform=test_transform_source
            )
            
            testLoader = DataLoader(testSet, batch_size=1, shuffle=True, **kwargs)

        elif args.testProp:  # if 'dirTestImage' is not given but 'testProp' is given, means only one data set is given. need to perform train_test_split.
            print('\n loading dataset, will split into train and test')
            dataManager = DM.DataManager(params['ModelParams']['dirTrainImage'],
                                        params['ModelParams']['dirTrainLabel'],
                                        params['ModelParams']['dirResult'],
                                        DataManagerParams)
            dataManager.loadTrainingData()  # required
            numpyImages = dataManager.getNumpyImages()
            numpyGT = dataManager.getNumpyGT()
            # pdb.set_trace()

            train_images, train_labels, test_images, test_labels = train_test_split(
                numpyImages, numpyGT, args.testProp)
            testSet = customDataset.customDataset(
                mode='test', images=test_images, task=task, GT=test_labels, transform=test_transform_source)
            testLoader = DataLoader(testSet, batch_size=1, shuffle=True, **kwargs)

        else:  # if both 'dirTestImage' and 'testProp' are not given, means the only one dataset provided is used as train set.
            assert False, "There needs to be a test set specified for testonly mode"

        if params['ModelParams']['dirTestImage']:
            dataManager_toTestFunc = dataManagerTest
        else:
            dataManager_toTestFunc = dataManager

        batchData = np.zeros((batch_size, DataManagerParams['VolSize'][0],
                            DataManagerParams['VolSize'][1],
                            DataManagerParams['VolSize'][2]), dtype=float)
        batchLabel = np.zeros((batch_size, DataManagerParams['VolSize'][0],
                            DataManagerParams['VolSize'][1],
                            DataManagerParams['VolSize'][2]), dtype=float)

        testF = open(os.path.join(resultDir, 'test.csv'), 'w')

        print(torch.cuda.is_available())

        epoch = 1
        test(dataManager_toTestFunc, args, epoch,
            model, testLoader, testF, resultDir)

        testF.close()