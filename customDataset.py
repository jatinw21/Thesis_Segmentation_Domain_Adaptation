import pdb
import torch
import torch.utils.data as data

import numpy as np

# We're overwriting some of the functions of the pytorch data.Dataset class here
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class customDataset(data.Dataset):
    '''
    For medical segmentation decathlon.
    '''

    def __init__(self, mode, images, GT, task, transform=None, GT_transform=None):
        if images is None:
            raise(RuntimeError("images must be set"))

        # train (for training), test (for normal testing) or infer (for unlabelled)
        self.mode = mode

        # training images and ground truth
        self.images = images
        self.GT = GT

        # NOTE: new - making this class aware of the task so changes like names of labels
        # can be made accordingly
        self.task = task
        
        # transformations
        self.transform = transform
        
        # REVIEW: why are we transforming the images but not the ground truth
        self.GT_transform = GT_transform

    # https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html
    # All datasets that represent a map from keys to data samples should subclass
    # it. All subclasses should overrite :meth:`__getitem__`, supporting fetching a
    # data sample for a given key. Subclasses could also optionally overwrite
    # :meth:`__len__`
    def __getitem__(self, index):
        """
        Args:
            index(int): Index
        Returns:
            tuple: (image, GT) where GT is index of the
        """
        if self.mode == "train":
            # keys = list(self.images.keys())
            # id = keys[index]
            # because of data augmentation, train images are stored in a 4-d array, with first d as sample index.
            image = self.images[index]
            # print("image shape from DataManager shown in PROMISE12:" + str(image.shape)) # e.g. 96,96,48.
            
            # REVIEW: Why transpose is needed here?
            image = np.transpose(image,[2,1,0]) # added by Chao
            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.expand_dims.html
            image = np.expand_dims(image, axis=0)
            # print("expanded image dims:{}".format(str(image.shape)))
            # pdb.set_trace()
            image = image.astype(np.float32)
            if self.transform is not None:
                image = torch.from_numpy(image)
                # image = self.transform(image)

            GT = self.GT[index]
            GT = np.transpose(GT, [2, 1, 0])
            if self.GT_transform is not None:
                GT = self.GT_transform(GT)
            return image, GT

        elif self.mode == "test":
            keys = list(self.images.keys())
            id = keys[index]
            image = self.images[id]
            
            # REVIEW: Why transpose is needed here?
            image = np.transpose(image, [2, 1, 0])  # added by Chao
            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.expand_dims.html
            image = np.expand_dims(image, axis=0)
            # print("expanded image dims:{}".format(str(image.shape)))
            # pdb.set_trace()
            image = image.astype(np.float32)
            if self.transform is not None:
                image = torch.from_numpy(image)
                # image = self.transform(image)

            # FIXED. NOTE: requires customisation
            if self.task == 'nci-isbi-2013'
                GT = self.GT[id+'_truth']
            else:
                GT = self.GT[id+'_segmentation'] # require customization
            
            GT = np.transpose(GT, [2, 1, 0])
            if self.GT_transform is not None:
                GT = self.GT_transform(GT)
            return image, GT, id

        elif self.mode == "infer":# added by Chao
            keys = list(self.images.keys())
            id = keys[index]
            image = self.images[id]
            # print("image shape from DataManager shown in PROMISE12:" + str(image.shape))
            
            # REVIEW: Why transpose is needed here?
            image = np.transpose(image,[2,1,0]) # added by Chao
            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.expand_dims.html
            image = np.expand_dims(image, axis=0)
            image = image.astype(np.float32)
            return image, id

    def __len__(self):
        return len(self.images)
