import os
from os import listdir
from os.path import isfile, join, splitext

import numpy as np
import SimpleITK as sitk

# What is isotropical scaling? https://stackoverflow.com/questions/43577231/what-is-anisotropic-scaling-in-computer-vision
# dataset isotropically scaled to 1x1x1.5mm, volume resized to 128x128x64
# The datasets were first normalised using the N4 bias filed correction function of the ANTs framework


class DataManager(object):
    # params=None
    # srcFolder=None
    # resultsDir=None
    #
    # fileList=None
    # gtList=None
    #
    # sitkImages=None
    # sitkGT=None
    # meanIntensityTrain = None

    def __init__(self, imageFolder, GTFolder, resultsDir, parameters):
        self.params = parameters
        self.imageFolder = imageFolder
        # GT folder is the train label folder
        self.GTFolder = GTFolder
        self.resultsDir = resultsDir

    def createImageFileList(self):
        '''Training images list'''
        self.imageFileList = [f for f in listdir(self.imageFolder)]
        # NOTE: uncomment later
        # print('imageFileList: ' + str(self.imageFileList))

    def createGTFileList(self):
        '''Training images segmentation (labels) list'''
        self.GTFileList = [f for f in listdir(self.GTFolder)]
        # NOTE: uncomment later
        # print('GTFileList: ' + str(self.GTFileList))

    def loadImages(self):
        self.sitkImages = dict()
        rescalFilt = sitk.RescaleIntensityImageFilter()
        rescalFilt.SetOutputMaximum(1)
        rescalFilt.SetOutputMinimum(0)

        stats = sitk.StatisticsImageFilter()
        m = 0.

        reader = sitk.ImageSeriesReader()
        # image_viewer = sitk.ImageViewer()
        # image_viewer.SetApplication('/Applications/ITK-SNAP.app/Contents/MacOS/ITK-SNAP')

        for folder in self.imageFileList:
            # the folder name is set as the id
            id = folder
            # print(folder)
            curr_folder_path = join(self.imageFolder, folder)
            # print(curr_folder_path)

            dicom_names = reader.GetGDCMSeriesFileNames(curr_folder_path)
            # print(dicom_names)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()

            # size = image.GetSize()
            # print( "Image size:", size[0], size[1], size[2] )

            self.sitkImages[id] = rescalFilt.Execute(
                sitk.Cast(
                    image,
                    sitk.sitkFloat32
                )
            )

            stats.Execute(self.sitkImages[id])
            m += stats.GetMean()

        self.meanIntensityTrain = m/len(self.sitkImages)
        # print(self.meanIntensityTrain)

    def loadGT(self):
        self.sitkGT = dict()

        for f in self.GTFileList:
            # the filename before extension is set as the id
            id = f.split('.')[0]
            # print(id)

            # Not using 0.45 threshold from here, because the results of this repo are just okay - 70%
            # https://github.com/mirzaevinom/prostate_segmentation/blob/master/codes/train.py
            self.sitkGT[id] = sitk.Cast(
                sitk.ReadImage(join(self.GTFolder, f)) > 0.5,
                sitk.sitkFloat32
            )

    def loadTrainingData(self):
        self.createImageFileList()
        self.createGTFileList()
        self.loadImages()
        self.loadGT()

    def loadTestingData(self):
        self.createImageFileList()
        self.createGTFileList()
        self.loadImages()
        self.loadGT()

    def loadInferData(self):
        self.createImageFileList()
        self.loadImages()

    def getNumpyImages(self):
        # http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/20_Expand_With_Interpolators.html
        dat = self.getNumpyData(self.sitkImages, sitk.sitkLinear)

        for key in dat.keys():  
            # why restrict to >0? By Chao.
            # https://github.com/faustomilletari/VNet/blob/master/VNet.py, line 147. For standardization?
            # why > 0? nothing should be -ve? and we're just excluding amssive amounts of 0, so mean calc is faster?
            mean = np.mean(dat[key][dat[key] > 0])
            std = np.std(dat[key][dat[key] > 0])

            # z-normalisation
            dat[key] -= mean
            dat[key] /= std

        return dat

    def getNumpyGT(self):
        dat = self.getNumpyData(self.sitkGT, sitk.sitkLinear)

        for key in dat:
            dat[key] = (dat[key] > 0.5).astype(dtype=np.float32)

        return dat

    def getNumpyData(self, dat, method):
        # return dictionary
        ret = dict()

        # for each image
        for key in dat:

            # Filling the volume we want to convert the current image into by 0s
            # VolSize --> default = [128, 128, 64]
            ret[key] = np.zeros([self.params['VolSize'][0], self.params['VolSize']
                                 [1], self.params['VolSize'][2]], dtype=np.float32)
            # get current img
            img = dat[key]

            # we rotate the image according to its transformation using the direction and according to the final spacing we want
            # dstRes --> default=[1, 1, 1.5]
            # https://itk.org/Doxygen/html/classitk_1_1ImageBase.html#aaadef7c0a9627cf22b0fbf2de6de913c

            # not sure how the factor thing exactly works - why we divide by dstRes --> what does that mean
            # https://stackoverflow.com/questions/54895602/difference-between-image-resizing-and-space-changing
            # Old spacing vs new spacing ratio, to get the factor difference
            factor = np.asarray(img.GetSpacing()) / [self.params['dstRes'][0], self.params['dstRes'][1],
                                                     self.params['dstRes'][2]]

            # get the size of new image with the new spacing. size of image depends on spacing
            factorSize = np.asarray(img.GetSize() * factor, dtype=float)

            # the newSize will be the transformed one after new spacing, but at least [128, 128, 64]
            # This is because that is the central area we need for our model
            newSize = np.max([factorSize, self.params['VolSize']], axis=0)

            newSize = newSize.astype(dtype='int')

            # http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/21_Transforms_and_Resampling.html
            # An affine transformation is any transformation that preserves collinearity (i.e., all points lying on a line 
            # initially still lie on a line after transformation) and ratios of distances 
            # (e.g., the midpoint of a line segment remains the midpoint after transformation).
            # https://itk.org/SimpleITKDoxygen/html/classitk_1_1simple_1_1AffineTransform.html
            # 3 is the number of dimensions
            T = sitk.AffineTransform(3)

            # from above link while affine scaling, setMatrix used
            T.SetMatrix(img.GetDirection())

            # first resample the original image to new size and spacing and then crop it to get region of interest
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(img)
            resampler.SetOutputSpacing(
                [self.params['dstRes'][0], self.params['dstRes'][1], self.params['dstRes'][2]])
            resampler.SetSize(newSize.tolist())

            # why use linear transformation?
            # SimpleITK has a large number of interpolators. In most cases linear interpolation, the default setting, is sufficient.
            # https://simpleitk.readthedocs.io/en/master/Documentation/docs/source/registrationOverview.html#interpolator
            resampler.SetInterpolator(method)

            # transformation not recommended in main file
            if self.params['normDir']:
                resampler.SetTransform(T.GetInverse())

            imgResampled = resampler.Execute(img)

            # taking the center part here
            # divide the whole thing by 2, to get central point of the newSize. 
            # So in case [128,128,64]. It will be the point [64,64,32]
            imgCentroid = np.asarray(newSize, dtype=float) / 2.0

            # if the newSize is same as vol, then this will be 0,0,0. Otherwise, if more, than more accordingly
            # This would be the starting pixel for region extraction of the central volSize.
            imgStartPx = (
                imgCentroid - self.params['VolSize'] / 2.0
            ).astype(dtype='int')

            # https://itk.org/SimpleITKDoxygen/html/classitk_1_1simple_1_1RegionOfInterestImageFilter.html#details
            regionExtractor = sitk.RegionOfInterestImageFilter()
            size_2_set = self.params['VolSize'].astype(dtype='int')
            regionExtractor.SetSize(size_2_set.tolist())
            regionExtractor.SetIndex(imgStartPx.tolist())

            imgResampledCropped = regionExtractor.Execute(imgResampled)

            # get the final resampled and cropped image
            ret[key] = np.transpose(
                sitk.GetArrayFromImage(imgResampledCropped).astype(dtype=float), 
                [2, 1, 0]
            )

        return ret

    def writeResultsFromNumpyLabel(self, result, key, resultTag, ext, resultDir):
        '''
        :param result: predicted mask
        :param key: sample id
        :return: register predicted mask (e.g. binary mask of size 96x96x48) to original image (e.g. CT volume of size 320x320x20), output the final mask of the same size as original image.
        '''
        img = self.sitkImages[key]  # original image
        # print("original img shape{}".format(img.GetSize()))

        toWrite = sitk.Image(img.GetSize()[0], img.GetSize()[
                             1], img.GetSize()[2], sitk.sitkFloat32)

        factor = np.asarray(img.GetSpacing()) / [self.params['dstRes'][0], self.params['dstRes'][1],
                                                 self.params['dstRes'][2]]

        factorSize = np.asarray(img.GetSize() * factor, dtype=float)

        newSize = np.max([factorSize, self.params['VolSize']], axis=0)

        newSize = newSize.astype(dtype=int)

        T = sitk.AffineTransform(3)
        T.SetMatrix(img.GetDirection())

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img)
        resampler.SetOutputSpacing(
            [self.params['dstRes'][0], self.params['dstRes'][1], self.params['dstRes'][2]])
        resampler.SetSize(newSize.tolist())
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)

        if self.params['normDir']:
            resampler.SetTransform(T.GetInverse())

        toWrite = resampler.Execute(toWrite)

        imgCentroid = np.asarray(newSize, dtype=float) / 2.0

        imgStartPx = (
            imgCentroid - self.params['VolSize'] / 2.0).astype(dtype=int)

        for dstX, srcX in zip(range(0, result.shape[0]), range(imgStartPx[0], int(imgStartPx[0]+self.params['VolSize'][0]))):
            for dstY, srcY in zip(range(0, result.shape[1]), range(imgStartPx[1], int(imgStartPx[1]+self.params['VolSize'][1]))):
                for dstZ, srcZ in zip(range(0, result.shape[2]), range(imgStartPx[2], int(imgStartPx[2]+self.params['VolSize'][2]))):
                    try:
                        toWrite.SetPixel(int(srcX), int(srcY), int(
                            srcZ), float(result[dstX, dstY, dstZ]))
                    except:
                        pass

        resampler.SetOutputSpacing(
            [img.GetSpacing()[0], img.GetSpacing()[1], img.GetSpacing()[2]])
        resampler.SetSize(img.GetSize())

        if self.params['normDir']:
            resampler.SetTransform(T)

        toWrite = resampler.Execute(toWrite)

        thfilter = sitk.BinaryThresholdImageFilter()
        thfilter.SetInsideValue(1)
        thfilter.SetOutsideValue(0)
        thfilter.SetLowerThreshold(0.5)
        toWrite = thfilter.Execute(toWrite)

        #connected component analysis (better safe than sorry)

        cc = sitk.ConnectedComponentImageFilter()
        toWritecc = cc.Execute(sitk.Cast(toWrite, sitk.sitkUInt8))

        arrCC = np.transpose(sitk.GetArrayFromImage(
            toWritecc).astype(dtype=float), [2, 1, 0])

        lab = np.zeros(int(np.max(arrCC)+1), dtype=float)

        for i in range(1, int(np.max(arrCC)+1)):
            lab[i] = np.sum(arrCC == i)

        activeLab = np.argmax(lab)

        toWrite = (toWritecc == activeLab)

        toWrite = sitk.Cast(toWrite, sitk.sitkUInt8)

        writer = sitk.ImageFileWriter()

        #print join(self.resultsDir, filename + '_result' + ext)
        writer.SetFileName(join(resultDir, key + resultTag + ext))
        writer.Execute(toWrite)
