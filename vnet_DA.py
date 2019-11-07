import torch
import torch.nn as nn
import torch.nn.functional as F
from functions import ReverseLayerF

import pdb

def passthrough(x, **kwargs):
    return x

def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

# normalization between sub-volumes is necessary
# for good performance
class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
            super()._check_input_dim(input) # added 4 spaces to indent this line into "if". not sure if right by Chao??
        # super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(16)
        self.relu1 = ELUCons(elu, 16)

    def forward(self, x):
        # do we want a PRELU here as well?
        # print("x before InputTransition shape:"+str(x.shape))
        out = self.bn1(self.conv1(x))
        # split input in to 16 channels (or is it right to say duplicate the input for 16 times to operate "torch.add()"?? By Chao) 
        x16 = torch.cat((x, x, x, x, x, x, x, x,
                         x, x, x, x, x, x, x, x), 1) # changed dim = 0 to 1 to operate on channels, and have "x16" the same size as "out" to operate "torch.add()". By Chao.
        # print("x16 shape:"+str(x16.shape))
        # pdb.set_trace()
        out = self.relu1(torch.add(out, x16))
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, elu, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, 2, kernel_size=1)
        self.relu1 = ELUCons(elu, 2)
        if nll:
            self.softmax = F.log_softmax
        else:
            self.softmax = F.softmax

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        # print("out shape before softmax:"+str(out.shape))

        # flatten z, y, x
        b, c, z, y, x = out.shape # b:batch_size, c:channels, z:depth, y:height, w:width. channels is 2? as the output channels of the last conv layer?
        out = out.view(b, c, -1)

        # pdb.set_trace()
        res = self.softmax(out, dim = 1)

        # make channels the last axis
        # out = out.permute(0, 2, 3, 4, 1).contiguous()
        # out = out.view(out.numel() // 2, 2)
        # out = self.softmax(out,dim=1)
        return res


class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu=True, nll=False):
        super(VNet, self).__init__()
        self.in_tr = InputTransition(16, elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)

        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, elu, nll)



        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(256 * 4 * 4 * 8, 512))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(512))
        # self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(512, 64))
        self.domain_classifier.add_module('d_bn2', nn.BatchNorm1d(64))
        self.domain_classifier.add_module('d_fc3', nn.Linear(64, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    
    def forward(self, x, alpha):
        # print("x before model shape:"+str(x.shape))
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        # print("out16(in_tr) shape:"+str(out16.shape))
        # print("out32(down_tr32) shape:"+str(out32.shape))
        # print("out64(down_tr64) shape:"+str(out64.shape))
        # print("out128(down_tr128) shape:"+str(out128.shape))
        out256 = self.down_tr256(out128)
        # print("out256(down_tr256) shape:"+str(out256.shape))

        feature = out256.view(-1, 256 * 4 * 4 * 8)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        domain_output = self.domain_classifier(reverse_feature)

        out = self.up_tr256(out256, out128)
        # print("up_tr256 shape:"+str(out.shape))
        out = self.up_tr128(out, out64)
        # print("up_tr128 shape:"+str(out.shape))
        out = self.up_tr64(out, out32)
        # print("up_tr64 shape:"+str(out.shape))
        out = self.up_tr32(out, out16)
        # print("up_tr32 shape:"+str(out.shape))
        out = self.out_tr(out)
        # print("out_tr shape:"+str(out.shape))
        return out, domain_output
