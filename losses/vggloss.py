import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.nn import CrossEntropyLoss
from torchvision import models
import numpy as np
from utils.transforms import create_part


class FeatureL2Norm(torch.nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) +
                         epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)


class NNLoss(nn.Module):

    def __init__(self):
        super(NNLoss, self).__init__()

    def forward(self, predicted, ground_truth, nh=5, nw=5):
        v_pad = nh // 2
        h_pad = nw // 2
        val_pad = nn.ConstantPad2d(
            (v_pad, v_pad, h_pad, h_pad), -10000)(ground_truth)

        reference_tensors = []
        for i_begin in range(0, nh):
            i_end = i_begin - nh + 1
            i_end = None if i_end == 0 else i_end
            for j_begin in range(0, nw):
                j_end = j_begin - nw + 1
                j_end = None if j_end == 0 else j_end
                sub_tensor = val_pad[:, :, i_begin:i_end, j_begin:j_end]
                reference_tensors.append(sub_tensor.unsqueeze(-1))

        reference = torch.cat(reference_tensors, dim=-1)
        ground_truth = ground_truth.unsqueeze(dim=-1)

        predicted = predicted.unsqueeze(-1)
        # return reference, predicted
        abs = torch.abs(reference - predicted)
        # sum along channels
        norms = torch.sum(abs, dim=1)
        # min over neighbourhood
        loss, _ = torch.min(norms, dim=-1)
        # loss = torch.sum(loss)/self.batch_size
        loss = torch.mean(loss)
        return loss


class VGGLoss(torch.nn.Module):
    def __init__(self, model_path, requires_grad=False):
        super(VGGLoss, self).__init__()
        self.model = models.vgg19()
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint)
        self.model = torch.nn.DataParallel(self.model).cuda()
        vgg_pretrained_features = self.model.module.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.loss = nn.L1Loss().cuda()
        self.lossmse = nn.MSELoss().cuda()
        self.norm = FeatureL2Norm().cuda()  # before cal loss
        self.nnloss = NNLoss().cuda()
        # vgg19_bn: 53 layers || vgg19 : 37 layers || vgg19 2-7-12-21-30  before relu
        for x in range(2):
            self.slice1.add_module(str(x),
                                   vgg_pretrained_features[x])  # conv1_2
        for x in range(7):
            self.slice2.add_module(str(x),
                                   vgg_pretrained_features[x])  # conv2_2
        for x in range(12):
            self.slice3.add_module(str(x),
                                   vgg_pretrained_features[x])  # conv3_2
        for x in range(21):
            self.slice4.add_module(str(x),
                                   vgg_pretrained_features[x])  # conv4_2
        for x in range(30):
            self.slice5.add_module(str(x),
                                   vgg_pretrained_features[x])  # conv5_2

        self.slice = [
            self.slice1, self.slice2, self.slice3, self.slice4, self.slice5
        ]

        for i in range(len(self.slice)):
            self.slice[i] = torch.nn.DataParallel(self.slice[i]).cuda()
        # self.nnloss = torch.nn.DataParallel(self.loss)

        if not requires_grad:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self,
                pred,
                target,
                target_parse,
                masksampled,
                gram,
                nearest,
                use_l1=True):

        weight = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4,
                  1.0]  # more high level info
        weight.reverse()
        loss = 0
        # print(self.slice[0](pred).shape)
        if gram:
            loss_conv12 = self.lossmse(self.gram(self.slice[0](pred)),
                                       self.gram(self.slice[0](target)))
        elif nearest:
            loss_conv12 = self.nnloss(self.slice[0](pred),
                                      self.slice[0](target))
        else:
            loss_conv12 = self.loss(self.slice[0](pred), self.slice[0](target))
            # reference, predicted = self.loss(self.norm(self.slice[0](pred)), self.norm(self.slice[0](target)))
            # abs = torch.abs(reference - predicted)
            # # sum along channels
            # norms = torch.sum(abs, dim=1)
            # # min over neighbourhood
            # loss,_ = torch.min(norms, dim=-1)
            # # loss = torch.sum(loss)/self.batch_size
            # loss_conv12 = torch.mean(loss)

        for i in range(5):
            if not masksampled:
                if gram:
                    gram_pred = self.gram(self.slice[i](pred))
                    gram_target = self.gram(self.slice[i](target))
                else:
                    gram_pred = self.slice[i](pred)
                    gram_target = self.slice[i](target)
                if use_l1:
                    loss = loss + weight[i] * self.loss(gram_pred, gram_target)
                else:
                    loss = loss + weight[i] * self.lossmse(
                        gram_pred, gram_target)
            else:
                pred = create_part(pred, target_parse, 'cloth')
                target = create_part(pred, target_parse, 'cloth')
                if gram:
                    gram_pred = self.gram(self.slice[i](pred))
                    gram_target = self.gram(self.slice[i](target))
                else:
                    gram_pred = self.slice[i](pred)
                    gram_target = self.slice[i](target)
                if use_l1:
                    loss = loss + weight[i] * self.loss(gram_pred, gram_target)
                else:
                    loss = loss + weight[i] * self.lossmse(
                        gram_pred, gram_target)
        return loss, loss_conv12

    # Calculate Gram matrix (G = FF^T)
    def gram(self, x):
        (bs, ch, h, w) = x.size()
        f = x.view(bs, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (ch * h * w)
        return G
