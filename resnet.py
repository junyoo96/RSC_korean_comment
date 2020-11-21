from torch import nn
from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck
import torch
from torch import nn as nn
from torch.autograd import Variable
import numpy.random as npr
import numpy as np
import torch.nn.functional as F
import random
import math

class ResNet(nn.Module):
    def __init__(self, block, layers, jigsaw_classes=1000, classes=100):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) # output feature map shape = (512,7,7)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.jigsaw_classifier = nn.Linear(512 * block.expansion, jigsaw_classes)
        self.class_classifier = nn.Linear(512 * block.expansion, classes)
        #self.domain_classifier = nn.Linear(512 * block.expansion, domains)
        self.pecent = 1/3

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def is_patch_based(self):
        return False

    def forward(self, x, gt=None, flag=None, epoch=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #train시만 수행 
        if flag:
            interval = 10
            #10epoch마다 pecent(batch percentage) update
            if epoch % interval == 0:
                self.pecent = 3.0 / 10 + (epoch / interval) * 2.0 / 10

            #eval 모드로 변경 
            self.eval()
            #x_new : x와 gradinet 계산이 상관없는 복사된 tensor
            
            x_new = x.clone().detach()
            
            #Variable : Graph에서 x_new.data를 노드로 표현하기 위해 사용
            x_new = Variable(x_new.data, requires_grad=True)
            #x_new shape : (128,512,7,7)
            #x_new_view shape : (128,512,1,1)
            x_new_view = self.avgpool(x_new)
            
            #x_new_view shape : (128,512)
            x_new_view = x_new_view.view(x_new_view.size(0), -1)
            output = self.class_classifier(x_new_view)
            # print("output",output.shape)
            class_num = output.shape[1]
            index = gt
            #num_rois : 128
            num_rois = x_new.shape[0]
            #num_channel : 512
            num_channel = x_new.shape[1]
            # H(Height) : 7 
            H = x_new.shape[2]
            # HW(Height * Width) = 7*7 = 49
            HW = x_new.shape[2] * x_new.shape[3]
            one_hot = torch.zeros((1), dtype=torch.float32).cuda()
            one_hot = Variable(one_hot, requires_grad=False)
            #sp_i shape = (2,128)
            sp_i = torch.ones([2, num_rois]).long()
            sp_i[0, :] = torch.arange(num_rois)
            # print("sp_i_0",sp_i[0])
            sp_i[1, :] = index
            # print("sp_i_1",sp_i[1])
            #sp_v shape : (128)
            sp_v = torch.ones([num_rois])
            #one_hot_sparse : 라벨 index들 one hot vector로 변경
                #one_hot_sparse shape : (128, 7)   
            one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()
            # print("one_hot_sparse",one_hot_sparse.shape,one_hot_sparse)
            one_hot_sparse = Variable(one_hot_sparse, requires_grad=False)
            #one_hot : class score vector인 output에서 label에 대응되는 class score값만 추출한 값을 모두 더한것
            one_hot = torch.sum(output * one_hot_sparse)
            #?
            self.zero_grad()
            #original input x를 copy한 x_new_view에 대해 backward해서 gradient 계산
            one_hot.backward()
            #grads_val : x_new_view의 gradient 계산한 값 
            grads_val = x_new.grad.clone().detach()
            # print("grads_val",grads_val.shape,grads_val)
            #feature map 각 channel의 gradient 평균을 계산 
                #grad_channel_mean shape : (128,512)
            grad_channel_mean = torch.mean(grads_val.view(num_rois, num_channel, -1), dim=2)
            channel_mean = grad_channel_mean
            grad_channel_mean = grad_channel_mean.view(num_rois, num_channel, 1, 1)
            #x_new의 각 channel마다 각channel의 gradient 평균을 곱하고 각 channel의 픽셀마다 값을 합함
                #spatial_mean shape : (128,7,7)
            spatial_mean = torch.sum(x_new * grad_channel_mean, 1)
            print("shape",(x_new * grad_channel_mean).shape)
            print("shape2",spatial_mean.shape)
            #spatial_mean shape : 128, 49
                #각 input의 feature의 gradient값이 저장되어 있음
            spatial_mean = spatial_mean.view(num_rois, HW)
            self.zero_grad()

            #50% 확률에 따라 spatial-wise 또는 channel wise 선택
            choose_one = random.randint(0, 9)
            if choose_one <= 9:
                # ---------------------------- spatial -----------------------
                #HW보다 크거나 같은 숫자 중 가장 작은 정수(HW가 49면,16), feature drop percentage인 33.3% 인듯
                spatial_drop_num = math.ceil(HW * 1 / 3.0)
                print(spatial_drop_num)
                #th18_mask_value shape : (128)
                    # 이게 뭔지 잘 모르겠음
                    # 각 채널의 spatial_mean을 sort해서 상위 33.3%에 위치한 값 찾기
                th18_mask_value = torch.sort(spatial_mean, dim=1, descending=True)[0][:, spatial_drop_num]
                ##th18_mask_value shape : (128,49)
                th18_mask_value = th18_mask_value.view(num_rois, 1).expand(num_rois, 49)
                # print("th18_mask_value",th18_mask_value.shape,th18_mask_value)
                #mask_all_cuda : mask feature map만드는 과정
                    # spatial_mean이 아까 찾은 상위 33.3%에 위치한 값보다 크면 mask feature map에서 그 자리를 0으로 채우고
                    # 아니면, 1로 채움 
                    # shape : (128,49)
                mask_all_cuda = torch.where(spatial_mean > th18_mask_value, torch.zeros(spatial_mean.shape).cuda(),
                                            torch.ones(spatial_mean.shape).cuda())
                # print("mask_all_cuda",mask_all_cuda.shape, mask_all_cuda)
                #mask_all : 실제 사용될 mask feature map 
                    # shape : (128,1, 7, 7)
                mask_all = mask_all_cuda.reshape(num_rois, H, H).view(num_rois, 1, H, H)
            else:
                # -------------------------- channel ----------------------------
                vector_thresh_percent = math.ceil(num_channel * 1 / 3.2)
                vector_thresh_value = torch.sort(channel_mean, dim=1, descending=True)[0][:, vector_thresh_percent]
                vector_thresh_value = vector_thresh_value.view(num_rois, 1).expand(num_rois, num_channel)
                vector = torch.where(channel_mean > vector_thresh_value,
                                     torch.zeros(channel_mean.shape).cuda(),
                                     torch.ones(channel_mean.shape).cuda())
                mask_all = vector.view(num_rois, num_channel, 1, 1)

            # ----------------------------------- batch ----------------------------------------
            cls_prob_before = F.softmax(output, dim=1)
            x_new_view_after = x_new * mask_all
            x_new_view_after = self.avgpool(x_new_view_after)
            x_new_view_after = x_new_view_after.view(x_new_view_after.size(0), -1)
            x_new_view_after = self.class_classifier(x_new_view_after)
            cls_prob_after = F.softmax(x_new_view_after, dim=1)

            sp_i = torch.ones([2, num_rois]).long()
            sp_i[0, :] = torch.arange(num_rois)
            sp_i[1, :] = index
            sp_v = torch.ones([num_rois])
            one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()
            before_vector = torch.sum(one_hot_sparse * cls_prob_before, dim=1)
            after_vector = torch.sum(one_hot_sparse * cls_prob_after, dim=1)
            change_vector = before_vector - after_vector - 0.0001
            change_vector = torch.where(change_vector > 0, change_vector, torch.zeros(change_vector.shape).cuda())
            th_fg_value = torch.sort(change_vector, dim=0, descending=True)[0][int(round(float(num_rois) * self.pecent))]
            drop_index_fg = change_vector.gt(th_fg_value).long()
            ignore_index_fg = 1 - drop_index_fg
            not_01_ignore_index_fg = ignore_index_fg.nonzero()[:, 0]
            mask_all[not_01_ignore_index_fg.long(), :] = 1

            self.train()
            mask_all = Variable(mask_all, requires_grad=True)
            x = x * mask_all

        
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        
        return self.class_classifier(x)


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model
