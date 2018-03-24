import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# class kres152(nn.Module):
#     def __init__(self,  num_keypoints):
#         super(kres152, self).__init__()
#         self.res152 = models.resnet152(pretrained=True)
#         self.outputs = nn.Linear(2048, 5*num_keypoints)  # each keypoint, there is 2 for the location and 3 for the visibility

#     def forward(self, x):
#         x = self.res152.conv1(x)
#         x = self.res152.bn1(x)
#         x = self.res152.relu(x)
#         x = self.res152.maxpool(x)

#         x = self.res152.layer1(x)
#         x = self.res152.layer2(x)
#         x = self.res152.layer3(x)
#         x = self.res152.layer4(x)

#         x = self.res152.avgpool(x)
#         x = x.view(x.size(0), -1)

#         x = self.outputs(x)

#         return x

class kres152(nn.Module):
    def __init__(self,  num_keypoints):
        super(kres152, self).__init__()
        self.res152 = models.densenet161(pretrained=True)
        self.outputs = nn.Linear(2208, 5*num_keypoints)  # each keypoint, there is 2 for the location and 3 for the visibility

    def forward(self, x):
        features = self.res152.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.outputs(out)
        return out

def get_kres152(state_dict=None, **kwargs):
    net = kres152(**kwargs)
    own_state = net.state_dict()
    if state_dict is not None:
        for name, param in state_dict.items():
            try:
                if name not in own_state:
                    continue
                else:
                    own_state[name].copy_(param)
            except:
                if 'weight' in name:
                    nn.init.xavier_normal(own_state[name])
                else:
                    own_state[name].zero_()
    else:
        for name, param in own_state.items():
            if 'outputs' in name:
                print(name)
                if 'weight' in name:
                    nn.init.xavier_normal(own_state[name])
                else:
                    own_state[name].zero_()

    return net

class KEYPointLoss(nn.Module):
    def __init__(self, num_keypoints, weight=[1,1], vis_weight=[0.5, 0.45, 0.05]):
        super(KEYPointLoss, self).__init__()
        self.num_keypoints = num_keypoints
        self.weight = weight
        self.vis_criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(vis_weight))
        self.location_criterion = nn.MSELoss(reduce=False)

    def forward(self, input, target):

        vis_loss = Variable(torch.Tensor([0]).type(torch.cuda.FloatTensor))
        if self.weight[1] != 0:
            for i in range(self.num_keypoints):
                if 'cuda' in str(type(target.data)):
                    visibility = target[:, 2*self.num_keypoints+i].type(torch.cuda.LongTensor)
                    vis_loss += self.vis_criterion(input[:, 2*self.num_keypoints+3*i:2*self.num_keypoints+3*i+3],
                                              visibility)
                else:
                    visibility = target[:, 2*self.num_keypoints+i].type(torch.LongTensor)
                    vis_loss += self.vis_criterion(input[:, 2*self.num_keypoints+3*i:2*self.num_keypoints+3*i+3],
                                              visibility)
        
        if 'cuda' in str(type(target.data)):
            location_mask = torch.ones(target.size()[0], 2 * self.num_keypoints).type(torch.cuda.FloatTensor)
        else:
            location_mask = torch.ones(target.size()[0], 2 * self.num_keypoints).type(torch.FloatTensor)

        for i in range(target.size()[0]):
            for j in range(self.num_keypoints):
                # label is {0, 1, 2} and 0 means point not exist
                if target.data[i][2*self.num_keypoints+j] == 0:
                    location_mask[i][2*j] = 0
                    location_mask[i][2*j+1] = 0
        location_mask = Variable(location_mask)


        location_loss = (self.location_criterion(input[:, :self.num_keypoints*2], target[:, :self.num_keypoints*2]) * location_mask).mean()
        # for i in range(self.num_keypoints):
        #     location_loss += self.location_criterion(input[:, i*2:i*2+2], target[:, i*2:i*2+2])

        with open('loss_tmp4.txt', 'a') as f:
            f.write('{}\t{}\n'.format(self.weight[0]*location_loss.data[0], self.weight[1]*vis_loss.data[0]))
        return self.weight[0]*location_loss + self.weight[1]*vis_loss

if __name__ == '__main__':
    import FashionData
    num_keypoints = len(FashionData.class_points['blouse'])
    net = get_kres152(num_keypoints=num_keypoints)
    lossm = KEYPointLoss(num_keypoints=num_keypoints, weight=[1, 0], vis_weight=[0,0,0])
    label = Variabel(torch.rand(10, ))
    loss = lossm(net(FashionData.data), FashionData.label)
