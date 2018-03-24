import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class localCNN(nn.Module):
    def __init__(self):
        super(localCNN, self).__init__()
        self.


class localfeaFromImage(nn.Module):
    def __init__(self,  num_keypoints):
        super(localfeaFromFeature, self).__init__()
        self.dense161 = models.densenet161(pretrained=True)
        self.outputs = nn.Linear(2208, 5*num_keypoints)  # each keypoint, there is 2 for the location and 3 for the visibility
        self.localcnns = nn.ModuleList()

    def forward(self, x, points):
        """
        points: [x1, y1, x2, y2, ...,xn, yn]
        location in resized image = (x+0.5)*resize
        """
        features = self.dense161.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.outputs(out)
        return out

class localfeaFromFeature(nn.Module):
    def __init__(self,  num_keypoints):
        super(localfeaFromFeature, self).__init__()
        self.dense161 = models.densenet161(pretrained=True)
        self.outputs = nn.Linear(2208, 5*num_keypoints)  # each keypoint, there is 2 for the location and 3 for the visibility

    def forward(self, x, points):
        """
        points: torch.Tensor()  batchsize X (2*num_keypoints)
        location in feature map = (x+0.5)*featuremap_size
        """
        x = self.dense161.features.conv0(x)
        x = self.dense161.features.norm0(x)
        x = self.dense161.features.relu0(x)
        x = self.dense161.features.pool0(x)
        x = self.dense161.features.denseblock1(x)
        x1 = self.dense161.features.transition1(x)

        x2 =  self.dense161.features.denseblock2(x1)
        x2 =  self.dense161.features.transition2(x2)

        x3 =  self.dense161.features.denseblock3(x2)
        x3 =  self.dense161.features.transition3(x3)

        x4 = self.dense161.features.denseblock4(x3)
        features = self.dense161.features.norm5(x4)

        # features = self.dense161.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        
        # out = self.dense161.classifier(out)
        out = self.outputs(out)

        # local features
        local_feas = []
        local_feas.append(out)
        for i in range(len(points)/2):
            featuremap_size = x1.size()[2]
            channel_num = x1.size()[1]
            loc_x = int((points[2*i] + 0.5) * featuremap_size)
            loc_y = int((points[2*i+1] + 0.5) * featuremap_size)
            locafea = x1[:,:,loc_x,loc_y]
            local_feas.append(locafea)
        local_fea = torch.cat(local_feas, 1)
        out = torch.cat((out, local_fea), 1)



        return  out

def get_kres152(state_dict=None, **kwargs):
    net = localfeaFromFeature(**kwargs)
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
# net = models.densenet161()
# net.load_state_dict(state_dict=torch.load("/home/wuxiaodong/.torch/models/densenet161-8d451a50.pth"))

# netout = net(input)
# localnetout = localnet(input)

# input = Variable(torch.rand(10,3,224,224))
# localnet = get_kres152(state_dict=torch.load("/home/wuxiaodong/.torch/models/densenet161-8d451a50.pth"), num_keypoints=200)
# out = localnet(input, [0.3,0.4,0.5,0.6,0.2,0.4,0.6,0.5])


class test(nn.Module):
    def __init__(self):
        super(test,self).__init__()
        self.cls = nn.Linear(10,10)
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])
    def forward(self, x):
        return self.cls(x)

net = test()