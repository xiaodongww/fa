import os
import time
import torch
torch.cuda.set_device(3)
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable

import kres152
import FashionData
from train import *

# losswei[1] must be 0
losswei = [1,0]
viswei = [0,0,0]

LR = 0.01

postfix = 'stage2_dense161_'
net1_path = "/home/wuxiaodong/fa/models/resnet152/snapshots/dense161__epoch280.pth"
net1 = kres152.get_kres152(state_dict=torch.load(net1_path),num_keypoints=num_keypoints)
net1.cuda()

net2 = kres152.get_kres152(state_dict=torch.load(net1_path),num_keypoints=num_keypoints)
net2.cuda()

criterion = kres152.KEYPointLoss(weight=losswei, vis_weight=viswei, num_keypoints=num_keypoints).cuda()


trainloader = torch.utils.data.DataLoader(trainSet, batch_size=20, shuffle=True,num_workers=2)
valloader = torch.utils.data.DataLoader(valSet, batch_size=20, shuffle=False,num_workers=2)

def train2():
    net1.train()
    net2.train()
    optimizer = torch.optim.SGD([{'params': net2.res152.parameters(), 'lr': LR },
                             {'params': net2.outputs.parameters(), 'lr': LR}, 
                             {'params': net1.parameters(), 'lr': LR*0.1},], momentum=0.9, weight_decay=0.0005,
                            nesterov=True)

    for batch_idx, (data, label)in enumerate(trainloader):
        state['total_batch_num'] += 1
        data = Variable(data.cuda())
        label = Variable(label.cuda())

        # data.volatile = True
        # output1 = Variable(net1(data).data)
        output1 = net1(data)

        # data.volatile = False
        output2 = net2(data)

        tmp = Variable(output2.data, requires_grad=False)
        optimizer.zero_grad()
        loss = criterion(label-output1[:, :3*num_keypoints], tmp[:, :3*num_keypoints])
        loss.backward()
        optimizer.step()

        tmp = Variable((label-output1[:, :3*num_keypoints]).data, requires_grad=False)
        optimizer.zero_grad()
        loss = criterion(output2[:, :3*num_keypoints], tmp)
        loss.backward()
        optimizer.step()




        state['loss_window'][state['total_batch_num'] % 20] = float(loss.data[0])

        if state['total_batch_num']>=20:
            loss_avg = sum(state['loss_window']) / float(20)
        else:
            loss_avg = sum(state['loss_window']) / float(state['total_batch_num'])
        state['loss_avg'] = loss_avg
        
        display = 20
        if (batch_idx + 1) % display == 0:
            toprint = '{}, LR: {}, epoch: {}, batch id: {}, avg_loss: {}, ne: {}'.format(
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), \
                LR, state['epoch'], batch_idx, round(loss_avg, 5), round(state['ne'], 5))
            print(toprint)
            with open(os.path.join(os.path.dirname(__file__), postfix+'_log.txt'), 'a') as f:
                f.write(toprint + '\n')

def test2():
    num = 0
    norm_deviation_sum = 0
    net1.eval()
    net2.eval()

    for batch_idx, (data, label) in enumerate(valloader):
        data = Variable(data.cuda(), volatile=True)
        output1 = net1(data)
        output1 = output1.cpu().data

        output2 = net2(data)
        output2 = output2.cpu().data

        output = output1 + output2

        batch_size = label.size()[0]
        distm = nn.PairwiseDistance(p=2)
        point1, point2 = FashionData.metric_points_location[clothes_type]

        norm_dis_vec = distm(label[:, (2*point1, 2*point1+1)], label[:, (2*point2, 2*point2+1)])

        for m in range(batch_size):
            if norm_dis_vec[m][0] < 1e-4:
                print('using back up nomalize distance')
                back_point1, back_point2 = FashionData.metric_points_location[clothes_type+'_backup']
                norm_dis_vec[m] = distm(label[:, (2*back_point1, 2*back_point1+1)], label[:, (2*back_point2, 2*back_point2+1)])[m]

        for i in range(num_keypoints):

            deviation = distm(label[:, (i*2, i*2+1)], output[:, (i*2, i*2+1)])
            norm_deviation = deviation / norm_dis_vec
            for j in range(batch_size):
                # label is {0, 1, 2} and 0 means point not exist
                if label[j][2*num_keypoints+i] != 2:
                    pass
                else:
                    num += 1
                    norm_deviation_sum += norm_deviation[j][0]
                    if norm_deviation[j][0]>100:
                        print(norm_deviation[j][0])
                        print(batch_idx, i, j)
                        

    ne = norm_deviation_sum/num
    state['ne'] = ne
    print('ne={}'.format(ne))


if __name__ == '__main__':
    for epoch in range(300):
        if epoch == 20:
            LR = LR * 0.1
        if epoch == 50:
            LR = LR * 0.1 
        state['epoch'] = epoch
        train2()
        test2()
        if epoch %20 == 0 :
            torch.save(net2.state_dict(), os.path.join(os.path.dirname(__file__),'snapshots' ,postfix+'net2_epoch{}.pth'.format(epoch)))
            torch.save(net1.state_dict(), os.path.join(os.path.dirname(__file__),'snapshots' ,postfix+'net1_epoch{}.pth'.format(epoch)))


