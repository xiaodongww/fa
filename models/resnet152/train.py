import os
import time
import torch
torch.cuda.set_device(2)
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable

import kres152
import FashionData

LR = 1e-2
losswei = [1,0.1]
viswei = [10,10,1]


clothes_type = 'trousers'  # 7158 images
postfix = '{}_dense161_'.format(clothes_type)

num_keypoints = len(FashionData.class_points[clothes_type])

# skirt
# train_sel = (0, 6618)
# val_sel = (6618, 7618)

# blouse
# train_sel = (0, 6158)
# val_sel = (6158, 7158)

# dress
# train_sel = (0, 4400)
# val_sel = (4400, 4912)

# trousers
train_sel = (0, 5600)
val_sel = (5600, 6347)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose(
    [transforms.ToTensor(), normalize])

trainSet = FashionData.fashionSet(selected=train_sel, classname=clothes_type,
                                  root='/home/wuxiaodong/fa/train/Images/',
                                  train='train', transform=transform)
valSet = FashionData.fashionSet(selected=val_sel, classname=clothes_type,
                                 root='/home/wuxiaodong/fa/train/Images/',
                                 train='val', transform=transform)

trainloader = torch.utils.data.DataLoader(trainSet, batch_size=40, shuffle=True,num_workers=2)
valloader = torch.utils.data.DataLoader(valSet, batch_size=40, shuffle=False,num_workers=2)

# fa = fashionSet(selected=(0, 1000), train='val', transform=train_transform)
# train_loader = torch.utils.data.DataLoader(fa, batch_size=10, shuffle=False)

# prepare net


# criterion = nn.MSELoss()

state = {}
state['loss_window'] = [0] * 200
state['total_batch_num'] = 0
state['epoch'] = 0
state['loss_avg'] = 0.0
state['ne'] = 0.0

def train():
    net.train()
    optimizer = torch.optim.SGD([{'params': net.res152.parameters(), 'lr': LR * 0.1},
                                 {'params': net.outputs.parameters(), 'lr': LR}, ], momentum=0.9, weight_decay=0.0005,
                                nesterov=True)
    for batch_idx, (data, label)in enumerate(trainloader):
        state['total_batch_num'] += 1
        data = Variable(data.cuda())
        label = Variable(label.cuda())
        output = net(data)

        optimizer.zero_grad()
        loss = criterion(output, label)
        # loss = criterion(output[:, :2*num_keypoints], label[:, :2*num_keypoints])
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


def test():
    num = 0
    norm_deviation_sum = 0

    net.eval()

    for batch_idx, (data, label) in enumerate(valloader):
        data = Variable(data.cuda(), volatile=True)
        output = net(data)
        output = output.cpu().data

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


def test_loss():
    loss_sum = 0

    net.eval()
    for batch_idx, (data, label) in enumerate(valloader):
        data = Variable(data.cuda(), volatile=True)
        label = Variable(label.cuda())
        output = net(data)
        # loss = criterion(output, label)
        loss = criterion(output[:, :2*num_keypoints], label[:, :2*num_keypoints])

        loss_sum += loss.data[0]
        del loss, output, label
    print('test loss = {}'.format(loss_sum/batch_idx))




if __name__ == '__main__':
    net = kres152.get_kres152(num_keypoints=len(FashionData.class_points[clothes_type]))
    net.cuda()
    criterion = kres152.KEYPointLoss(weight=losswei, vis_weight=viswei, num_keypoints=len(FashionData.class_points[clothes_type])).cuda()

    for epoch in range(300):
        if epoch == 10:
            losswei = [1, 0.01]
            viswei = [10, 10, 1]
            LR = LR * 0.1
        if epoch == 100:
            losswei = [1, 0.001]
            viswei = [10, 10, 1]
            LR = LR * 0.1 
        state['epoch'] = epoch
        train()
        test()
        if epoch %20 == 0 :
            torch.save(net.state_dict(), os.path.join(os.path.dirname(__file__),'snapshots' ,postfix+'_epoch{}.pth'.format(epoch)))

