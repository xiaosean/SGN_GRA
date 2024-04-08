from torch import nn
import torch
import math
from torch.optim.lr_scheduler import _LRScheduler
import os
import warnings
import copy

class SGN(nn.Module):
    def __init__(self, config, bias = True):
        super(SGN, self).__init__()

        self.dim1 = 256
        self.representation_dim = 1024
        self.dataset = config["dataset"]
        self.seg = config["seg"]
        self.num_joint = config["num-joint"]
        bs = config["batch-size"]

        self.tem_embed = embed(self.seg, 64*4, self.num_joint, norm=False, bias=bias)
        self.spa_embed = embed(self.num_joint, 64, norm=False, bias=bias)
        self.joint_embed = embed(3, 64, self.num_joint, norm=True, bias=bias)
        self.dif_embed = embed(3, 64, self.num_joint, norm=True, bias=bias)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.sp_maxpool = nn.AdaptiveMaxPool2d((1, 20))
        self.sp_avgpool = nn.AvgPool2d((1, 20))
        self.cnn = local(self.dim1, self.representation_dim, bias=bias)
        self.compute_g1 = compute_g_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn1 = gcn_spa(self.dim1 // 2, self.dim1 // 2, bias=bias)
        self.gcn2 = gcn_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn3 = gcn_spa(self.dim1, self.dim1, bias=bias)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        nn.init.constant_(self.gcn1.w.cnn.weight, 0)
        nn.init.constant_(self.gcn2.w.cnn.weight, 0)
        nn.init.constant_(self.gcn3.w.cnn.weight, 0)

    def one_hot(self, bs, spa, tem):

        y = torch.arange(spa).unsqueeze(-1)
        y_onehot = torch.FloatTensor(spa, spa)

        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)

        y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)
        y_onehot = y_onehot.repeat(bs, tem, 1, 1)

        return y_onehot


    def forward(self, input):
        
        # Dynamic Representation
        bs, step, dim = input.size()

        self.spa = self.one_hot(bs, self.num_joint, self.seg)
        self.spa = self.spa.permute(0, 3, 2, 1).cuda()
        self.tem = self.one_hot(bs, self.seg, self.num_joint)
        self.tem = self.tem.permute(0, 3, 1, 2).cuda()

        num_joints = dim //3
        input = input.view((bs, step, num_joints, 3))
        input = input.permute(0, 3, 2, 1).contiguous()
        dif = input[:, :, :, 1:] - input[:, :, :, 0:-1]
        dif = torch.cat([dif.new(bs, dif.size(1), num_joints, 1).zero_(), dif], dim=-1)
        pos = self.joint_embed(input)
        tem1 = self.tem_embed(self.tem)
        spa1 = self.spa_embed(self.spa)
        dif = self.dif_embed(dif)
        dy = pos + dif
        # Joint-level Module
        input= torch.cat([dy, spa1], 1)
        g = self.compute_g1(input)
        #input = pos
        #g = self.compute_g1(input)
        input = self.gcn1(input, g)
        input = self.gcn2(input, g)
        input = self.gcn3(input, g)
        spatial_info = self.sp_avgpool(self.sp_maxpool(input))
        # Frame-level Module
        input = input + tem1
        input = self.cnn(input)
        # Classification
        output = self.maxpool(input)
        
        # print("spatial_info: ", spatial_info.size())
        # print("output: ", output.size())
        # exit()

        return output, spatial_info

class norm_data(nn.Module):
    def __init__(self, dim = 64, num_joint = 25):
        super(norm_data, self).__init__()

        self.bn = nn.BatchNorm1d(dim * num_joint)

    def forward(self, x):
        bs, c, num_joints, step = x.size()
        x = x.view(bs, -1, step)
        #print(x.shape)
        x = self.bn(x)
        x = x.view(bs, -1, num_joints, step).contiguous()
        return x

class embed(nn.Module):
    def __init__(self, dim = 3, dim1 = 128, num_joint = 25, norm = True, bias = False):
        super(embed, self).__init__()

        if norm:
            self.cnn = nn.Sequential(
                norm_data(dim, num_joint),
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )

    def forward(self, x):
        x = self.cnn(x)
        return x

class cnn1x1(nn.Module):
    def __init__(self, dim1 = 3, dim2 =3, bias = True):
        super(cnn1x1, self).__init__()
        self.cnn = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.cnn(x)
        return x

class local(nn.Module):
    def __init__(self, dim1 = 3, dim2 = 3, bias = False):
        super(local, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((1, 20))
        self.cnn1 = nn.Conv2d(dim1, dim1, kernel_size=(1, 3), padding=(0, 1), bias=bias)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU()
        self.cnn2 = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim2)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x1):
        x1 = self.maxpool(x1)
        x = self.cnn1(x1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x
        
class local(nn.Module):
    def __init__(self, dim1 = 3, dim2 = 3, bias = False):
        super(local, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((1, 20))
        self.cnn1 = nn.Conv2d(dim1, dim1, kernel_size=(1, 3), padding=(0, 1), bias=bias)
        self.bn1 = nn.BatchNorm2d(dim1)

        self.cnn2 = nn.Conv2d(dim1, dim2, kernel_size=(1, 3), padding=(0, 1), bias=bias)
        self.bn2 = nn.BatchNorm2d(dim2)
        
        self.cnn3 = nn.Conv2d(dim2, dim2, kernel_size=(1, 3), padding=(0, 1), bias=bias)
        self.bn3 = nn.BatchNorm2d(dim2)
        
        #self.cnn4 = nn.Conv2d(dim2, dim2, kernel_size=1, bias=bias)
        self.cnn4 = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)
        self.bn4 = nn.BatchNorm2d(dim2)
        self.dropout = nn.Dropout2d(0.2)
        self.relu = nn.ReLU()

    def forward(self, x1):

        x1 = self.maxpool(x1)
        x = self.cnn1(x1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.cnn4(x)
        x = self.bn4(x)
        x = self.relu(x)
        return x

class gcn_spa(nn.Module):
    def __init__(self, in_feature, out_feature, bias = False):
        super(gcn_spa, self).__init__()
        self.bn = nn.BatchNorm2d(out_feature)
        self.relu = nn.ReLU()
        self.w = cnn1x1(in_feature, out_feature, bias=False)
        self.w1 = cnn1x1(in_feature, out_feature, bias=bias)


    def forward(self, x1, g):
        x = x1.permute(0, 3, 2, 1).contiguous()
        x = g.matmul(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.w(x) + self.w1(x1)
        x = self.relu(self.bn(x))
        return x

class compute_g_spa(nn.Module):
    def __init__(self, dim1 = 64 *3, dim2 = 64*3, bias = False):
        super(compute_g_spa, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.g1 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.g2 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1):

        g1 = self.g1(x1).permute(0, 3, 2, 1).contiguous()
        g2 = self.g2(x1).permute(0, 3, 1, 2).contiguous()
        g3 = g1.matmul(g2)
        g = self.softmax(g3)
        return g

class Linear_clf(nn.Module):
    def __init__(self, num_classes, dim, layer_num):
        super(Linear_clf, self).__init__()
        # Ken 2021-03-07 新增 layer_num 參數控制 linear 層數，嘗試 1 ~ 3 層的效果
        self.layer_num = layer_num
        if layer_num == 3:
            # Ken 2021-03-07 改回 3 層做 semi supervised 的實驗
            self.fc1 = nn.Linear(dim, dim//2)
            self.fc2 = nn.Linear(dim//2, dim//4)
            self.fc3 = nn.Linear(dim//4, num_classes)
        elif layer_num == 2:
            self.fc1 = nn.Linear(dim, dim//2)
            self.fc2 = nn.Linear(dim//2, num_classes)
        else:
            self.fc1 = nn.Linear(dim, num_classes)

    def forward(self, x):
        # Ken 2021-03-07 新增 layer_num 參數控制 linear 層數，嘗試 1 ~ 3 層的效果
        if self.layer_num == 3:
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
        elif self.layer_num == 2:
            x = self.fc1(x)
            x = self.fc2(x)
        else:
            x = self.fc1(x)

        return x


def save_checkpoint(net, clf, epoch, config, script_name, save_format):
    if clf == None:
        state = {
            'net': copy.deepcopy(net.state_dict()),
            'epoch': epoch,
            'script': script_name
        }
    else:
        state = {
            'net': copy.deepcopy(net.state_dict()),
            'clf': copy.deepcopy(clf.state_dict()),
            'epoch': epoch,
            'script': script_name
        }


    destination = f'./checkpoint/{config["dataset"]}/{config["case"]}/{config["data-volume"]}/{config["ckpt-name"]}/{save_format}'
    print(f'Saving checkpoint to {destination}')
    dir_path = "/".join(destination.split("/")[:-1])
    if not os.path.isdir(dir_path):
        # os.mkdir(dir_path)
        os.makedirs(dir_path, exist_ok=True)
    torch.save(state, destination)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            #print(true_dist)
            #print(torch.sum(-true_dist * pred, dim=self.dim))
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
