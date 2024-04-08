import torch
import random
import numpy as np
from torch import nn
import torch.nn.functional as F

SEED = 1130
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def _rot(rot):
    cos_r, sin_r = rot.cos(), rot.sin()
    zeros = rot.new(rot.size()[:2] + (1,)).zero_()
    ones = rot.new(rot.size()[:2] + (1,)).fill_(1)

    r1 = torch.stack((ones, zeros, zeros),dim=-1)
    rx2 = torch.stack((zeros, cos_r[:,:,0:1], sin_r[:,:,0:1]), dim = -1)
    rx3 = torch.stack((zeros, -sin_r[:,:,0:1], cos_r[:,:,0:1]), dim = -1)
    rx = torch.cat((r1, rx2, rx3), dim = 2)

    ry1 = torch.stack((cos_r[:,:,1:2], zeros, -sin_r[:,:,1:2]), dim =-1)
    r2 = torch.stack((zeros, ones, zeros),dim=-1)
    ry3 = torch.stack((sin_r[:,:,1:2], zeros, cos_r[:,:,1:2]), dim =-1)
    ry = torch.cat((ry1, r2, ry3), dim = 2)

    rz1 = torch.stack((cos_r[:,:,2:3], sin_r[:,:,2:3], zeros), dim =-1)
    r3 = torch.stack((zeros, zeros, ones),dim=-1)
    rz2 = torch.stack((-sin_r[:,:,2:3], cos_r[:,:,2:3],zeros), dim =-1)
    rz = torch.cat((rz1, rz2, r3), dim = 2)

    rot = rz.matmul(ry).matmul(rx)
    return rot

def _reshape(x):
    return x.contiguous().view(x.size()[:2] + (-1, 3))

def _copy_np_arr(x):
    return x.detach().cpu().numpy().copy()

def _to_tensor_and_reshape(temp):
    x = torch.from_numpy(temp).to(torch.float)
    x = x.contiguous().view(x.size()[:2] + (-1,))
    return x

def _rotation(x, theta):
    x = _reshape(x)
    rot = x.new(x.size()[0],3).uniform_(-theta, theta)
    rot = rot.repeat(1, x.size()[1])
    rot = rot.contiguous().view((-1, x.size()[1], 3))
    rot = _rot(rot)

    x = torch.transpose(x, 2, 3)
    x = torch.matmul(rot, x)
    x = torch.transpose(x, 2, 3)
    x = x.contiguous().view(x.size()[:2] + (-1,))
    return x

# Ken 2021-03-09 新增 Shear function
def _shear(x):
    x = _reshape(x)
    temp = _copy_np_arr(x)
    s1_list = [random.uniform(-1, 1),random.uniform(-1, 1),random.uniform(-1, 1)]
    s2_list = [random.uniform(-1, 1),random.uniform(-1, 1), random.uniform(-1, 1)]
    R = np.array([
        [1         , s1_list[0], s2_list[0]],
        [s1_list[1], 1         , s2_list[1]],
        [s1_list[2], s2_list[2], 1         ]
    ])
    R = R.transpose()
    temp = np.dot(temp, R)
    return _to_tensor_and_reshape(temp)

# Ken 2021-03-10 新增 Gaussian blur function
class GaussianBlurConv(nn.Module):
    def __init__(self, channels, kernel, sigma):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        self.kernel = kernel
        self.min_max_sigma = sigma
        radius = int(kernel / 2)
        self.kernel_index = np.arange(-radius, radius + 1)

    def __call__(self, x):
        sigma = random.uniform(self.min_max_sigma[0], self.min_max_sigma[1])
        blur_flter = np.exp(-np.power(self.kernel_index, 2.0) / (2.0 * np.power(sigma, 2.0)))
        kernel = torch.from_numpy(blur_flter).unsqueeze(0).unsqueeze(0).float()
        kernel = kernel.repeat(self.channels, 1, 1, 1) # (3,1,1,15)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        prob = np.random.random_sample()
        x = torch.from_numpy(x)
        if prob < 0.5:
            # x = x.permute(3,0,2,1) # M,C,V,T
            x = x.permute(0,3,2,1) # M,C,V,T
            x = F.conv2d(x, self.weight, padding=(0, int((self.kernel - 1) / 2 )),   groups=self.channels)
            # x = x.permute(1,-1,-2, 0) #C,T,V,M
            x = x.permute(0,3,2,1) #C,T,V,M

        return x.numpy()

def _gaus_blur(x):
    # Initialize Gaussian filter
    g = GaussianBlurConv(3, 15, [0.1, 2])
    x = _reshape(x)
    temp = _copy_np_arr(x)
    return _to_tensor_and_reshape(g(temp)) # Apply filter

# Ken 2021-03-12 新增 Joint mask
def _joint_mask(x):

    x = _reshape(x)
    temp = _copy_np_arr(x)
    skeleton_num, frame_num, vertexes_num, coordinates = temp.shape # 1, 20, 25, 3

    # joint
    random_int =  random.randint(5, 15)
    all_joints = [i for i in range(vertexes_num)]
    joint_list_ = random.sample(all_joints, random_int)
    joint_list_ = sorted(joint_list_)

    # frame
    random_int = random.randint(10, 20)
    all_frames = [i for i in range(frame_num)]
    time_range_ = random.sample(all_frames, random_int)
    time_range_ = sorted(time_range_)

    # 生成對應 sample 的 mask
    x_new = np.zeros((skeleton_num, len(time_range_), len(joint_list_), coordinates))
    # print("data_numpy",data_numpy[:, time_range, joint_list, :].shape)
    temp2 = temp[:, time_range_, :, :].copy()
    temp2[:, :, joint_list_, :] = x_new
    temp[:, time_range_, :, :] = temp2
    
    return _to_tensor_and_reshape(temp)

# Ken 2021-03-12 新增 Channel Mask
def _channel_mask(x):
    x = _reshape(x)
    temp = _copy_np_arr(x)
    temp = temp.transpose([3, 0, 1, 2])

    zero_x = np.zeros((1, 20, 25))
    target_channel = random.randint(0,2)
    target_channel = 0 # Ken 測試用
    temp[target_channel] = zero_x

    temp = temp.transpose([1, 2, 3, 0])
    return _to_tensor_and_reshape(temp)

# Ken 2021-04-01 新增 Mirror
def _mirror(x):
    """
    對骨架做鏡射的翻轉
    """
    x = _reshape(x)
    temp = _copy_np_arr(x)
    temp = temp.transpose([3, 0, 1, 2])
    target_channel = random.randint(0,2)
    target_channel = 0 # Ken 測試用
    temp[target_channel] = temp[target_channel] * -1 # Mirror 就是將特定維度乘上 -1
    temp = temp.transpose([1, 2, 3, 0])
    return _to_tensor_and_reshape(temp)

def spatial_transform(x, spatial_transform_list, rotation_theta = 0.5):
    # TODO:多種參數時，新增機率機制
    if "rot" in spatial_transform_list and random.random()> 0.5:
    #if "rot" in spatial_transform_list:
        x = _rotation(x, rotation_theta)
        #print("rot")

    if "sh" in spatial_transform_list and random.random()> 0.5:
    #if "sh" in spatial_transform_list:
        x = _shear(x)
        #print("sh")

    # Guassian Blur
    if "GB" in spatial_transform_list and random.random()> 0.5:
    #if "GB" in spatial_transform_list:
        x = _gaus_blur(x)
        #print("GB")

    # Joint Mask
    #if "JM" in spatial_transform_list and random.random()> 0.5:
    if "JM" in spatial_transform_list:
        x = _joint_mask(x)
        #print("JM")
    
    # Channel Mask
    if "CM" in spatial_transform_list and random.random()> 0.5:
    #if "CM" in spatial_transform_list:
        x = _channel_mask(x)
        #print("CM")

    if "MI" in spatial_transform_list and random.random()> 0.5:
    #if "MI" in spatial_transform_list:
        x = _mirror(x)
        #print("MI")

    return x

def temporal_transform(x, temporal_transform_list, mask_num = 2):
    # reverse
    if "re" in temporal_transform_list:
        x = torch.flip(x, [1])

    # shuffle
    if "shu" in temporal_transform_list:
        idx = torch.randperm(x.shape[1])
        x = x[:, idx]

    # speed up
    if "SU" in temporal_transform_list and random.random()> 0.5:
    #if "SU" in temporal_transform_list:
        transform_x = x.clone()
        for idx in range(x.shape[1]//2):
            if idx % 2 == 0:
                transform_x[:, idx]= x[:, idx//2]
            else:
                transform_x[:, idx] = (x[:, idx//2] + x[:, (idx//2)+1])/2
        x = transform_x

    # Random Mask
    if "RM" in temporal_transform_list and random.random()> 0.5:
    # if "RM" in temporal_transform_list:

        for f in range(x.shape[0]):
            idx = torch.randperm(x.shape[1])[:mask_num]
            x[f, idx] = torch.zeros_like(x[0,0,:])

    return x

def jm_SSL(x):
    x = _reshape(x)
    temp = _copy_np_arr(x)
    skeleton_num, frame_num, vertexes_num, coordinates = temp.shape # 1, 20, 25, 3
    #joint
    if vertexes_num == 25:
        mask_type = [[6,7,21,22],[10,11,23,24],[5,6,7,21,22],[9,10,11,23,24],[14,15],[18,19],[13,14,15],[17,18,19],[]]
    elif vertexes_num == 20:
        mask_type = [[6,7],[10,11],[5,6,7],[9,10,11],[14,15],[18,19],[13,14,15],[17,18,19],[]]
    probability = [0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05, 0.4]
    #probability = [0.5/6, 0.5/6, 0.5/12, 0.5/12, 0.5/6, 0.5/6, 0.5/12, 0.5/12, 0.5]
    #probability = [0.4/6, 0.4/6, 0.4/12, 0.4/12, 0.4/6, 0.4/6, 0.4/12, 0.4/12, 0.6]
    m_type_ = np.asarray([np.random.choice(9, frame_num, p=probability) for _ in range(skeleton_num)])
    #m_type_ = np.asarray([np.random.choice(9, frame_num) for _ in range(skeleton_num)])
    
    for f in range(skeleton_num):
        for i in range(frame_num):
            m_t = mask_type[m_type_[f][i]]
            temp[f, i, m_t, :] = np.zeros_like(temp[f, i, m_t, :])
  
    return _to_tensor_and_reshape(temp), x.view(skeleton_num, frame_num, -1)

def fm_SSL(x, percent=0.2):

    temp = x.clone()
    #gt = temp [:,-mask_num:,:]
    #temp[:,-mask_num:,:] = torch.zeros_like(temp[0,-mask_num:,:])
    for f in range(temp.shape[0]):
        idx = torch.randperm(temp.shape[1])[:int(percent*temp.shape[1])]
        temp[f, idx] = torch.zeros_like(temp[0,0,:])

    return temp, idx