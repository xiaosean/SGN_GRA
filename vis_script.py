import os
import h5py
import glob
import random
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D

import torch
from torch.utils.data import Dataset, DataLoader
from data import NTUDataLoaders
from tqdm import tqdm

import itertools

SEED = 1130
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

SKELETON_NODES_PARENTS = [-1, 0, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18, 1, 7, 7, 11, 11]

# NTU RGB+D 60/120 Action Classes
ACTION_NAMES = {
    1: "drink water",
    2: "eat meal/snack",
    3: "brushing teeth",
    4: "brushing hair",
    5: "drop",
    6: "pickup",
    7: "throw",
    8: "sitting down",
    9: "standing up (from sitting position)",
    10: "clapping",
    11: "reading",
    12: "writing",
    13: "tear up paper",
    14: "wear jacket",
    15: "take off jacket",
    16: "wear a shoe",
    17: "take off a shoe",
    18: "wear on glasses",
    19: "take off glasses",
    20: "put on a hat/cap",
    21: "take off a hat/cap",
    22: "cheer up",
    23: "hand waving",
    24: "kicking something",
    25: "reach into pocket",
    26: "hopping (one foot jumping)",
    27: "jump up",
    28: "make a phone call/answer phone",
    29: "playing with phone/tablet",
    30: "typing on a keyboard",
    31: "pointing to something with finger",
    32: "taking a selfie",
    33: "check time (from watch)",
    34: "rub two hands together",
    35: "nod head/bow",
    36: "shake head",
    37: "wipe face",
    38: "salute",
    39: "put the palms together",
    40: "cross hands in front (say stop)",
    41: "sneeze/cough",
    42: "staggering",
    43: "falling",
    44: "touch head (headache)",
    45: "touch chest (stomachache/heart pain)",
    46: "touch back (backache)",
    47: "touch neck (neckache)",
    48: "nausea or vomiting condition",
    49: "use a fan (with hand or paper)/feeling warm",
    50: "punching/slapping other person",
    51: "kicking other person",
    52: "pushing other person",
    53: "pat on back of other person",
    54: "point finger at the other person",
    55: "hugging other person",
    56: "giving something to other person",
    57: "touch other person's pocket",
    58: "handshaking",
    59: "walking towards each other",
    60: "walking apart from each other",
    61: "put on headphone",
    62: "take off headphone",
    63: "shoot at the basket",
    64: "bounce ball",
    65: "tennis bat swing",
    66: "juggling table tennis balls",
    67: "hush (quite)",
    68: "flick hair",
    69: "thumb up",
    70: "thumb down",
    71: "make ok sign",
    72: "make victory sign",
    73: "staple book",
    74: "counting money",
    75: "cutting nails",
    76: "cutting paper (using scissors)",
    77: "snapping fingers",
    78: "open bottle",
    79: "sniff (smell)",
    80: "squat down",
    81: "toss a coin",
    82: "fold paper",
    83: "ball up paper",
    84: "play magic cube",
    85: "apply cream on face",
    86: "apply cream on hand back",
    87: "put on bag",
    88: "take off bag",
    89: "put something into a bag",
    90: "take something out of a bag",
    91: "open a box",
    92: "move heavy objects",
    93: "shake fist",
    94: "throw up cap/hat",
    95: "hands up (both hands)",
    96: "cross arms",
    97: "arm circles",
    98: "arm swings",
    99: "running on the spot",
    100: "butt kicks (kick backward)",
    101: "cross toe touch",
    102: "side kick",
    103: "yawn",
    104: "stretch oneself",
    105: "blow nose",
    106: "hit other person with something",
    107: "wield knife towards other person",
    108: "knock over other person (hit with body)",
    109: "grab other person’s stuff",
    110: "shoot at other person with a gun",
    111: "step on foot",
    112: "high-five",
    113: "cheers and drink",
    114: "carry something with other person",
    115: "take a photo of other person",
    116: "follow other person",
    117: "whisper in other person’s ear",
    118: "exchange things with other person",
    119: "support somebody with hand",
    120: "finger-guessing game (playing rock-paper-scissors)",
}

def load_data():
    train_X = None
    train_Y = None
    with h5py.File("./data/ntu/NTU_CS_subset_1_percent.h5" , 'r') as f:
        train_X = f['train_x'][:]
        train_Y = f['train_y'][:]
    # print(train_X.shape)
    # print(train_Y.shape)
    return train_X, train_Y

def count_valid_frame(action):
    valid_num = 0
    if action.shape[0] == 300:
        for frame_num in range(300):
            pos = action[frame_num]
            if np.count_nonzero(pos) != 0:
                valid_num += 1
    else:
        valid_num = 19

    return valid_num

def data_vis(action, label, aug_type=None):
    """
    閱讀數據格式
    x : 3D coordinate
    y : label
    """
    action_name = ACTION_NAMES[label + 1]
    # print("1 action data:\n", action.shape) # 一個動作有 300 個 Frame
    # print("Action name: ", action_name)
    # 由於部份動作是有對象的，所以每個 Frame 包含兩個人的資料 ( 150 個數值都是 3D 座標 )
    # 可用 int(action.shape[1] / 2) = 75 作為 index 切分成兩組資訊
    seperate_index = int(action.shape[1] / 2)

    # # 抽出第一個 Frame (只是用來理解數據格式)
    # skeletons = action[0]
    # print("First frame:\n", skeletons)
    # # 分離兩個人的骨架資訊
    # s_1 = skeletons[:seperate_index]
    # s_2 = skeletons[seperate_index:]
    # print("1st person:\n", s_1)
    # print("2nd person:\n", s_2)

    # 共用參數
    radius = 1.5
    filneame = "Action_{}_{}.gif".format(label, action_name.replace("/", "_or_").replace(" ", "_"))
    total_frame_num = action.shape[0]
    fps = 30

    # 計算有效的 Frame 數量
    valid_frame_num = count_valid_frame(action)

    # # 計算有效的 Frame 數量
    # valid_frame_num = 0
    # for frame_num in range(300):
    #     pos = action[frame_num]
    #     if np.count_nonzero(pos) != 0:
    #         valid_frame_num += 1

    # 繪圖設定
    plt.ioff()
    fig = plt.figure()
    fig.suptitle('3D skeleton - {}'.format(action_name), fontsize=16)
    
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlim3d([-radius/2, radius/2])
    ax.set_zlim3d([-radius/2, radius/2])
    ax.set_ylim3d([-radius/2, radius/2])
    ax.dist = 12.5

    def plot_skeleton(ax, s, color):
        for j, j_parent in enumerate(SKELETON_NODES_PARENTS):
            if j_parent == -1:
                continue # 從第二個點開始 畫 跟前一個點的 Edge

            # c : coordinates
            c = [
                s[3*j    ], s[3*j_parent    ],
                s[3*j + 2], s[3*j_parent + 2],
                s[3*j + 1], s[3*j_parent + 1]
            ]

            if 0 in c and aug_type == "JM":
                continue # 避免 joint mask 畫到與原點相連的線
            else:
                ax.plot(
                    [c[0], c[1] ],
                    [c[2], c[3] ],
                    [c[4], c[5] ],
                    color=color
                )

    def update_video(frame_num):
        """
        將一個動作的過程 plot 成 gif
        """
        fig.tight_layout()
        print('{} {}/{}'.format(filneame, frame_num+1, valid_frame_num+1), end='\r')
        pos = action[frame_num]
        ax.clear()
        plot_range = [-radius/2, radius/2]
        ax.set_xlim3d(plot_range)
        ax.set_zlim3d(plot_range)
        ax.set_ylim3d(plot_range)

        # seperate skeleton data and plot it
        s_1 = pos[:seperate_index] # 1st person
        plot_skeleton(ax, s_1, 'red')
        s_2 = pos[seperate_index:] # 2nd person
        if np.count_nonzero(s_2) != 0:
            ax.set_xlim3d([-radius, radius])
            ax.set_ylim3d([-radius, radius])
            plot_skeleton(ax, s_2, 'blue')

    anim = FuncAnimation(fig, update_video, frames=np.arange(0, valid_frame_num+1), interval=1000/fps, repeat=False)
    anim.save(filneame, dpi=80, writer='pillow')
    print()


def main():
    data_index = 8
    # 直接載入原始資料
    # x, y, = load_data()
    # print(x.shape)

    # 抽出其中一筆資料
    # action = x[data_index]
    # label  = y[data_index]
    # data_vis(action, label)

    # 模仿 Dataloader 格式 
    ntu_loaders = NTUDataLoaders('NTU', "CS", seg=20, data_volume=1)
    s_aug = "CM"
    ntu_loaders.spatial_transform_list = [s_aug]

    batch_size = 1
    train_loader = ntu_loaders.get_train_loader(batch_size, 16)

    sample_index = data_index
    k = int(np.floor(sample_index/batch_size))
    loader_data = next(itertools.islice(train_loader, k, None))

    s_1 = np.array(loader_data[0][0]) # 回傳的第一組骨架是變形過的
    print(s_1.shape)
    s_2 = np.array(loader_data[1][0]) # 回傳的第二組骨架是原本樣子
    action = np.concatenate((s_1, s_2), axis=1)
    label = np.array(loader_data[2][0]) # 最後一組數據是 label
    print(label)
    data_vis(action, label, s_aug) # 將 transform 結果可視化

if __name__ == "__main__":
    main()