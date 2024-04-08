import os
import csv
import os.path as osp
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from data import NTUDataLoaders

def get_n_params(model):
    """
    # Get the number of model parameter.
    # Input:
    #   model:              Pytorch model.
    # Output:
    #   pp:                 Parameter of model.
    """
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def get_dataloader(config, situation):
    """
    # Prepare dataloader for training or testing.
    # Input:
    #   config:             The setting of training or testing. It's in yaml file.
    #   situation:          The situation for testing set.       
    #
    # Output:
    #   train_loader:       The dataloader of whole training set.
    #   test_loader:        The dataloader of testing set.
    #   train_subloader:    The dataloader of Training subset. (NTU60: 5%, 10%, 20%, 40%) (NUCLA: 5%, 15%, 30%, 40%) 
    #   tsne_loader:        The dataloader for T-SNE visualization.
    """

    if config["dataset"] == "NTU" or "NUCLA":
        # Get NTU RGB-D 60 dataset
        ntu_loaders = NTUDataLoaders(
            config["dataset"], config["case"],
            seg=config["seg"],
            data_volume=int(config["data-volume"])
        )

        ntu_loaders.get_tsne_subset()
        train_loader = ntu_loaders.get_train_loader(config["batch-size"], config["workers"])
        test_loader = ntu_loaders.get_test_loader(config["batch-size"], config["workers"], situation)
        train_subloader = ntu_loaders.get_train_subloader(config["batch-size"], config["workers"])
        tsne_loader = ntu_loaders.get_tsne_loader(config["batch-size"], config["workers"])

        train_size = ntu_loaders.get_train_size()
        train_subsize = ntu_loaders.get_train_subsize()
        test_size = ntu_loaders.get_test_size()

        log_dir = f'./log/{config["dataset"]}/{config["case"]}/{config["data-volume"]}/'
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        current_time = datetime.now(timezone(timedelta(hours=+8))).strftime("%Y-%m-%d_%H:%M:%S")
        file_name = log_dir + "settings.txt"
        with open(file_name, "a+") as fp:
            fp.write(f'File path : {file_name}\n')
            fp.write(f'Time stamp : {current_time}\n')
            fp.write(f'{config["dataset"]} setting detail:\n')
            fp.write(f'Case: {config["case"]}%\n')
            fp.write(f'Data Volume: {config["data-volume"]}%\n')
            fp.write(f'Train size:{train_size}\n')
            fp.write(f'Train {config["data-volume"]}% size:{train_subsize}\n')
            fp.write(f'Test size:{test_size}\n')
            fp.write(f'-----------------------------------\n')
            fp.close()

        print(f'{config["dataset"]} setting detail:\n')
        print(f'Case: {config["case"]}')
        print(f'Data Volume: {config["data-volume"]}')
        print(f'Train size:{train_size}')
        print(f'Train {config["data-volume"]}% size:{train_subsize}')
        print(f'Test size:{test_size}')
        print(f'===============================================')
    else:
        print(f'GRA does not provide {config["dataset"]}!')

    return train_loader, test_loader, train_subloader, tsne_loader 

def make_dir(dataset):
    if dataset == 'NTU':
        output_dir = os.path.join('./results/NTU/')
    elif dataset == 'NTU120':
        output_dir = os.path.join('./results/NTU120/')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return output_dir

def get_num_classes(dataset):
    if dataset == 'NTU':
        return 60
    elif dataset == 'NTU120':
        return 120
    elif dataset == "NUCLA":
        return 10


def write_training_log(config, training_loss, training_acc, testing_loss, testing_acc, best_epoch, e=0, psize=0, train_type=None):
    """
    自動寫 log 記下每次產生 best accuracy 時的實驗數據
    """
    log_dir = f'./log/{config["dataset"]}/{config["case"]}/{config["data-volume"]}/{config["ckpt-name"]}/{train_type}/'
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.now(timezone(timedelta(hours=+8))).strftime("%Y-%m-%d_%H:%M:%S")
    file_name = log_dir + config["ckpt-name"].replace(".pth", ".txt")
    fp = open(file_name, "a+")
    fp.write(f"File path : {file_name}\n")
    fp.write(f"Time stamp : {current_time}\n")
    fp.write("| Train loss | Train  acc | Test  acc | Test  acc | epoch | P-size |\n")
    fp.write("| {} | {}% | {} | {}% | {}-{} epochs | {} |\n".format(
        str(round(training_loss, 3)),
        str(round(training_acc, 3)),
        str(round(testing_loss, 3)),
        str(round(testing_acc, 3)),
        str(best_epoch+1),
        str(e+1),
        str(psize)+"%"
    ))
    fp.write(f'Training epochs : {config["max-epoches"]}\n')
    fp.write(f"-----------------------------------\n")
    fp.close()

def t_sne_vis(backbone, tsne_loader, device, epoch, train_type, save_path=f"./fig/", t_sne_dim=2):

    print(f"Saving figs to {save_path}")
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)

    start = time.time()
    backbone.eval()
    total = len(tsne_loader)
    r_container = []
    label_container = []
    # 先拿到整個測試集的 representation
    for batch_idx, (x1, targets) in enumerate(tsne_loader):
        print("Collecting representation : ", round((batch_idx/total)*100, 1), " %", end="\r")
        x1, targets = x1.to(device), targets.to(device)
        representation, _ = backbone(x1)
        representation = representation.view(representation.shape[0], representation.shape[1]).contiguous()
        r_container, label_container = collect_data_for_TSNE(representation, targets, r_container, label_container)
        # if batch_idx == 600:
        #     break
    print()
    total = len(label_container)
    # 做降維後畫圖
    r_reduction = TSNE(n_components=t_sne_dim, perplexity=50).fit_transform(r_container)
    c_map = [round((x/255), 4) for x in range(120, 256, 15)]
    if t_sne_dim == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title(f"2D {train_type} {epoch}")
    else: # 3D rotated gif
        from matplotlib.animation import FuncAnimation, writers
        from mpl_toolkits.mplot3d import Axes3D
        plt.ioff()
        fig = plt.figure()
        fig.suptitle(f"3D {file_name}", fontsize=16)
        ax = fig.gca(projection='3d')  # 3D投影模式
        ax.view_init(elev=50, azim=-60)  # 設定 3D 視角
    for ind in range(total):
        print("Ploting distribution : ", round((ind/total)*100, 1), " %", end="\r")#, end="\r"
        label = label_container[ind]
        # print("label:", label)
        if label >= 50:
            color = (c_map[label-50], 0, 0)
        elif label >= 40:
            color = (0, c_map[label-40], 0)
        elif label >= 30:
            color = (0, 0, c_map[label-30])
        elif label >= 20:
            color = (c_map[label-20], c_map[label-20], 0)
        elif label >= 10:
            color = (0, c_map[label-10], c_map[label-10])
        else:
            color = (c_map[label], 0, c_map[label])
        coordinates = r_reduction[ind]
        if t_sne_dim == 2:
            ax.scatter(coordinates[0], coordinates[1], color=color, s=0.9)
        else:
            ax.scatter(coordinates[0], coordinates[1], coordinates[2], color=color, s=0.9)
    # 存檔
    if t_sne_dim == 2:
        plt.savefig(f"{save_path}2D_{train_type}_{epoch}.png")
        plt.close('all')
    else: # 3D rotated gif
        def update_gif(a):
            """
            將旋轉過程 plot 成 gif
            """
            fig.tight_layout()
            print('{} {}/{}'.format(file_name, a+1, 36), end='\r')
            ax.view_init(elev=50, azim= -60 + (a*10))
        anim = FuncAnimation(fig, update_gif, frames=np.arange(0, 36), interval=1000/6, repeat=False) # interval 1000/FPS
        anim.save(f"{save_path}3D_{train_type}_{epoch}.gif", dpi=80, writer='pillow')
    end = time.time()
    print(f"Plot time : {end-start} seconds")

def collect_data_for_TSNE(representation, targets, r_container, label_container):
    np_r = representation.detach().cpu().numpy()
    np_l = targets.detach().cpu().numpy()
    for ind in range(len(representation)):
        r = np_r[ind]
        label = np_l[ind]
        r_container.append(r)
        label_container.append(label)
    return r_container, label_container
