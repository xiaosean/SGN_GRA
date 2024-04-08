import argparse
import os
from datetime import datetime, timezone, timedelta
import random

import numpy as np
import torch
import yaml
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from data import NTUDataLoaders
from model_SGN import SGN, Linear_clf
from model_CTRGCN import CTRGCN
# from model_EfficientGCN import EfficientGCN
from trainer import Trainer
from util import make_dir, get_num_classes, get_dataloader
from data import NTUDataLoaders

# Set random seed for whole training process
SEED = 27
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

def train(config, situation):
    train_loader, test_loader, train_subloader, tsne_loader = get_dataloader(config, situation)
    #####################
    # model = SGN(config)
    if config['ckpt-name'] in ['SGN_GRA', 'SGN_GRA_test', 'SGN_GRA_RM']:
        model = SGN(config)
    elif config['ckpt-name'] == 'CTRGCN_GRA':
        model = CTRGCN(config)
        # model = torch.nn.DataParallel(
        #     model.to('cuda:0'), device_ids=[0,1,2,3], output_device=0
        # )
        # model = model.module
    # elif config['ckpt-name'] == 'EfficientGCN_GRA':
    #     model = EfficientGCN(config)
    
    #####################
    trainer = Trainer(model, train_loader, test_loader, train_subloader, tsne_loader, config)
    print("Start supervised training!")
    trainer.train_semi(200)
    print("Start self-training!")
    trainer.pseudo_label_new(25, config["threshold"])
    # trainer.pseudo_label_new(50, config["threshold"])# for CTR-GCN

def test(config, situation):
    train_loader, test_loader, train_subloader, tsne_loader = get_dataloader(config, situation)
    # train_loader, test_loader, train_subloader = get_dataloader(config, situation)
    #####################
    # model = SGN(config)
    if config['ckpt-name'] in ['SGN_GRA', 'SGN_GRA_test', 'SGN_GRA_RM']:
        model = SGN(config)
    elif config['ckpt-name'] == 'CTRGCN_GRA':
        model = CTRGCN(config)       
        # model = torch.nn.DataParallel(
        #     model.to('cuda:0'), device_ids=[0,1,2,3], output_device=0
        # )
        # model = model.module
    #####################
    if config["dataset"] == "NTU":
        num_classes = 60
    elif config["dataset"] == "NUCLA":
        num_classes = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clf = Linear_clf(num_classes, model.representation_dim, 1).to(device)
    model.load_state_dict(torch.load(f'./checkpoint/{config["dataset"]}/{config["case"]}/{config["data-volume"]}/{config["ckpt-name"]}/pseudo/select.pth')["net"])
    clf.load_state_dict(torch.load(f'./checkpoint//{config["dataset"]}/{config["case"]}/{config["data-volume"]}/{config["ckpt-name"]}/pseudo/select.pth')["clf"])
    trainer = Trainer(model, train_loader, test_loader, train_subloader, tsne_loader, config, situation)
    trainer.test(trainer.test_loader, clf)


def _main(args):
    args.num_classes = get_num_classes(args.dataset)
    args.ckpt_name = args.ckpt_name + f"{args.threshold}"
    print(args.ckpt_name)
    train_loader, test_loader, train_subloader, pseudo_loader_w, pseudo_loader_s, tsne_loader = get_dataloader(args)
    print(len(train_loader))
    print(len(train_subloader))
    backbone = SGN(args)
    #print(get_n_params(backbone))
    #backbone = Backbone(args)
    #print(get_n_params(backbone))    

    #if args.pretrain == 1:
    #    print(f"Load weight: ./checkpoint/{args.ckpt_name}/semi/best.pth")
    #    checkpoint = torch.load(f"./checkpoint/{args.ckpt_name}/semi/best_229.pth")
    #    backbone.load_state_dict(checkpoint['net'])

    # spatial_aug & temporal_aug 改用字串作為輸入，才可透過 Bash script 執行多個實驗
    # 轉為 list , 進到 transfrom 那邊才可正常指定 augmentaion
    
    if args.pretrain == 1:
        trainer = Trainer(backbone, train_loader, test_loader, train_subloader, pseudo_loader_w, pseudo_loader_s, tsne_loader, args)
        #trainer.test(trainer.test_loader, clf)
        #trainer.t_sne_vis("pseudo")
        #trainer.pseudo_label_new(50)
        
        #trainer.train_semi(200)
        trainer.pseudo_label_new(25, args.threshold)
        #trainer.pseudo_label_fixmatch(args.max_epoches)
    elif args.pretrain == 0: 
        trainer = Trainer(backbone, train_loader, test_loader, train_subloader, pseudo_loader_w, pseudo_loader_s, tsne_loader, args)
        #trainer.train_SSL(args.max_epoches-100)
        #print(f"Load weight: ./checkpoint/{args.dataset}/{args.case}/{args.data_volume}/{args.ckpt_name}/SSL/best.pth")
        #checkpoint = torch.load(f"./checkpoint/{args.dataset}/{args.case}/{args.data_volume}/{args.ckpt_name}/SSL/best.pth")
        #checkpoint = torch.load(f"./checkpoint/{args.dataset}/{args.case}/{args.data_volume}/{args.ckpt_name}/SSL/best.pth")
        #backbone.load_state_dict(checkpoint['net'])
        trainer.train_semi(200)
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        clf = Linear_clf(args.num_classes, backbone.representation_dim, args.clf_layer_num).to(device)
        backbone.load_state_dict(torch.load(f"./checkpoint/{args.dataset}/{args.case}/{args.data_volume}/{args.ckpt_name}/pseudo/select.pth")["net"])
        clf.load_state_dict(torch.load(f"./checkpoint/{args.dataset}/{args.case}/{args.data_volume}/{args.ckpt_name}/pseudo/select.pth")["clf"])
        trainer = Trainer(backbone, train_loader, test_loader, train_subloader, pseudo_loader_w, pseudo_loader_s, tsne_loader, args)
        trainer.test(trainer.test_loader, clf)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Skeleton-Based Action Recgnition')
    """
    # Dataset
    parser.add_argument('--dataset', type=str, default='NUCLA',
                        help='select dataset to evlulate (NTU or NUCLA)')
    parser.add_argument('--case', type=str, default="CS",
                        help='select which case')
    parser.add_argument('--num-classes', type=int, default=60,
                        help='the number of classes')
    parser.add_argument('--data-volume', type=int, default=5,
                        help='the number of classes')

    # Training parameters
    parser.add_argument('--max-epoches', type=int, default=200,
                        help='max number of epochs to run')
    parser.add_argument('-th', '--threshold', type=float, default=0.8,
                        help='threshold of self-training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', '--wd', type=float, default=1e-6,
                        help='weight decay (default: 1e-6)')
    parser.add_argument('--test-freq', '-p', type=int, default=1,
                        help='print frequency (default: 10)')
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                        help='N-UCLA: 32, NTU: 128')
    parser.add_argument('--train', type=int, default=1,
                        help='train or test')
    parser.add_argument('--workers', type=int, default=16,
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--seg', type=int, default=20,
                         help='number of segmentation')
    parser.add_argument('--temperature', type=float, default=0.9,
                         help='temperature for contrastive learning')

    parser.add_argument("-g", "--graph-info", type=str, default=config.graph_info, 
                        help='add graph loss or not. (default = False)')
    parser.add_argument("--clf-layer-num", type=str, default=config.clf_layer_num, 
                        help='layer num of clf (default = 1)')
    parser.add_argument("-cn", "--ckpt_name", type=str, default="SGN_STRAV2_", 
                        help='save weight path.')
    parser.add_argument("-pre", "--pretrain", type=int, default=config.pre_train, 
                        help='set pretrain.')
    
    # Augmentations
    parser.add_argument("-s", "--spatial_aug", type=str, default=config.s_aug,
                        help='spatial augmentation. (example:rot-sh-GB-JM-CM) (default = None)')
    parser.add_argument("-t", "--temporal_aug", type=str, default=config.t_aug,
                        help='temporal augmentation. (example:RM) (default = None)')
    """

    parser.add_argument('-th', '--threshold', type=float, default=0.8,
                        help='threshold of self-training')
    parser.add_argument("-y", "--yaml_path", type = str, help="path of config (yaml file).")
    parser.add_argument("-t", "--train", type = int, help="0:train, 1:test, 2:test in joint masking, 3: test in frame masking.")
    args = parser.parse_args()


    with open(args.yaml_path) as f:
        config = yaml.safe_load(f)

    print(f"===============================================")
    print(config)
    print(f"===============================================")

    if args.train == 0:
        train(config, args.train)
    elif args.train == 1 or args.train == 2 or args.train == 3:
        test(config, args.train)

