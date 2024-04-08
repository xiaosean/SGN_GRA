import argparse
import os
import fit
from model import Backbone
from trainer import Trainer
from util import make_dir, get_num_classes

import random
import numpy as np
import torch

import config



SEED = 1130
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='Skeleton-Based Action Recgnition')
fit.add_fit_args(parser)
parser.set_defaults(
    network=config.network,
    dataset = 'NTU',
    case = "CS",
    batch_size=2,
    num_epochs=config.num_epochs,
    lr=0.001,
    weight_decay=1e-6,
    workers=16,
    seg = 20,
    temperature = 0.9,
    momentum = 0.9,
    cosine_anneal = False,
    test_freq = 10,
    graph_info = config.graph_info,
    ckpt_name = config.ckpt_name,
    data_volume = config.data_volume,
    pre_train = config.pre_train,
    clf_layer_num = config.clf_layer_num,
    spatial_aug = config.s_aug,
    temporal_aug = config.t_aug
)
args = parser.parse_args()
args.num_classes = 60
backbone = Backbone(args.num_classes, args.dataset, args.seg, args)

# load_file_name = "vol_100_freeze_backbone/00_.pth"
load_file_name = "Baseline/rot-05-RM.pth"

print("Load weight: ", load_file_name)

checkpoint = torch.load("./checkpoint/"+load_file_name)
backbone.load_state_dict(checkpoint['net'])

trainer = Trainer(backbone, args)
trainer.t_sne_vis(t_sne_dim=3, file_name="rot_RM")