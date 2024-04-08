import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import sys
import yaml
import argparse
from datetime import datetime, timezone, timedelta

import torch
import numpy as np
import matplotlib.pyplot as plt

from util import make_dir, get_num_classes, get_dataloader

toolbar_width = 30

def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')

parser = argparse.ArgumentParser(description='Skeleton-Based Action Recgnition')
parser.add_argument("-y", "--yaml_path", type = str, help="path of config (yaml file).")
parser.add_argument("-t", "--train", type = int, help="0:train, 1:test, 2:test in joint masking, 3: test in frame masking.")
args = parser.parse_args()


with open(args.yaml_path) as f:
    config = yaml.safe_load(f)

print(f"===============================================")
print(config)
print(f"===============================================")
config['seg'] = 300
config['dataset'] = 'NUCLA'
situation = 0
train_loader, test_loader, _, _ = get_dataloader(config, situation)


# 分析 Missing frames & joints 的情形
missing_frame_samples = []
missing_joints = {
    '0':0,
    '1':0,
    '2':0,
    '3':0,
    '4':0,
    '5':0,
    '6':0,
    '7':0,
    '8':0,
    '9':0,
    '10':0,
    '11':0,
    '12':0,
    '13':0,
    '14':0,
    '15':0,
    '16':0,
    '17':0,
    '18':0,
    '19':0,
    # '20':0,
    # '21':0,
    # '22':0,
    # '23':0,
    # '24':0
}

frame_person_counter = 0

for i,(data, label) in enumerate(test_loader):
    # print(i, data.size(), label.size()) 
    B, T, V = data.size() # 64, 300, 75

    for b in range(B):
        batch_data = data[b,:,:].view(3, T, int(V/3)).cpu().detach().numpy()
        # print(batch_data.shape) # 3, 300, 25
        
        for t in range(T):
            frame_data = batch_data[:,t,:]
            frame_person_counter += 1

            if not np.any(frame_data):
                missing_frame_samples.append(i)
            else:
                for j in range(int(V/3)):
                    vertex_data = frame_data[:, j]
                    # print(vertex_data)
                    if not np.any(vertex_data):
                        missing_joints[str(j)] += 1

    print_toolbar(i * 1.0 / len(test_loader),
                      '({:>5}/{:<5}) Processing data: '.format(
                          i + 1, len(test_loader)))

    if i == 30:
        break

print()
print(f"Missing frame samples number: {round(len(missing_frame_samples)/frame_person_counter*100, 2)} %", )
print("Missing joints:")
for k, v in missing_joints.items():
    print(f'Joint {k} : {v} ( {round(v/frame_person_counter*100, 2)} % )')


x = np.arange(len(missing_joints))

scores = list(missing_joints.values())
scores = [s / frame_person_counter * 100 for s in scores]

joint_index = list(missing_joints.keys())
plt.bar(x, scores)
plt.xticks(x, joint_index)
plt.xlabel('Joint index')
plt.ylabel('Missing percent (%)')
plt.ylim(ymax = 100, ymin = 0)
plt.title(f'{config["dataset"]} Missing joints analysis')
plt.savefig(f'{config["dataset"]}_missing_joints_analysis.png')