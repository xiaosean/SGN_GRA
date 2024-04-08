import yaml
import argparse
from util import make_dir, get_num_classes, get_dataloader

parser = argparse.ArgumentParser(description='Skeleton-Based Action Recgnition')
parser.add_argument("-y", "--yaml_path", type = str, help="path of config (yaml file).")
parser.add_argument("-t", "--train", type = int, help="0:train, 1:test, 2:test in joint masking, 3: test in frame masking.")
args = parser.parse_args()

args.yaml_path = "./configs/train.yaml"

with open(args.yaml_path) as f:
    config = yaml.safe_load(f)

print(f"===============================================")
print(config)
print(f"===============================================")

situation = 0

train_loader, test_loader, train_subloader, tsne_loader = get_dataloader(config, situation)