# SGN_GRA 

Welcome to the SGN_GRA repository, the official implementation of our research paper. This guide will help you set up and run the SGN_GRA model.

## Quick Start Guide

Follow these steps to get started with SGN_GRA:

1. Create and start a Docker container:
```bash
docker create --gpus all -it --name GRA --rm nvcr.io/nvidia/pytorch:23.08-py3
docker start GRA
```
2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Run the main program:
```
python main.py -y configs/train.yaml -t 0
```


## Configuration Example
Here's an example of how to configure your training session in train.yaml:

``` yaml
# configs/train.yaml

# Dataset Configuration
dataset: !!str "NTU"
case: !!str "CV"            # Use 'CS' for NUCLA dataset.
Full-Data: !!bool True      # Set to True for using the full dataset for training.
data-volume: !!int 5        # Data volume options: NUCLA - 5%, 15%, 30%, 40%; NTU - 5%, 10%, 20%, 40%.
num-joint: !!int 25         # Joint count: NTU - 25, NUCLA - 20.

# Training Parameters
max-epoches: !!int 25
threshold: !!float 0.6
lr: !!float 0.001
weight-decay: !!float 0.000001
test-freq: !!int 1
batch-size: !!int 128       # Batch size: NUCLA - 32, NTU - 128.
workers: !!int 16           # Number of workers.
seg: !!int 20               # Segment count.
ckpt-name: !!str "SGN_GRA"  # Checkpoint name.
```

# Acknowledgments

We would like to express our gratitude to the following contributors for their significant contributions to this research:

```
Kuan-Hung Huang, Yao-Bang Huang, Yong-Xiang Lin, Kai-Lung Hua, Mohammad Tanveer, Xuequan Lu, Imran Razzak. "GRA: Graph Representation Alignment for Semi-Supervised Action Recognition," scheduled for publication in IEEE Transactions on Neural Networks and Learning Systems (IEEE TNNLS), 2024. (Impact Factor 2022: 10.451).
```

Thank you for exploring our project. If you have any questions or need further assistance, please feel free to reach out.