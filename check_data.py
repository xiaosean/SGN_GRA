from data import NTUDataLoaders

ntu_loaders = NTUDataLoaders(
    'NUCLA', 'CS',
    seg=20,
    data_volume=5)
)

train_subloader = ntu_loaders.get_train_subloader(32, 16)

# t = tqdm(enumerate(self.train_subloader), desc='Loss: **** ', total=len(self.train_subloader), bar_format='{desc}{bar}{r_bar}')
# for batch_idx, (x, target, x1, gt1, x2, gt2) in t:
#     semi_optimizer.zero_grad()
#     x1,  targets = x.to(self.device), target.to(self.device)