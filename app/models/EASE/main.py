import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy import sparse
from utils import argparsing
import pandas as pd
from dataloader import DataLoader
from models import EASE
from trainers import test, inference
import time
from runners import ease_runner
import wandb

## setting
current_time = time.strftime('%y%m%d_%H%M%S')
args = argparsing()
torch.manual_seed(args.seed)

## wandb
# wandb.login()
# wandb.init(
#     project="level3-EASE", config=vars(args), entity="boostcamp6-recsys6"
# )
# wandb.run.name = f"Hyeongjin Cho" + current_time
# wandb.run.save()



## Load data
loader = DataLoader(args)
n_items, train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te, data_inf = loader.data_loading()
N = train_data.shape[0]


## Build the model
p_dims = [200, 600, n_items]
models = {'EASE':EASE(31360, n_items)}
losses = {'EASE': None}
runners = {'EASE':ease_runner}

print(f'INITIALIZING {args.model}....')
print(f'current time: {current_time}')
model = models[args.model].to(args.device)
runner = runners[args.model]
criterion = losses[args.model]

optimizer = None if args.model=='EASE' else optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)


## Train
print('\nTRAINING....')
best_n100 = -np.inf
args.update_count = 0

for epoch in range(1, args.epochs+1):
    n100 = runner(args, model, criterion, optimizer, data_inf, vad_data_tr, vad_data_te, epoch, N, data_inf)

    # Save the model if the n100 is the best we've seen so far.
    if n100 > best_n100:
        model_path = os.path.join(args.pro_dir, 'model_files')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        with open(f'{model_path}/{args.model} {current_time}.pt', 'wb') as f:
            # torch.save(model, f)
            torch.save(model, f)
        best_n100 = n100
        best_epoch = epoch

print('best epoch:', best_epoch)
print(f'best score n100:{best_n100}')

## load best model
with open(f'{args.model_path}/{args.model} {current_time}.pt', 'rb') as f:
    model = torch.load(f)

## Test
print('\nTESTING....')
test(args, model, criterion, test_data_tr, test_data_te)

## Inference
print('\nINFERING....')
inference(args, model, data_inf, current_time)

# wandb.finish()