from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import argparse
import random
SEED = 1

random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def adjust_learning_rate(optimizer, epoch, init_lr=0.1, decay=0.1, per_epoch=10):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    print("[adjust_learning_rate] optimizer's learning rate is going to be decayed")
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 1 / (1 + decay)
    return optimizer, float(param_group['lr'])


def maginal_loss(normal_ee, error_ee,device='cuda:0'):
    margin = 1 - normal_ee + error_ee
    zero_t = torch.zeros(margin.shape).to(device)
    loss = torch.max(zero_t, margin)
    return torch.mean(loss)


def train(model, train_X, epochs=500, lr=0.01, batch_size=100,lemma_dict=None,device='cuda:0'):
    """
    [update_note] train_X's dimension is changed. Now is [# of structure , # of data , length of each data point]
    """
    optimizer = torch.optim.Adam(model.parameters(), lr)
    input_length = train_X.shape[1]

    for epoch in range(1, epochs + 1):
        if epoch % 10 == 0 and epoch != 0 :
            optimizer, lr_int = \
                adjust_learning_rate(optimizer, epoch, init_lr=lr, decay=0.05, per_epoch=10)
        model.train(); batch_count = 0

        corrupted_X = train_X.clone()
        corrupted_X[0] = torch.tensor(np.random.randint(0,len(lemma_dict),size=(input_length,3)))

        for idx in range(0, input_length, batch_size):
            start_idx = idx
            end_idx = idx + batch_size
            if end_idx > input_length:
                end_idx = -1

            batch_inputs = train_X[:, start_idx:end_idx, :]
            corrupt_inputs = corrupted_X[:, start_idx:end_idx, :]

            batch_count += 1
            if batch_count % 1000 == 0 :
                print("{} epochs - {}batch is on progress. fyi, loss is {}"\
                      .format(epoch, batch_count,torch.mean(normal_ee + error_ee)))

            batch_inputs = batch_inputs.to(device)
            corrupt_inputs = corrupt_inputs.to(device)

            normal_ee = model(batch_inputs)
            error_ee = model(corrupt_inputs)

            loss = maginal_loss(normal_ee, error_ee,device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("epoch : {} | loss : {}".format(epoch, loss))

if __name__ == '__init__' :

    parser = argparse.ArgumentParser(description='hello')
    parser.add_argument('--train_X_path', type=str, required=True, help='train_X path. it should be h5py file format')
    parser.add_arguemnt('--lemma_dict_path', type=str, required=True, help='lemma_dict path. it should be pickle file format')
    args = parser.parse_args()

    with h5py.File(args.train_X_path, 'r') as f:
        train_X = f['train_X'][:]
    train_X = torch.tensor(train_X)
    lemma_dict = pd.read_pickle(args.lemma_dict_path)
    train_X = train_X[:, :(train_X.shape[1] / 50) * 50]

    params = \
        {'VOCAB_SIZE': len(lemma_dict),
         'EMBED_SIZE': 100,
         'HID_SIZE': 75,
         'BATCH_SIZE': 50,
         'DEVICE': 'cuda:0', }

    model = eelm.EELM(**params)
    model = model.cuda()

    train.train(model, train_X)