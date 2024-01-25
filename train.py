# -*- coding: utf-8 -*-

import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from load_data import load_data
from model.cnn import cnn
from model.dnn import dnn
from model.rnn import rnn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
epoch_num = 40
min_epochs = 5
min_val_loss = 5
model_type = 'cnn'  # 'dnn', 'cnn', 'rnn'


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def draw_model(model_name):
    model = None
    if model_name == 'cnn':
        train_loader, _, _ = load_data(batch_size=16)
        images, labels = next(iter(train_loader))
        model = cnn()
    if model_name == 'dnn':
        train_loader, _, _ = load_data(H=64, W=64, batch_size=16)
        images, labels = next(iter(train_loader))
        model = dnn()
    if model_name == 'rnn':
        train_loader, _, _ = load_data(H=64, W=64, batch_size=16)
        images, labels = next(iter(train_loader))
        model = rnn()
    writer = SummaryWriter()
    writer.add_graph(model, images)


def get_val_loss(model, Val):
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    val_loss = []
    for (data, target) in Val:
        data, target = Variable(data).to(device), Variable(target.long()).to(device)
        output = model(data)
        loss = criterion(output, target)
        val_loss.append(loss.cpu().item())

    return np.mean(val_loss)


def train():
    global min_val_loss
    if model_type == 'cnn':
        train_loader, val_loader, test_loader = load_data(H=256, W=256, batch_size=batch_size)
        model = cnn().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

    elif model_type == 'dnn':
        train_loader, val_loader, test_loader = load_data(H=64, W=64, batch_size=batch_size)
        model = dnn().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

    elif model_type == 'rnn':
        train_loader, val_loader, test_loader = load_data(H=64, W=64, batch_size=batch_size, enhance=True)
        model = rnn().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.00003)

    else:
        raise NotImplementedError('model_type must be "cnn" or "rnn" or "dnn"')

    criterion = nn.CrossEntropyLoss().to(device)
    writer = SummaryWriter(os.path.join('./logs', 'summary'))

    best_model = None
    last_model = None

    print('begin training ...')
    print("-----------------------------------------------------------------------------------")
    for epoch in tqdm(range(epoch_num), ascii=True):
        train_loss = []
        for batch_idx, (data, target) in enumerate(train_loader, 0):
            data, target = Variable(data).to(device), Variable(target.long()).to(device)
            optimizer.zero_grad()
            output = model(data)
            # print(output)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.cpu().item())
        # validation
        val_loss = get_val_loss(model, val_loader)
        model.train()
        if epoch + 1 > min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)
        if epoch == epoch_num-1:
            last_model = copy.deepcopy(model)
        tqdm.write('Epoch {:03d} train_loss {:.5f} val_loss {:.5f}'.format(epoch, np.mean(train_loss), val_loss))
        writer.add_scalars('loss', {"train_loss": np.mean(train_loss),
                                    "val_loss": val_loss}, epoch)
    if not os.path.exists('logs'):
        os.mkdir("./logs")
    torch.save(best_model.state_dict(), "logs/{}_best.pt".format(model_type))
    torch.save(last_model.state_dict(), "logs/{}_last.pt".format(model_type))


def test():
    if model_type == 'cnn':
        train_loader, val_loader, test_loader = load_data(H=256, W=256, batch_size=batch_size)
        model = cnn().to(device)
    elif model_type == 'dnn':
        train_loader, val_loader, test_loader = load_data(H=64, W=64, batch_size=batch_size)
        model = dnn().to(device)
    elif model_type == 'rnn':
        train_loader, val_loader, test_loader = load_data(H=64, W=64, batch_size=batch_size)
        model = rnn().to(device)
    else:
        raise NotImplementedError('model_type must be "cnn" or "rnn" or "dnn"')

    model.load_state_dict(torch.load("logs/{}_best.pt".format(model_type)), False)
    model.eval()
    total = 0
    current = 0
    for (data, target) in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        predicted = torch.max(outputs.data, 1)[1].data
        total += target.size(0)
        current += (predicted == target).sum()

    print('Accuracy:%d%%' % (100 * current / total))


if __name__ == '__main__':
    # setup_seed(20)
    # for i in range(1):
    #     train()
    #     test()
    draw_model(model_type)
