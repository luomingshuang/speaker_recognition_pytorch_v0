import os
import random
import numpy as np

import tensorboardX
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader

import options as opt
from data_load import Mydataset
from model import DeepSpeakerModel
from audio_fbank import read_mfcc, sample_from_mfcc

import tensorflow as tf

import tqdm

if (__name__=='__main__'):
    torch.manual_seed(55)
    torch.cuda.manual_seed_all(55)

if (__name__=='__main__'):
    model = DeepSpeakerModel(embedding_size=opt.embedding_size,
                      num_classes=opt.classes).cuda()
    print(model)
    writer = SummaryWriter()

    if (hasattr(opt, 'weights')):
        pretrained_dict = torch.load(opt.weights)
        model_dict = model.state_dict()
        pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        missed_params = [k for k,v in model_dict.items() if not k in pretrained_dict.keys()]

        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict), len(model_dict)))
        print('miss matched params:{}'.format(missed_params))

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    train_dataset = Mydataset(opt.data_path, 'train')
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, drop_last=False, shuffle=True)

    test_dataset = Mydataset(opt.data_path, 'test')
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, drop_last=False, shuffle=True)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    iteration = 0
    best_acc = 0

    for epoch in range(opt.max_epochs):
        train_corr = 0
        train_all = 0
        test_corr = 0
        test_all = 0

        for i, batch in enumerate(train_loader):
            inputs, targets = batch[0].cuda(), batch[1].cuda()
            outputs = model(inputs)
            #print(targets)
            _, preds = torch.max(F.softmax(outputs, dim=1).data, 1)
            train_corr += torch.sum(preds==targets.data)
            train_all += len(inputs)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()

            iteration += 1

            loss.backward()
            optimizer.step()

            train_loss = loss.item()

            writer.add_scalar('data/train_loss', train_loss, iteration)

            iteration += 1
            print('epoch: {} iteration: {} train_loss: {}'.format(epoch, iteration, train_loss))
            #if iteration == 40:
             #   break

        train_acc = train_corr.item() / train_all
        writer.add_scalar('data/train_acc', train_acc, epoch)
        print('train_acc: ', train_acc)

        if epoch % 2 == 0:
            with torch.no_grad():
                for j, batch in enumerate(test_loader):
                    inputs, targets = batch[0].cuda(), batch[1].cuda()
                    outputs = model(inputs)
                    _, preds = torch.max(F.softmax(outputs, dim=1).data, 1)

                    test_corr += torch.sum(preds==targets.data)
                    test_all += len(inputs)
                    print('{} iteration for test acc {}'.format(j, test_corr.item()/test_all))
                    #if j == 10:
                     #   break
            
            test_acc = test_corr.item() / test_all
            print('test_acc: ', test_acc)
            writer.add_scalar('data/test_acc', test_acc, epoch)

            if test_acc >= 0.95:
                savename = os.path.join(opt.save_dir, 'olr_{}_epoch_test_acc_{}.pt'.format(epoch, test_acc))
                torch.save(model.state_dict(), savename)


