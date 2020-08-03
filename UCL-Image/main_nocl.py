# -*- coding: utf-8 -*-
# To achieve a no-curriculum training baseline.

import numpy as np
import pdb, os
import torch
import torch.nn.functional as F
import torch.optim as optim
# print("load matplotlib")
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt

# plt.style.use(["science", "ieee"])

from model import SimpleCNN
from dataset import load_data

from utils import setup_seed, img_preprocess, impose_label_noise
from config import opt

def train(**kwargs):
    # pre-setup
    setup_seed(opt.seed)
    opt.parse(kwargs)
    log_dir = os.path.join(opt.result_dir, "nocl_" + opt.model + "_"  + opt.data_name + "_" + opt.print_opt)
    ckpt_dir = os.path.join(os.path.join(log_dir, "ckpt"))
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # load data & preprocess
    x_tr, y_tr, x_va, y_va, x_te, y_te = load_data(opt.data_name)
    
    if opt.noise_ratio > 0:
        print("put noise in label, ratio = ", opt.noise_ratio)
        y_tr = impose_label_noise(y_tr, opt.noise_ratio)
            
    x_tr, y_tr = img_preprocess(x_tr, y_tr, opt.use_gpu)
    x_va, y_va = img_preprocess(x_va, y_va, opt.use_gpu)
    x_te, y_te = img_preprocess(x_te, y_te, opt.use_gpu)
    print("load data done")

    # load model
    model = SimpleCNN(10)
    if opt.use_gpu:
        model.cuda()

    # compute for batch learning
    batch_size = opt.batch_size
    num_all_batch = int(np.ceil(len(x_tr)/batch_size))
    all_tr_idx = np.arange(len(x_tr))

    # initialize optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, weight_decay=opt.weight_decay)

    te_acc_list = []

    print("start training")
    for epoch in range(opt.num_epoch):
        total_loss = 0
        np.random.shuffle(all_tr_idx)

        model.train()
        for idx in range(num_all_batch):
            batch_idx = all_tr_idx[idx*batch_size:(idx+1)*batch_size]
            x_batch = x_tr[batch_idx]
            y_batch = y_tr[batch_idx]

            pred = model(x_batch)

            loss = F.cross_entropy(pred, y_batch, reduction="sum")
            avg_loss = loss / len(x_batch)

            optimizer.zero_grad()
            avg_loss.backward()
            optimizer.step()

            total_loss = total_loss + loss

        print("epoch: {}, loss: {}".format(epoch, total_loss.item()))

        # evaluate model on test set to get acc, start from the first epoch
        pred_te = predict(model, x_te)
        acc_te = eval_metric(pred_te, y_te)
        print("epoch: {}, acc: {}".format(epoch, acc_te.item()))
        te_acc_list.append(acc_te.data.cpu().numpy())

        # plot the figure
        plt.figure(1)
        plt.plot(te_acc_list)
        plt.ylabel("acc")
        plt.xlabel("epoch")
        plt.savefig(os.path.join(log_dir, "epoch_acc_te.png"))
        plt.close(1)
        
        np.save(os.path.join(log_dir, "te_acc.npy"), te_acc_list)

        # save ckpt
        ckpt_path = os.path.join(ckpt_dir, "model_{}.pth".format(epoch+1))
        torch.save(model.state_dict(), ckpt_path)
        

def predict(model, x):
    model.eval()
    batch_size = 1000
    num_all_batch = np.ceil(len(x)/batch_size).astype(int)
    pred = []
    for i in range(num_all_batch):
        with torch.no_grad():
            pred_ = model(x[i*batch_size:(i+1)*batch_size])
            pred.append(pred_)

    pred_all = torch.cat(pred) # ?, num_class
    return pred_all

def eval_metric(pred, y):
    pred_argmax = torch.max(pred, 1)[1]
    acc = torch.sum((pred_argmax == y).float()) / len(y)
    return acc

if __name__ == '__main__':
    import fire
    fire.Fire()






