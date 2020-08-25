# -*- coding: utf-8 -*-
import numpy as np
import pdb, os
from numpy import linalg as la

import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim

from dataset import load_data
from model import TextCNN
from utils import save_model, load_model, predict, eval_metric
from config import opt
from utils import setup_seed, text_preprocess, impose_label_noise

import matplotlib
#matplotlib.use("AGG")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# sns.set_context(fontscale=1.5)
sns.despine()
sns.set_style("whitegrid")

def main(**kwargs):
    # pre-setup
    setup_seed(opt.seed)
    opt.parse(kwargs)
    log_dir = os.path.join(opt.result_dir, "{}_".format(opt.ucl) + opt.model + "_"  + opt.data_name + "_" + opt.print_opt)

    # log_dir = os.path.join(opt.result_dir, "cltl_" + opt.model + "_"  + opt.data_name + "_" + opt.print_opt)

    print("output log dir", log_dir)

    ckpt_dir = os.path.join(os.path.join(log_dir, "ckpt"))
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # intermediate file for early stopping
    early_stop_ckpt_path = os.path.join(ckpt_dir, "best_va.pth")
    base_model_path = os.path.join(log_dir, "base_model.pth")
    score_path = os.path.join(log_dir, "score_{}.npy".format(opt.bnn))
    # score_path = os.path.join(log_dir, "tl_score.npy")

    # load data & preprocess
    x_tr, y_tr, x_va, y_va, x_te, y_te, vocab_size = load_data(opt.data_name)
    num_class = np.unique(y_va).shape[0]

    if opt.noise_ratio > 0:
        print("put noise in label, ratio = ", opt.noise_ratio)
        y_tr = impose_label_noise(y_tr, opt.noise_ratio)

    x_tr, y_tr = text_preprocess(x_tr, y_tr, opt.use_gpu)
    x_va, y_va = text_preprocess(x_va, y_va, opt.use_gpu)
    x_te, y_te = text_preprocess(x_te, y_te, opt.use_gpu)
    print("load data done")

    model = TextCNN(vocab_size, num_class)
    if opt.use_gpu:
        model.cuda()
    model.use_gpu = opt.use_gpu
    
    load_model(early_stop_ckpt_path, model)
    print("load pretrained CL model.")

    # rank uncertainty based on mode
    score_ucl  = np.load(score_path)
    arg_indices = np.argsort(score_ucl)
    batch_size = 64
    all_idx = np.arange(len(x_tr))
    num_all_tr_batch = int(np.ceil(len(all_idx) / batch_size))
    model.eval()

    score_list, acc_list = [], []
    for idx in range(num_all_tr_batch):
        print(idx)
        # from low uncertainty to high uncertainty
        batch_idx = arg_indices[idx*batch_size:(idx+1)*batch_size]
        x_batch = x_tr[batch_idx]
        y_batch = y_tr[batch_idx]
        pred_ = predict(model, x_batch)
        acc_ = eval_metric(pred_, y_batch)
        acc_list.append(acc_)
        score_list.append(score_ucl[batch_idx].mean())

    score_ar = np.array(score_list)
    acc_ar = np.array(acc_list)
    plt.scatter(score_ar, acc_ar)

    # filter
    # score_ar_pos = score_ar[score_ar>0]
    # acc_ar_pos = acc_ar[score_ar>0]
    # plt.scatter(score_ar_pos, acc_ar_pos)

    score_ar = np.array(score_list)
    acc_ar = np.array(acc_list)
    df = pd.DataFrame({opt.bnn:score_ar, "acc":acc_ar,})

    sns.regplot(x=df[opt.bnn], y=df["acc"], scatter_kws={"color":"darkblue", "alpha":0.3, "s":20}, 
        line_kws={"color":"r","alpha":0.7,"lw":5})

    # plt.xlabel(opt.bnn + " uncertainty")
    # plt.ylabel("Average ACC")
    plt.ylim([0.5, 1.0])
    plt.xticks(fontsize=18, rotation=40)
    plt.yticks(fontsize=18)
    plt.xlabel(opt.bnn, fontsize=24)
    plt.ylabel("acc", fontsize=24)

    plt.tight_layout()
    # plt.savefig("{}_score2acc_{}.pdf".format(opt.ucl, opt.bnn))
    plt.savefig("{}_score2acc_{}.png".format(opt.ucl, opt.bnn))

    plt.show()

if __name__ == "__main__":
    import fire
    fire.Fire()


