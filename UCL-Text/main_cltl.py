# -*- coding: utf-8 -*-
"""Implementation of CL based on transfer learning [1, 2].
[1] Weinshall, D., Cohen, G., & Amir, D. (2018). Curriculum learning by transfer learning: Theory and experiments with deep networks. arXiv preprint arXiv:1802.03796.
[2] Hacohen, G., & Weinshall, D. (2019). On the power of curriculum learning in training deep networks. arXiv preprint arXiv:1904.03626.
"""

import numpy as np
import pdb, os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from model import TextCNN, BertMLP
from utils import train, predict, eval_metric, eval_metric_binary
from utils import setup_seed, impose_label_noise, text_preprocess
from utils import save_model, load_model
from config import opt
from dataset import load_data

def main(**kwargs):
    # pre-setup
    setup_seed(opt.seed)
    opt.parse(kwargs)
    log_dir = os.path.join(opt.result_dir, "cltl_" + opt.model + "_"  + opt.data_name + "_" + opt.print_opt)
    print("output log dir", log_dir)

    if not torch.cuda.is_available():
        print("[WARNING] do not find cuda device, change use_gpu=False!")
        opt.use_gpu = False

    ckpt_dir = os.path.join(os.path.join(log_dir, "ckpt"))
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # intermediate file for early stopping
    early_stop_ckpt_path = os.path.join(ckpt_dir, "best_va.pth")
    bert_ckpt_path = os.path.join(ckpt_dir, "bertmlp.pth")

    # load data & preprocess
    x_tr, y_tr, x_va, y_va, x_te, y_te, vocab_size = load_data(opt.data_name, mode="onehot")
    x_tr_b, _, x_va_b, _, x_te_b, _, vocab_size = load_data(opt.data_name, mode="bert")
    
    all_tr_idx = np.arange(len(x_tr))
    num_class = np.unique(y_tr).shape[0]

    if opt.noise_ratio > 0:
        print("put noise in label, ratio = ", opt.noise_ratio)
        y_tr = impose_label_noise(y_tr, opt.noise_ratio)
    
    x_tr, y_tr = text_preprocess(x_tr, y_tr, opt.use_gpu)
    x_va, y_va = text_preprocess(x_va, y_va, opt.use_gpu)
    x_te, y_te = text_preprocess(x_te, y_te, opt.use_gpu)
    x_tr_b = text_preprocess(x_tr_b, None, opt.use_gpu, "bert")
    x_va_b = text_preprocess(x_va_b, None, opt.use_gpu, "bert")
    x_te_b = text_preprocess(x_te_b, None, opt.use_gpu, "bert")
    print("load data done")

    bert_mlp = BertMLP(num_class, x_tr_b.shape[1])
    if opt.use_gpu:
        bert_mlp.cuda()

    # finetune the bertmlp on this dataset
    if os.path.exists(bert_ckpt_path):
        print("load from finetuned resnet50 model.")
        load_model(bert_ckpt_path, bert_mlp)
    else:
        all_tr_idx = np.arange(len(x_tr))
        _ = train(bert_mlp, all_tr_idx, x_tr_b, y_tr, x_va_b, y_va,
            50, opt.batch_size, opt.lr, opt.weight_decay, bert_ckpt_path, 5)

    if os.path.exists(os.path.join(log_dir, "tl_score.npy")):
        print("load precomputed difficulty scores.")
        score_list = np.load(os.path.join(log_dir, "tl_score.npy"))
    else:
        if num_class > 2:
            score_list = compute_tl_score(bert_mlp, x_tr_b, y_tr)
        else:
            score_list = compute_tl_score_binary(bert_mlp, x_tr_b, y_tr)

        np.save(os.path.join(log_dir,"tl_score.npy"), score_list)

    model = TextCNN(vocab_size, num_class)
    if opt.use_gpu:
        model.cuda()
    model.use_gpu = opt.use_gpu

    # compute for batch learning
    batch_size = opt.batch_size

    te_acc_list = []
    print("start training")

    all_tr_idx = np.arange(len(x_tr))

    # design difficulty by uncertainty difficulty
    # transfer learning score are smaller 
    curriculum_idx_list = one_step_pacing(y_tr, -score_list, num_class, 0.2)

    te_acc_list = []

    # training on simple set
    va_acc = train(model,
            curriculum_idx_list[0],
            x_tr, y_tr,
            x_va, y_va,
            20,
            opt.batch_size,
            opt.lr,
            opt.weight_decay,
            early_stop_ckpt_path,
            5)
    
    # training on all set
    va_acc = train(model,
        all_tr_idx,
        x_tr, y_tr,
        x_va, y_va, 
        50,
        opt.batch_size,
        opt.lr*0.1,
        opt.weight_decay,
        early_stop_ckpt_path,
        5)

    pred_te = predict(model, x_te)
    if num_class > 2:
        acc_te = eval_metric(pred_te, y_te)
    else:
        acc_te = eval_metric_binary(pred_te, y_te)
    print("curriculum: {}, acc: {}".format("one-step pacing", acc_te.item()))
    te_acc_list.append(acc_te.item())
    print(te_acc_list)

def compute_tl_score(large_net, x_tr, y_tr):
    """Receive a large pretrained net, e.g., resnet50, get its penultimate features,
    for training a classifier, e.g., rbf-svm, to get the margin as the difficulty scores.
    """
    print("start computing transfer learning difficulty scores.")
    from sklearn import svm
    from sklearn.preprocessing import MinMaxScaler

    clf = svm.LinearSVC(verbose=True, max_iter=100)
    
    # processing the x_tr to features with dim 2048, for training the rbf-svm.
    large_net.eval()

    batch_size = 256
    all_tr_idx = np.arange(len(x_tr))
    num_all_tr_batch = int(np.ceil(len(all_tr_idx) / batch_size))

    x_tr_feat = []
    for idx in range(num_all_tr_batch):
        batch_idx = all_tr_idx[idx*batch_size:(idx+1)*batch_size]
        x_batch = x_tr[batch_idx]
        y_batch = y_tr[batch_idx]

        with torch.no_grad():
            x_ft = large_net.encoder(x_batch)

        x_tr_feat.append(x_ft)

    # fit a svm-rbf to get the scores
    x_tr_feat = torch.cat(x_tr_feat).cpu().numpy()
    y_tr_cpu = y_tr.cpu().numpy()

    # scaler = MinMaxScaler()
    # x_tr_feat = scaler.fit_transform(x_tr_feat)

    # preprocessing
    clf.fit(x_tr_feat, y_tr_cpu)

    score_list = []
    for idx in range(num_all_tr_batch):
        batch_idx = all_tr_idx[idx*batch_size:(idx+1)*batch_size]
        x_batch = x_tr_feat[batch_idx]
        y_batch = y_tr_cpu[batch_idx]
        x_score = clf.decision_function(x_batch)
        indices = (np.arange(len(y_batch)), y_batch)
        score_list.append(x_score[indices])

    score_list = np.concatenate(score_list, 0)
    return score_list

def compute_tl_score_binary(large_net, x_tr, y_tr):
    """Receive a large pretrained net, e.g., resnet50, get its penultimate features,
    for training a classifier, e.g., rbf-svm, to get the margin as the difficulty scores.
    """
    print("start computing transfer learning difficulty scores.")
    from sklearn import svm
    from sklearn.preprocessing import MinMaxScaler

    clf = svm.LinearSVC(verbose=True, max_iter=100)
    
    # processing the x_tr to features with dim 2048, for training the rbf-svm.
    large_net.eval()

    batch_size = 256
    all_tr_idx = np.arange(len(x_tr))
    num_all_tr_batch = int(np.ceil(len(all_tr_idx) / batch_size))

    x_tr_feat = []
    for idx in range(num_all_tr_batch):
        batch_idx = all_tr_idx[idx*batch_size:(idx+1)*batch_size]
        x_batch = x_tr[batch_idx]
        y_batch = y_tr[batch_idx]

        with torch.no_grad():
            x_ft = large_net.encoder(x_batch)

        x_tr_feat.append(x_ft)

    # fit a svm-rbf to get the scores
    x_tr_feat = torch.cat(x_tr_feat).cpu().numpy()
    y_tr_cpu = y_tr.cpu().numpy()

    # scaler = MinMaxScaler()
    # x_tr_feat = scaler.fit_transform(x_tr_feat)

    # preprocessing
    clf.fit(x_tr_feat, y_tr_cpu)

    score_list = []
    for idx in range(num_all_tr_batch):
        batch_idx = all_tr_idx[idx*batch_size:(idx+1)*batch_size]
        x_batch = x_tr_feat[batch_idx]
        y_batch = y_tr_cpu[batch_idx]
        x_score = clf.decision_function(x_batch)
        # indices = (np.arange(len(y_batch)), y_batch)
        score_list.append(x_score)

    score_list = np.concatenate(score_list, 0)
    return score_list


def one_step_pacing(y, score, num_class, ratio=0.2):
    """Execute one-step pacing based on difficulty score,
    Keep the sampled subset balanced in class ratio.
    Args:
        ratio: ratio of the first step samples, default 20%.
    """
    assert ratio <= 1 and ratio > 0
    all_class = np.arange(num_class)
    all_tr_idx = np.arange(len(y))
    curriculum_list = []
    curriculum_size = int(ratio * len(y))
    curriculum_size_c = int(curriculum_size / num_class)
    rest_curriculum_size = curriculum_size - (num_class - 1) * curriculum_size_c

    y_cpu = y.cpu()
    sub_idx = []
    for c in all_class:
        all_tr_idx_c = all_tr_idx[y_cpu == c]
        score_c = score[all_tr_idx_c]

        if c != num_class - 1:
            sub_idx_c = all_tr_idx_c[np.argsort(score_c)][:curriculum_size_c]
        else:
            sub_idx_c = all_tr_idx_c[np.argsort(score_c)][:rest_curriculum_size]

        sub_idx.extend(sub_idx_c.tolist())
    
    curriculum_list.append(sub_idx)
    remain_idx = np.setdiff1d(all_tr_idx, sub_idx)
    curriculum_list.append(remain_idx)
    return curriculum_list

if __name__ == "__main__":
    import fire
    fire.Fire()