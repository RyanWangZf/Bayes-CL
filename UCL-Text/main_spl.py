# -*- coding: utf-8 -*-
"""Implementation of CL based on
Self-paced learning (SPL) [1],
Self-paced Curriculum Learning (SPCL) [2],
Self-paced Learning via Implicit Regularizer (SPL-IR) [3].
[1] Kumar, M.; Packer, B.; and Koller, D. 2010. Self-paced learning for latent variable models. In NIPS.
[2] Jiang, L., Meng, D., Zhao, Q., Shan, S., & Hauptmann, A. G. (2015, February). Self-paced curriculum learning. In Twenty-Ninth AAAI Conference on Artificial Intelligence.
[3] Fan, Y., He, R., Liang, J., & Hu, B. (2017, February). Self-paced learning: an implicit regularization perspective. In Thirty-First AAAI Conference on Artificial Intelligence.
"""

import numpy as np
import pdb, os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from dataset import load_data

from model import TextCNN
from utils import train, predict, eval_metric, eval_metric_binary
from utils import setup_seed, impose_label_noise, text_preprocess
from utils import save_model, load_model
from config import opt
from dataset import load_data

def main(**kwargs):
    # pre-setup
    setup_seed(opt.seed)
    opt.parse(kwargs)
    log_dir = os.path.join(opt.result_dir, "{}_".format(opt.spl) + opt.model + "_"  + opt.data_name + "_" + opt.print_opt)
    print("output log dir", log_dir)

    if not torch.cuda.is_available():
        print("[WARNING] do not find cuda device, change use_gpu=False!")
        opt.use_gpu = False

    ckpt_dir = os.path.join(os.path.join(log_dir, "ckpt"))
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # intermediate file for early stopping
    early_stop_ckpt_path = os.path.join(ckpt_dir, "best_va.pth")

    # load data & preprocess
    # x_tr, y_tr, x_va, y_va, x_te, y_te = load_data(opt.data_name, [0, 1])
    x_tr, y_tr, x_va, y_va, x_te, y_te, vocab_size = load_data(opt.data_name, mode="onehot")
    num_class = np.unique(y_tr).shape[0]

    if opt.noise_ratio > 0:
        print("put noise in label, ratio = ", opt.noise_ratio)
        y_tr = impose_label_noise(y_tr, opt.noise_ratio)
    
    x_tr, y_tr = text_preprocess(x_tr, y_tr, opt.use_gpu, "onehot")
    x_va, y_va = text_preprocess(x_va, y_va, opt.use_gpu, "onehot")
    x_te, y_te = text_preprocess(x_te, y_te, opt.use_gpu, "onehot")
    print("load data done")

    model = TextCNN(vocab_size, num_class)
    if opt.use_gpu:
        model.cuda()
    model.use_gpu = opt.use_gpu

    # compute for batch learning
    batch_size = opt.batch_size

    te_acc_list = []
    print("start training")

    all_tr_idx = np.arange(len(x_tr))


    # start spl training
    if opt.spl == "spl":
        print("Do self-paced learning, binary.")
        va_acc_init = train_spl(model,
            all_tr_idx,
            x_tr, y_tr, 
            x_va, y_va, 
            opt.num_epoch,
            opt.batch_size,
            opt.lr, 
            opt.weight_decay,
            early_stop_ckpt_path,
            5,)
    
    elif opt.spl == "spcl":
        print("Do self-paced curriculum learning, linear.")
        va_acc_init = train_spcl(model,
            all_tr_idx,
            x_tr, y_tr, 
            x_va, y_va, 
            opt.num_epoch,
            opt.batch_size,
            opt.lr, 
            opt.weight_decay,
            early_stop_ckpt_path,
            5,
            1e-5,
            "linear")

    elif opt.spl == "splir":
        print("Do self-paced curriculum learning via implicit regularization (spl-ir))")
        va_acc_init = train_splir(model,
            all_tr_idx,
            x_tr, y_tr, 
            x_va, y_va, 
            opt.num_epoch,
            opt.batch_size,
            opt.lr, 
            opt.weight_decay,
            early_stop_ckpt_path,
            5,
            1e-1,)


    # evaluate model on test set
    pred_te = predict(model, x_te)
    if num_class > 2:
        acc_te = eval_metric(pred_te, y_te)
    else:
        acc_te = eval_metric_binary(pred_te, y_te)

    print("curriculum: {}, acc: {}".format(0, acc_te.item()))
    te_acc_list.append(acc_te.item())
    print(te_acc_list)

def train_spl(model,
    sub_idx,
    x_tr, y_tr, 
    x_va, y_va, 
    num_epoch,
    batch_size,
    lr, 
    weight_decay,
    early_stop_ckpt_path,
    early_stop_tolerance=None,
    lambda_v = 1e-5,
    step_size = 1.5,
    ):
    """Self-paced Learning [1]
    Args:
        early_stop_tolerance: set as None means no early stopping.
    
    [1] Kumar, M.; Packer, B.; and Koller, D. 2010. Self-paced learning for latent variable models. In NIPS.
    """
    # early stop setup
    best_va_acc = 0
    num_all_train = 0
    early_stop_counter = 0

    # num class
    num_class = torch.unique(y_va).shape[0]
    num_all_tr_batch = int(np.ceil(len(sub_idx) / batch_size))

    # define weight variable for all training samples
    num_all_train = len(sub_idx)
    v_vector = 0.5 * torch.ones(num_all_train).to(x_tr.device)
    v_vector.requires_grad=True

    # init training
    optimizer = optim.Adam( [{"params": model.parameters()}, 
        {"params": v_vector, "weight_decay":0,}],
        lr = opt.lr, weight_decay=opt.weight_decay,)
    
    for epoch in range(num_epoch):
        total_loss = 0
        model.train()
        np.random.shuffle(sub_idx)

        for idx in range(num_all_tr_batch):
            batch_idx = sub_idx[idx*batch_size:(idx+1)*batch_size]
            x_batch = x_tr[batch_idx]
            y_batch = y_tr[batch_idx]

            pred = model(x_batch)

            # get weight of samples
            v_batch = v_vector[batch_idx]

            if num_class > 2:
                loss = F.cross_entropy(pred, y_batch,
                    reduction="none")
            else:
                loss = F.binary_cross_entropy(pred[:,0], y_batch.float(), 
                    reduction="none")
            
            weighted_avg_loss = torch.sum(v_batch * loss)

            loss_all = weighted_avg_loss + lambda_v * torch.sum(v_batch)

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            
            total_loss = total_loss + weighted_avg_loss.detach()

        # evaluate on va set
        model.eval()
        pred_va = predict(model, x_va)
        if num_class > 2:
            acc_va = eval_metric(pred_va, y_va)
        else:
            acc_va = eval_metric_binary(pred_va, y_va)

        print("epoch: {}, acc: {}".format(epoch, acc_va.item()))

        if epoch == 0:
            best_va_acc = acc_va

        if acc_va > best_va_acc:
            best_va_acc = acc_va
            early_stop_counter = 0
            # save model
            save_model(early_stop_ckpt_path, model)

        else:
            early_stop_counter += 1

        if early_stop_tolerance is not None:
            # do early stopping
            if early_stop_counter >= early_stop_tolerance:
                print("early stop on epoch {}, val acc {}".format(epoch, best_va_acc))
                # load model from the best checkpoint
                load_model(early_stop_ckpt_path, model)
                break

        # update step size
        lambda_v = lambda_v * step_size

    return best_va_acc

def train_spcl(model,
    sub_idx,
    x_tr, y_tr, 
    x_va, y_va, 
    num_epoch,
    batch_size,
    lr, 
    weight_decay,
    early_stop_ckpt_path,
    early_stop_tolerance=None,
    lambda_v = 1e-5,
    sp_fun = "linear",
    step_size = 1.5,
    ):
    """Self-paced Curriculum Learning credits to [2]
    Args:
        early_stop_tolerance: set as None means no early stopping.
        sp_fun: self-paced function, could be "binary", "linear", "log";
    [2] Jiang, L., Meng, D., Zhao, Q., Shan, S., & Hauptmann, A. G. (2015, February). Self-paced curriculum learning. In Twenty-Ninth AAAI Conference on Artificial Intelligence.
    """
    assert sp_fun in ["linear", "binary", "log"]

    # early stop setup
    best_va_acc = 0
    num_all_train = 0
    early_stop_counter = 0

    # num class
    num_class = torch.unique(y_va).shape[0]
    num_all_tr_batch = int(np.ceil(len(sub_idx) / batch_size))

    # define weight variable for all training samples
    num_all_train = len(sub_idx)
    v_vector = 0.5 * torch.ones(num_all_train).to(x_tr.device)
    v_vector.requires_grad=True

    # init training
    optimizer = optim.Adam( [{"params": model.parameters()}, 
        {"params": v_vector, "weight_decay":0,}],
        lr = opt.lr, weight_decay=opt.weight_decay,)
    
    for epoch in range(num_epoch):
        total_loss = 0
        model.train()
        np.random.shuffle(sub_idx)

        for idx in range(num_all_tr_batch):
            batch_idx = sub_idx[idx*batch_size:(idx+1)*batch_size]
            x_batch = x_tr[batch_idx]
            y_batch = y_tr[batch_idx]

            pred = model(x_batch)

            # get weight of samples
            v_batch = v_vector[batch_idx]

            if num_class > 2:
                loss = F.cross_entropy(pred, y_batch,
                    reduction="none")
            else:
                loss = F.binary_cross_entropy(pred[:,0], y_batch.float(), 
                    reduction="none")
            
            weighted_avg_loss = torch.sum(v_batch * loss)

            # plus a self-paced function f(v; lambda_v)
            if sp_fun == "binary":
                # equivalent to SPL
                sp_term = lambda_v * torch.sum(v_batch)

            elif sp_fun == "linear":
                sp_term = 0.5 * lambda_v * torch.sum(v_batch ** 2 - 2 * v_batch)

            elif sp_fun == "log":
                gamma = 1 - lambda_v
                sp_term = torch.sum(gamma * v_batch - gamma ** v_batch / (1e-12 + np.log(gamma)))

            loss_all  = weighted_avg_loss + sp_term

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            
            total_loss = total_loss + weighted_avg_loss.detach()

        # evaluate on va set        
        model.eval()
        pred_va = predict(model, x_va)
        if num_class > 2:
            acc_va = eval_metric(pred_va, y_va)
        else:
            acc_va = eval_metric_binary(pred_va, y_va)

        print("epoch: {}, acc: {}".format(epoch, acc_va.item()))

        if epoch == 0:
            best_va_acc = acc_va

        if acc_va > best_va_acc:
            best_va_acc = acc_va
            early_stop_counter = 0
            # save model
            save_model(early_stop_ckpt_path, model)

        else:
            early_stop_counter += 1

        if early_stop_tolerance is not None:
            # do early stopping
            if early_stop_counter >= early_stop_tolerance:
                print("early stop on epoch {}, val acc {}".format(epoch, best_va_acc))
                # load model from the best checkpoint
                load_model(early_stop_ckpt_path, model)
                break

        # update step size
        lambda_v = lambda_v * step_size


    return best_va_acc

def train_splir(model,
    sub_idx,
    x_tr, y_tr, 
    x_va, y_va, 
    num_epoch,
    batch_size,
    lr, 
    weight_decay,
    early_stop_ckpt_path,
    early_stop_tolerance=None,
    lambda_v = 1e-1,
    step_size=1.2,
    ):
    """Self-paced Curriculum Learning by Implicit Regularization credits to [3]
    Args:
        early_stop_tolerance: set as None means no early stopping.
    [3] Fan, Y., He, R., Liang, J., & Hu, B. (2017, February). Self-paced learning: an implicit regularization perspective. In Thirty-First AAAI Conference on Artificial Intelligence.
    """

    # early stop setup
    best_va_acc = 0
    num_all_train = 0
    early_stop_counter = 0

    # num class
    num_class = torch.unique(y_va).shape[0]
    num_all_tr_batch = int(np.ceil(len(sub_idx) / batch_size))

    # define weight variable for all training samples
    num_all_train = len(sub_idx)
    
    optimizer = optim.Adam(model.parameters(), lr = opt.lr, weight_decay=opt.weight_decay,)

    for epoch in range(num_epoch):
        total_loss = 0
        model.train()
        np.random.shuffle(sub_idx)

        for idx in range(num_all_tr_batch):
            batch_idx = sub_idx[idx*batch_size:(idx+1)*batch_size]
            x_batch = x_tr[batch_idx]
            y_batch = y_tr[batch_idx]

            pred = model(x_batch)

            if num_class > 2:
                loss = F.cross_entropy(pred, y_batch,
                    reduction="none")
            else:
                loss = F.binary_cross_entropy(pred[:,0], y_batch.float(), 
                    reduction="none")

            # plus a self-paced function f(v; lambda_v)
            # compute l1-l2 v
            if epoch == 0 or epoch % 2 == 0:
                weighted_avg_loss = torch.mean(loss)
            else:
                v_batch = 1 / ((lambda_v + loss ** 2)**0.5)
                weighted_avg_loss = torch.sum(v_batch * loss)

            loss_all  = weighted_avg_loss

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            
            total_loss = total_loss + weighted_avg_loss.detach()

        # evaluate on va set
        model.eval()
        pred_va = predict(model, x_va)
        if num_class > 2:
            acc_va = eval_metric(pred_va, y_va)
        else:
            acc_va = eval_metric_binary(pred_va, y_va)

        print("epoch: {}, acc: {}".format(epoch, acc_va.item()))

        if epoch == 0:
            best_va_acc = acc_va
            save_model(early_stop_ckpt_path, model)

        if acc_va > best_va_acc:
            best_va_acc = acc_va
            early_stop_counter = 0
            # save model
            save_model(early_stop_ckpt_path, model)

        else:
            early_stop_counter += 1

        if early_stop_tolerance is not None:
            # do early stopping
            if early_stop_counter >= early_stop_tolerance:
                print("early stop on epoch {}, val acc {}".format(epoch, best_va_acc))
                # load model from the best checkpoint
                load_model(early_stop_ckpt_path, model)
                break

        lambda_v = lambda_v / step_size

    return best_va_acc

if __name__ == "__main__":
    import fire
    fire.Fire()
