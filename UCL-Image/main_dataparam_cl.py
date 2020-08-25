# -*- coding: utf-8 -*-
"""An implementation of differentiable curriculum learning [1].

[1] Saxena, S., Tuzel, O., & DeCoste, D. (2019). Data parameters: A new family of parameters for learning a differentiable curriculum. In Advances in Neural Information Processing Systems (pp. 11095-11105).
"""

import numpy as np
import pdb, os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from dataset import load_data

from utils import setup_seed, img_preprocess, impose_label_noise
from model import LeNet
from utils import save_model, load_model
from utils import predict, eval_metric, eval_metric_binary
from config import opt

from utils import adjust_learning_rate

def main(**kwargs):
    # pre-setup
    setup_seed(opt.seed)
    opt.parse(kwargs)
    log_dir = os.path.join(opt.result_dir, "dataparam_" + opt.model + "_"  + opt.data_name + "_" + opt.print_opt)
    print("output log dir", log_dir)
    ckpt_dir = os.path.join(os.path.join(log_dir, "ckpt"))
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # intermediate file for early stopping
    early_stop_ckpt_path = os.path.join(ckpt_dir, "best_va.pth")
    output_result_path = os.path.join(log_dir, "dataparam.result")

    # load data & preprocess
    x_tr, y_tr, x_va, y_va, x_te, y_te = load_data(opt.data_name)
    num_class = np.unique(y_va).shape[0]

    if opt.noise_ratio > 0:
        print("put noise in label, ratio = ", opt.noise_ratio)
        y_tr = impose_label_noise(y_tr, opt.noise_ratio)
    
    x_tr, y_tr = img_preprocess(x_tr, y_tr, opt.use_gpu)
    x_va, y_va = img_preprocess(x_va, y_va, opt.use_gpu)
    x_te, y_te = img_preprocess(x_te, y_te, opt.use_gpu)
    print("load data done")

    # load model
    _, in_channel, in_size, _ = x_tr.shape
    model = LeNet(num_class=num_class, in_size=in_size, in_channel=in_channel)
    if opt.use_gpu:
        model.cuda()
    model.use_gpu = opt.use_gpu

    # compute for batch learning
    batch_size = opt.batch_size
    print("start training")

    all_tr_idx = np.arange(len(x_tr))

    va_acc = train_dataparam(model, all_tr_idx,
        x_tr, y_tr, x_va, y_va,
        opt.num_epoch,
        opt.batch_size,
        opt.lr,
        opt.weight_decay,
        early_stop_ckpt_path,
        5,)

    # evaluate model on test set
    pred_te = predict(model, x_te)
    if num_class > 2:
        acc_te = eval_metric(pred_te, y_te)
    else:
        acc_te = eval_metric_binary(pred_te, y_te)

    print("curriculum: {}, acc: {}".format(0, acc_te.item()))
    with open(output_result_path, "w") as f:
        f.write(str(acc_te.item()) + "\n")

def train_dataparam(model,
    sub_idx,
    x_tr, y_tr, 
    x_va, y_va, 
    num_epoch,
    batch_size,
    lr, 
    weight_decay,
    early_stop_ckpt_path,
    early_stop_tolerance=None,
    ):
    """Differentiable data parameters for curriculum learning.
    """
    model.train()

    # early stop setup
    best_va_acc = 0
    num_all_train = 0
    early_stop_counter = 0

    # num class & num all batches
    num_class = torch.unique(y_va).shape[0]
    num_all_tr_batch = int(np.ceil(len(sub_idx) / batch_size))

    # define data parameters & class parameters
    num_all_train = len(sub_idx)
    v_vector = 0.001 * torch.ones(num_all_train).to(x_tr.device)
    v_vector.requires_grad= True

    c_vector = torch.ones(num_class).to(x_tr.device)
    c_vector.requires_grad = True

    # init training
    optimizer = optim.Adam( [{"params": model.parameters()}, 
        {"params": v_vector, "weight_decay":0, "lr":1e-4},
        {"params": c_vector, "weight_decay":0, "lr":1e-4}],
        lr = opt.lr, weight_decay=opt.weight_decay)

    for epoch in range(num_epoch):
        model.train()
        np.random.shuffle(sub_idx)

        for idx in range(num_all_tr_batch):
            batch_idx = sub_idx[idx*batch_size:(idx+1)*batch_size]
            x_batch = x_tr[batch_idx]
            y_batch = y_tr[batch_idx]

            logits = model.forward_logits(x_batch)

            # if torch.isnan(logits).sum() > 0:
            #     pdb.set_trace()
            #     pass

            v_batch = v_vector[batch_idx]
            c_batch = c_vector[y_batch]

            sigma_batch = v_batch + c_batch

            # weighted softmax 
            # TODO weighted sigmoid?
            if num_class > 2:
                max_logit = torch.max(logits)
                logits = logits - max_logit # prevent numerical overflow
                logit_exp = torch.exp(logits / sigma_batch.unsqueeze(1))
                pred = logit_exp / logit_exp.sum(1).unsqueeze(1)
                xent_loss = F.cross_entropy(pred, y_batch,
                    reduction="none")
            else:
                print("#TODO# weighted sigmoid!!")
                raise NotImplementedError
            
            # penalty loss
            penalty_loss = torch.norm(torch.log(1e-10 + v_batch)) + torch.norm(torch.log(1e-10 + c_batch))

            # all loss
            loss_all = xent_loss.mean() + 5e-4 * penalty_loss

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

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

    return best_va_acc

if __name__ == "__main__":
    import fire
    fire.Fire()
    pass