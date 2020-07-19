# -*- coding: utf-8 -*-
import numpy as np
import pdb, os
from numpy import linalg as la

import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim

import math

from dataset import load_data

from utils import setup_seed, img_preprocess, impose_label_noise
from model import BNN
from utils import save_model, load_model
from config import opt

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

def eval_metric_binary(pred, y):
    pred_label = np.ones(len(pred))
    y_label = y.detach().cpu().numpy()
    pred_prob = pred.flatten().cpu().detach().numpy()
    pred_label[pred_prob < 0.5] = 0.0
    acc = torch.Tensor(y_label == pred_label).float().mean()
    return acc

def train(model,
    sub_idx,
    x_tr, y_tr, 
    x_va, y_va, 
    num_epoch,
    batch_size,
    lr, 
    weight_decay,
    early_stop_ckpt_path,
    early_stop_tolerance=3,
    ):
    """Given selected subset, train the model until converge.
    """
    # early stop
    best_va_acc = 0
    num_all_train = 0
    early_stop_counter = 0

    # init training
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, weight_decay=opt.weight_decay)
    num_all_tr_batch = int(np.ceil(len(sub_idx) / batch_size))

    # num class
    num_class = torch.unique(y_va).shape[0]

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

            sum_loss = torch.sum(loss)
            avg_loss = torch.mean(loss)

            num_all_train += len(x_batch)
            optimizer.zero_grad()
            avg_loss.backward()
            optimizer.step()

            total_loss = total_loss + sum_loss.detach()
        
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

        if early_stop_counter >= early_stop_tolerance:
            print("early stop on epoch {}, val acc {}".format(epoch, best_va_acc))
            # load model from the best checkpoint
            load_model(early_stop_ckpt_path, model)
            break

    return best_va_acc

def main(**kwargs):
    # pre-setup
    setup_seed(opt.seed)
    opt.parse(kwargs)
    log_dir = os.path.join(opt.result_dir, "bcl_" + opt.model + "_"  + opt.data_name + "_" + opt.print_opt)
    ckpt_dir = os.path.join(os.path.join(log_dir, "ckpt"))
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # intermediate file for early stopping
    early_stop_ckpt_path = os.path.join(ckpt_dir, "best_va.pth")

    # load data & preprocess
    # x_tr, y_tr, x_va, y_va, x_te, y_te = load_data(opt.data_name, [0, 1])
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
    model = BNN(num_class=num_class)
    if opt.use_gpu:
        model.cuda()
    model.use_gpu = opt.use_gpu

    # compute for batch learning
    batch_size = opt.batch_size

    te_acc_list = []
    print("start training")

    all_tr_idx = np.arange(len(x_tr))

    # first train on the full set
    va_acc_init = train(model,
        all_tr_idx,
        x_tr, y_tr, 
        x_va, y_va, 
        opt.num_epoch,
        opt.batch_size,
        opt.lr, 
        opt.weight_decay,
        early_stop_ckpt_path,
        3,)

    # evaluate model on test set
    pred_te = predict(model, x_te)
    if num_class > 2:
        acc_te = eval_metric(pred_te, y_te)
    else:
        acc_te = eval_metric_binary(pred_te, y_te)

    print("curriculum: {}, acc: {}".format(0, acc_te.item()))
    te_acc_list.append(acc_te.item())

    # try to do uncertainty inference
    # first let's calculate Fisher Information Matrix
    if num_class > 2:
        emp_fmat = compute_emp_fisher_multi(model, x_tr, y_tr, num_class, 100)
    else:
        emp_fmat = compute_emp_fisher_binary(model, x_tr, y_tr, 100)

    # compute a PSD precision matrix based on empirical fisher information matrix
    prec_mat = compute_precision_mat(emp_fmat, len(x_tr))
    model.set_bayesian_precision(prec_mat)

    # do Bayesian inference, compute uncertainty with uncertainty formula
    pred, epis, alea = model.bayes_infer(x_tr[:10], T=5) # T, ?, c; ?,c; ?,c

    # design difficulty by uncertainty difficulty
    # TODO
    pdb.set_trace()
    pass

def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    def isPD(B):
        """Returns true when input is positive-definite, via Cholesky"""
        try:
            _ = la.cholesky(B)
            return True
        except la.LinAlgError:
            return False

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    I = np.eye(A.shape[0])

    max_iteration = 100
    k = 1
    for i in range(max_iteration):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
        # check converge condition
        is_pd = isPD(A3)
        if is_pd:
            break

    return A3

def compute_precision_mat(emp_fmat, num_all_sample):
    """Compute Multivariate Gaussian covariance (precision matrix) by empirical Fisher Information Matrix.
    Note that the output precision matrix must be PSD.
    """

    def isPD(B):
        """Returns true when input is positive-definite, via Cholesky"""
        try:
            _ = la.cholesky(B)
            return True
        except la.LinAlgError:
            return False

    # check if emp_fmat is PD
    emp_fmat_ar = emp_fmat.cpu().numpy()
    is_pd = isPD(emp_fmat_ar)
    if not is_pd:
        # transform emp_fmat_ar to PD
        emp_fmat_pd = nearestPD(emp_fmat_ar)
        emp_fmat = torch.from_numpy(emp_fmat_pd).to(emp_fmat.device)
        
    prec_mat = num_all_sample**2 * emp_fmat
    return prec_mat

def compute_emp_fisher_binary(model, x_tr, y_tr, batch_size=100):
    model.eval()
    all_tr_idx = np.arange(len(x_tr))
    np.random.shuffle(all_tr_idx)

    num_all_batch = int(np.ceil(len(x_tr)/batch_size))

    pmat = None
    for idx in range(num_all_batch):
        x_batch = x_tr[batch_size*idx: batch_size*(idx+1)]
        y_batch = y_tr[batch_size*idx: batch_size*(idx+1)]

        with torch.no_grad():
            pred, x_h = model(x_batch, hidden=True)

        # compute grad w
        grad_w = torch.unsqueeze(1/ len(pred) * x_h.T @ (pred[:,0] - y_batch.float()), 1)
        pmat_ = grad_w  @  grad_w.T
        if pmat is None:
            pmat = pmat_
        else:
            pmat = pmat + pmat_

    pmat = 1 / num_all_batch * pmat

    return pmat

def compute_emp_fisher_multi(model, x_tr, y_tr, num_class, batch_size=100):
    def one_hot_transform(y, num_class=10):
        one_hot_y = F.one_hot(y, num_classes=num_class)
        return one_hot_y.float()
    model.eval()

    all_tr_idx = np.arange(len(x_tr))
    np.random.shuffle(all_tr_idx)

    num_all_batch = int(np.ceil(len(x_tr)/batch_size))

    pmat = None
    for idx in range(num_all_batch):
        x_batch = x_tr[batch_size*idx: batch_size*(idx+1)]
        y_batch = y_tr[batch_size*idx: batch_size*(idx+1)]

        y_oh_batch = one_hot_transform(y_batch, num_class)

        with torch.no_grad():
            pred, x_h = model(x_batch, hidden=True)
        
        diff_y = pred - y_oh_batch
        grad_w = 1/len(pred) * (x_h.T @ diff_y)

        # to vector
        grad_w = grad_w.flatten().unsqueeze(1) # c*d, 1

        pmat_ = grad_w  @  grad_w.T

        if pmat is None:
            pmat = pmat_
        else:
            pmat = pmat + pmat_

    pmat = 1 / num_all_batch * pmat

    return pmat


if __name__ == "__main__":
    import fire
    fire.Fire()

# if __name__ == '__main__':
#     from dataset import load_data
#     from utils import img_preprocess
#     from torch.autograd import grad

#     x_tr, y_tr, x_va, y_va, x_te, y_te = load_data("cifar10", [0, 1])

#     # image processing
#     x_tr, y_tr = img_preprocess(x_tr, y_tr, False)


#     # compute Fisher Information Matrix (FIM) of the dense layer
#     batch_idx = np.random.choice(np.arange(len(y_tr)), 10)
#     batch_x = x_tr[batch_idx]
#     batch_y = y_tr[batch_idx]

#     pred, h_res = model.forward(batch_x, hidden=True)
#     # loss = F.binary_cross_entropy(pred[:,0], batch_y.float())

#     # grad_w = grad(loss, [model.dense.W_mu], create_graph=True)[0]

#     grad_cal = torch.unsqueeze(1/ len(pred) * h_res.T @ (pred[:,0] - batch_y.float()), 1)
#     pmat = grad_cal @ grad_cal.T
#     model.set_bayesian_precision(pmat + 1e-8 * torch.eye(len(pmat)))
#     res_list = []
#     for _ in range(100):
#         res = model.forward(batch_x, sample=True)
#         res_list.append(res.flatten().detach().numpy().tolist())
#     res_ar = np.array(res_list)
#     pdb.set_trace()
    
#     pass

