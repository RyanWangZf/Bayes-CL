# -*- coding: utf-8 -*-

import numpy as np
import pdb, os
from numpy import linalg as la

import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim

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
    log_dir = os.path.join(opt.result_dir, "ucl_" + opt.model + "_"  + opt.data_name + "_" + opt.print_opt)
    print("output log dir", log_dir)

    if not torch.cuda.is_available():
        print("[WARNING] do not find cuda device, change use_gpu=False!")
        opt.use_gpu = False

    ckpt_dir = os.path.join(os.path.join(log_dir, "ckpt"))
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # intermediate file for early stopping
    early_stop_ckpt_path = os.path.join(ckpt_dir, "best_va.pth")
    base_model_path = os.path.join(log_dir, "base_model.pth")
    score_path = os.path.join(log_dir, "score_{}.npy".format(opt.bnn))

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

    if os.path.exists(base_model_path):
        load_model(base_model_path, model)
        print("load pretrained base model.")
    
    else:
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
            5,)

        # evaluate model on test set
        pred_te = predict(model, x_te)
        if num_class > 2:
            acc_te = eval_metric(pred_te, y_te)
        else:
            acc_te = eval_metric_binary(pred_te, y_te)

        print("curriculum: {}, acc: {}".format(0, acc_te.item()))
        te_acc_list.append(acc_te.item())

        # save base model
        save_model(base_model_path, model)
        print("base model saved in", base_model_path)
    
    # try to do uncertainty inference
    # first let's calculate Fisher Information Matrix
    if num_class > 2:
        emp_fmat = compute_emp_fisher_multi(model, x_tr, y_tr, num_class, 100)
    else:
        emp_fmat = compute_emp_fisher_binary(model, x_tr, y_tr, 100)

    # compute a PSD precision matrix based on empirical fisher information matrix
    prec_mat = compute_precision_mat(emp_fmat, len(x_tr))
    model.set_bayesian_precision(prec_mat)

    # compute Bayesian uncertainty form for score of samples
    if num_class > 2:
        tr_score = compute_uncertainty_score(model, x_tr, y_tr, opt.bnn, 32, 5)
    else:
        tr_score = compute_uncertainty_score_binary(model, x_tr, y_tr, opt.bnn, 32, 5)
    np.save(score_path, tr_score)
    print("score saved.")
    
    # design difficulty by uncertainty difficulty
    curriculum_idx_list = one_step_pacing(y_tr, tr_score, num_class, 0.2)

    # training on simple set
    model._initialize_weights()

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

# ----------------
# BNN toolkit
# ----------------

def compute_uncertainty_score(model, x_tr, y_tr, mode="mix", batch_size=64, T=5):
    """Compute uncertainty guided score of samples.
    Args:
        mode: should in "mix", "epis", "alea" and "snr";
            "epis" & "alea": epistemic and aleatoric only
            "mix": epis + alea
            "snr": pred / mix, is larger is easier (above three are smaller is easier)
        batch_size, T: # of samples per batch during inference and # of MC sampling,
            note that due to the parallelled implementation of MC by sample copy trick,
            the true batch_size would be (batch_size * T), which might encounter oom problem if
            both are set too large.
    """
    assert mode in ["mix", "epis", "alea", "snr"]
    model.eval()
    all_tr_idx = np.arange(len(x_tr))
    num_all_batch = int(np.ceil(len(all_tr_idx)/batch_size))

    score_list = []
    for idx in range(num_all_batch):
        batch_idx = all_tr_idx[idx*batch_size:(idx+1)*batch_size]
        x_batch = x_tr[batch_idx]
        y_batch = y_tr[batch_idx]
        with torch.no_grad():
            pred, epis, alea = model.bayes_infer(x_batch, T=T)
        
        indices = (np.arange(len(y_batch)), y_batch.cpu().numpy())

        if mode == "mix":
            score_ = epis[indices] + alea[indices]
        elif mode == "epis":
            score_ = epis[indices]
        elif mode == "alea":
            score_ = alea[indices]
        elif mode == "snr":
            score_ = epis[indices] + alea[indices]
            score_ = pred.mean(0)[indices] / (1e-16 + score_)
            score_ = - score_ # snr is smaller is harder

        score_list.extend(score_.cpu().numpy().tolist())

    return np.array(score_list)

def compute_uncertainty_score_binary(model, x_tr, y_tr, mode="mix", batch_size=64, T=5):
    """Compute uncertainty guided score of samples.
    Args:
        mode: should in "mix", "epis", "alea" and "snr";
            "epis" & "alea": epistemic and aleatoric only
            "mix": epis + alea
            "snr": pred / mix, is larger is easier (above three are smaller is easier)
        batch_size, T: # of samples per batch during inference and # of MC sampling,
            note that due to the parallelled implementation of MC by sample copy trick,
            the true batch_size would be (batch_size * T), which might encounter oom problem if
            both are set too large.
    """
    assert mode in ["mix", "epis", "alea", "snr"]
    model.eval()
    all_tr_idx = np.arange(len(x_tr))
    num_all_batch = int(np.ceil(len(all_tr_idx)/batch_size))

    score_list = []
    for idx in range(num_all_batch):
        batch_idx = all_tr_idx[idx*batch_size:(idx+1)*batch_size]
        x_batch = x_tr[batch_idx]
        y_batch = y_tr[batch_idx]
        with torch.no_grad():
            pred, epis, alea = model.bayes_infer(x_batch, T=T)
        
        if mode == "mix":
            score_ = epis + alea
        elif mode == "epis":
            score_ = epis
        elif mode == "alea":
            score_ = alea
        elif mode == "snr":
            score_ = epis + alea
            score_ = pred.mean(0) / (1e-16 + score_)
            score_ = - score_ # snr is smaller is harder

        score_list.extend(score_.cpu().numpy().tolist())

    return np.array(score_list).squeeze()


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
        print("precision matrix is not PD, do conversion.")
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
