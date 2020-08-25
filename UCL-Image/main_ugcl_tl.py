# -*- coding: utf-8 -*-

"""An implementation of Uncertainty-guided Bayesian Transfer Curriculum Learning.
Use ResNet50 for predicting uncertainty of data.
"""

import numpy as np
import pdb, os
from numpy import linalg as la

import scipy
from scipy import sparse
from scipy.sparse import linalg as sp_la

import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
import torchvision.models as th_models

import math

from dataset import load_data

from utils import setup_seed, img_preprocess, impose_label_noise, babystep_pacing
from model import LeNet, ResNet
from utils import save_model, load_model, predict, train, eval_metric, eval_metric_binary
from config import opt

def main(**kwargs):
    # pre-setup
    setup_seed(opt.seed)
    opt.parse(kwargs)
    log_dir = os.path.join(opt.result_dir, "ugcl_tl_" + opt.model + "_"  + opt.data_name + "_" + opt.print_opt)
    print("output log dir", log_dir)

    ckpt_dir = os.path.join(os.path.join(log_dir, "ckpt"))
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # intermediate file for early stopping
    early_stop_ckpt_path = os.path.join(ckpt_dir, "best_va.pth")
    early_stop_stu_path = os.path.join(ckpt_dir, "best_va_stu_{}.pth".format(opt.bnn))

    resnet50_ckpt_path = os.path.join(ckpt_dir, "resnet.pth")
    prec_mat_path = os.path.join(log_dir, "prec_mat.npy")
    score_path = os.path.join(log_dir, "score_{}.npy".format(opt.bnn))
    output_result_path = os.path.join(log_dir, "ugcl_{}.result".format(opt.bnn))

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

    # train resnet for transferable uncertainty
    raw_resnet = th_models.resnet50(pretrained = True)
    resnet50 = ResNet(raw_resnet, num_class)

    if opt.use_gpu:
        resnet50.cuda()

    # finetune the resnet50 on this dataset
    if os.path.exists(resnet50_ckpt_path):
        print("load from finetuned resnet50 model.")
        load_model(resnet50_ckpt_path, resnet50)

    else:
        all_tr_idx = np.arange(len(x_tr))
        _ = train(resnet50, all_tr_idx, x_tr, y_tr, x_va, y_va,
            50, 128, 1e-3, 1e-5, resnet50_ckpt_path, 5)

    if os.path.exists(prec_mat_path):
        print("load precomputed PD precision matrix.")
        prec_mat = np.load(prec_mat_path)
        prec_mat = torch.from_numpy(prec_mat).to(x_tr.device)
        prec_mat = prec_mat.float()

    else:
        # compute empirical fisher matrix for resnet50, we need hook for penultimate layer outputs.
        if num_class > 2:
            emp_fmat = compute_emp_fisher_multi(resnet50, x_tr, y_tr, num_class, 100)
        else:
            emp_fmat = compute_emp_fisher_binary(resnet50, x_tr, y_tr, 100)

        # ############################
        # switch resnet50 to Bayesian version!
        # ############################

        # take block diagonal to speed up the processing process
        emp_fmat = set_block_diagonal(emp_fmat, 20)
        prec_mat = compute_precision_mat(emp_fmat, len(x_tr))
        np.save(prec_mat_path, prec_mat.cpu().numpy())
        print("precision matrix computed.")
        prec_mat = prec_mat.to(x_tr.device).float()
    
    # set bayesian matrix
    set_bayesian_precision(resnet50, prec_mat)

    # compute Bayesian uncertainty form for score of samples
    tr_score = compute_uncertainty_score(resnet50, x_tr, y_tr, opt.bnn, 32, 5)
    np.save(score_path, tr_score)
    print("score path saved.")

    # design difficulty by uncertainty difficulty
    curriculum_idx_list = babystep_pacing(y_tr, tr_score, num_class, 2)

    # load model
    _, in_channel, in_size, _ = x_tr.shape
    model = LeNet(num_class=num_class, in_size=in_size, in_channel=in_channel)
    if opt.use_gpu:
        model.cuda()
    model.use_gpu = opt.use_gpu

    all_tr_idx = np.arange(len(x_tr))

    va_acc = train(model,
            curriculum_idx_list[0],
            x_tr, y_tr,
            x_va, y_va,
            30,
            opt.batch_size // 2,
            opt.lr,
            opt.weight_decay,
            early_stop_stu_path,
            5)

    # evaluate test acc
    pred_te = predict(model, x_te)
    if num_class > 2:
        acc_te = eval_metric(pred_te, y_te)
    else:
        acc_te = eval_metric_binary(pred_te, y_te)
    first_stage_acc = acc_te
    print("first stage acc:", first_stage_acc)

    # training on all set
    va_acc = train(model,
        all_tr_idx,
        x_tr, y_tr,
        x_va, y_va,
        50,
        opt.batch_size,
        opt.lr,
        opt.weight_decay,
        early_stop_stu_path,
        5)

    pred_te = predict(model, x_te)
    acc_te = eval_metric(pred_te, y_te)
    print("curriculum: {}, acc: {}".format("one-step pacing", acc_te.item()))
    res_list = [str(_) for _ in [first_stage_acc.item(), acc_te.item()]]
    with open(output_result_path, "w") as f:
        f.write("\n".join(res_list) + "\n")

# ----------------
# Bayesian toolkit
# ----------------
def set_block_diagonal(emp_fmat, max_range=20):
    """Only select a diagonal block matrix from the original emp_fmat,
    to reduce the computation complexity next.
    If max_range set 0 or 1, we only pick the diagonal elements from emp_fmat.
    """
    print("pick diagonal blocks from matrix, range is", max_range)

    if max_range in [0,1]:
        # only take diagonal element
        rec_emp_fmat = torch.diag(torch.diag(emp_fmat))

    else:
        # first transform to block diagonal matrix
        import scipy.linalg
        mat = emp_fmat.cpu().numpy()
        temp = extract_block_diag(mat, max_range, 0)
        blocks = [_ for _ in temp]
        mat_block_diag = scipy.linalg.block_diag(*blocks)
        rec_emp_fmat = torch.from_numpy(mat_block_diag).to(emp_fmat.device)

    return rec_emp_fmat        


def extract_block_diag(A, M, k=0):
    """Extracts blocks of size M from the kth diagonal
    of square matrix A, whose size must be a multiple of M."""

    # Check that the matrix can be block divided
    if A.shape[0] != A.shape[1] or A.shape[0] % M != 0:
        raise InterruptedError('Matrix must be square and a multiple of block size')

    # Assign indices for offset from main diagonal
    if abs(k) > M - 1:
        raise InterruptedError('kth diagonal does not exist in matrix')

    elif k > 0:
        ro = 0
        co = abs(k)*M 
    elif k < 0:
        ro = abs(k)*M
        co = 0
    else:
        ro = 0
        co = 0

    blocks = np.array([A[i+ro:i+ro+M,i+co:i+co+M] 
                       for i in range(0,len(A)-abs(k)*M,M)])
    return blocks

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
            pred, epis, alea = bayes_infer(model, x_batch, T)
            # pred, epis, alea = model.bayes_infer(x_batch, T=T)
        
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

def sparse_nearestPD(A):
    """Find the nearest positive-definite matrix to input,
    with scipy.sparse implementation. Input is np.array!
    """

    def isPD(B):
        """Returns true when input is positive-definite, via Cholesky"""
        try:
            _ = la.cholesky(B)
            return True
        except la.LinAlgError:
            return False

    A_sp = sparse.csr_matrix(A)
    B = (A_sp + A_sp.T) / 2
    _, s, V = sp_la.svds(B, k=100)
    print("svd in sparse nearest PD done.")

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
    print("svd in nearest PD done.")

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
        print("nearestPD:", i)
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
    print("compute precision matrix from the empirical fisher matrix.")
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
        # emp_fmat_pd = nearestPD(emp_fmat_ar)
        emp_fmat_pd = sparse_nearestPD(emp_fmat_ar)

        emp_fmat = torch.from_numpy(emp_fmat_pd).to(emp_fmat.device)
        
    prec_mat = num_all_sample**2 * emp_fmat
    return prec_mat

def compute_emp_fisher_binary(model, x_tr, y_tr, batch_size=100):
    """Compute empirical fisher matrix with penultimate layer outpus with hooks.
    """
    # define hooks
    def hook(module, input, output):
        """return inputs of the last fc layer.
        """
        # print("hooker working.")
        return output, input

    model.eval()
    all_tr_idx = np.arange(len(x_tr))
    np.random.shuffle(all_tr_idx)

    num_all_batch = int(np.ceil(len(x_tr)/batch_size))

    # register hook
    hook_handle = model.fc.register_forward_hook(hook)

    pmat = None
    for idx in range(num_all_batch):
        x_batch = x_tr[batch_size*idx: batch_size*(idx+1)]
        y_batch = y_tr[batch_size*idx: batch_size*(idx+1)]

        with torch.no_grad():
            # pred, x_h = model(x_batch, hidden=True)
            pred, x_h = model(x_batch)
            x_h = x_h[0]
            pred = torch.softmax(pred, 1)

        # compute grad w
        grad_w = torch.unsqueeze(1/ len(pred) * x_h.T @ (pred[:,0] - y_batch.float()), 1)
        pmat_ = grad_w  @  grad_w.T
        if pmat is None:
            pmat = pmat_
        else:
            pmat = pmat + pmat_

    pmat = 1 / num_all_batch * pmat

    hook_handle.remove()
    return pmat

def compute_emp_fisher_multi(model, x_tr, y_tr, num_class, batch_size=100):
    """Compute empirical fisher matrix with penultimate layer outpus with hooks.
    """
    # define hooks
    def hook(module, input, output):
        """return inputs of the last fc layer.
        """
        # print("hooker working.")
        return output, input

    def one_hot_transform(y, num_class=10):
        one_hot_y = F.one_hot(y, num_classes=num_class)
        return one_hot_y.float()

    model.eval()

    all_tr_idx = np.arange(len(x_tr))
    np.random.shuffle(all_tr_idx)

    num_all_batch = int(np.ceil(len(x_tr)/batch_size))
    
    # register hook
    hook_handle = model.fc.register_forward_hook(hook)

    pmat = None
    for idx in range(num_all_batch):
        x_batch = x_tr[batch_size*idx: batch_size*(idx+1)]
        y_batch = y_tr[batch_size*idx: batch_size*(idx+1)]

        y_oh_batch = one_hot_transform(y_batch, num_class)

        with torch.no_grad():
            # pred, x_h = model(x_batch, hidden=True)
            pred, x_h = model(x_batch)
            x_h = x_h[0]
            pred = torch.softmax(pred, 1)
                
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

    hook_handle.remove()
    return pmat

def set_bayesian_precision(network, prec_mat):
    """Set the last linear layer covariance matrix by the given postive-definite precision matrix.
    """
    from model import BayesLinear
    # replace the last layer by BayesianLinear layer, copy the mu matrix.
    num_class, num_in_feature = network.fc.weight.shape
    w_mu, b_mu = network.fc.weight, network.fc.bias

    # copy mean vector
    bayes_fc = BayesLinear(num_in_feature, num_class, precision_matrix=prec_mat, bias=True)
    bayes_fc.W_mu.data = w_mu.data
    bayes_fc.bias_mu.data = b_mu.data

    # replace network fc layer
    network.fc = bayes_fc
    print("bayesian replacement done.")

def bayes_infer(model, inputs, T=10):
    """Do bayesian inference on the modifed bayesian neural network.
    Note that `fc` is the output layer.
    """
    # the output dense layer must have already been Bayesian
    assert model.fc.eps_distribution is not None
    assert T > 0

    batch_size = inputs.shape[0]

    # deterministic mapping
    x = model.encoder(inputs)

    out = model.fc.bayes_forward(x, T) # T, ?, C

    num_class = out.shape[2]

    # transform to probability distribution
    # &
    # compute uncertainty
    if  num_class == 1:
        # binary classification scenario
        out = torch.sigmoid(out)

        # ---------------------
        # epistemic uncertainty
        # ---------------------
        pbar = out.mean(0)
        temp = out - pbar.unsqueeze(0)
        temp = temp.unsqueeze(3)
        epis = torch.einsum("ijkm,ijnm->ijkn", temp, temp).mean(0) # ?, 1, 1 for binary
        # get diagonal elements of epis
        epis = torch.einsum("ijj->ij", epis) # ?, 1 (?,c for multi-class)

        # ---------------------
        # aleatoric uncertainty
        # ---------------------
        temp = out.unsqueeze(3) # T, ?, c, 1
        phat_op = torch.einsum("ijkm,ijnm->ijkn", temp, temp) # T, ?, c, c
        alea = (out.unsqueeze(2) - phat_op).mean(0).squeeze(2)
        
    else:
        # multi-class
        out = torch.softmax(out, 2) # T, ?, c

        # ---------------------
        # epistemic uncertainty
        # ---------------------
        pbar = out.mean(0) # ?, c
        temp = out - pbar.unsqueeze(0) # T, ?, c
        temp = temp.unsqueeze(3) # T, ?, c, 1
        epis = torch.einsum("ijkm,ijnm->ijkn", temp, temp).mean(0) # ?, c, c
        # get diagonal elements of epis
        epis = torch.einsum("ijj->ij", epis) # ?, c for multi-class

        # ---------------------
        # aleatoric uncertainty
        # ---------------------
        temp = out.unsqueeze(3) # T, ?, c, 1
        phat_op = torch.einsum("ijkm,ijnm->ijkn", temp, temp) # T, ?, c, c
        
        temp = torch.repeat_interleave(temp, num_class , 3) # T, ?, c, c
        eye_mat = torch.eye(num_class).unsqueeze(0).unsqueeze(0).to(inputs.device) # 1, 1, c, c
        diag_phat = temp * eye_mat
        alea = (diag_phat - phat_op).mean(0) # ?, c, c
        # get diagonal elements of alea
        alea = torch.einsum("ijj->ij", alea) # ?, c for multi-class

    return out, epis, alea


if __name__ == "__main__":
    import fire
    fire.Fire()