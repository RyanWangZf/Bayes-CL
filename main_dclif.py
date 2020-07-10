# -*- coding: utf-8 -*-
# To achieve a curriculum learning method based on group influence function.

from collections import defaultdict
import numpy as np
import pdb, os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad

# print("load matplotlib")
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt

from model import SimpleCNN
from dataset import load_data

from utils import setup_seed, img_preprocess, impose_label_noise
from config import opt

def train(**kwargs):
    # pre-setup
    setup_seed(opt.seed)
    opt.parse(kwargs)
    log_dir = os.path.join(opt.result_dir, "dclif_" + opt.model + "_"  + opt.data_name + "_" + opt.print_opt)
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

    # set the last linear layer's weight and bias for calculating if
    theta = [model.dense.weight, model.dense.bias]

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

        if epoch == 0:
            # the first epoch optimizes as usual
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

        else:
            # execute curriculum learning
            for idx in range(num_all_batch):
                # sampling by curriculum score
                curriculum_id = np.random.choice(np.arange(len(curriculum_prob)), p=curriculum_prob)

                # get curriculum sample indices and random sample
                all_sample_idx = curriculum_list[curriculum_id]
                np.random.shuffle(all_sample_idx)
                batch_idx = all_sample_idx[:batch_size]

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

        # ---------------------------------------------------------------
        # start to compute the curriculum, follow the batch of training data
        # ---------------------------------------------------------------

        print("computing curriculum scores.")

        # ---------------------------------------------------------------
        # compute s_va, g_va, which are same for all groups of training data
        # ---------------------------------------------------------------
        va_loss_avg = evaluate_va_loss(model, x_va, y_va)
        g_va = list(grad(va_loss_avg, theta, create_graph=True))
        grad_val = [g.detach() for g in g_va]
        s_va = cal_inverse_hvp_lissa(model, x_tr, y_tr, grad_val, theta)

        curriculum_size = opt.curriculum_size
        num_curriculum_batch = int(np.ceil(len(x_tr)/curriculum_size))
        
        # create curriculum list key:value is {curriculum id: list of training sample idx}
        curriculum_list = defaultdict(list)
        curriculum_score = []

        # TODO
        # now we get curriculum randomly split, but if its possible for us to
        # split the curriculum with infinitesimal jackknife (individual influence function)
        all_tr_idx = np.arange(len(x_tr))
        np.random.shuffle(all_tr_idx)

        for idx in range(num_curriculum_batch):
            batch_idx = all_tr_idx[idx*curriculum_size:(idx+1)*curriculum_size]

            curriculum_list[idx] = batch_idx

            # ---------------------------------------------------------------
            # compute g_u and s_u = s_va * H(u), which are unique for each group
            # ---------------------------------------------------------------
            c_loss_avg = evaluate_va_loss(model, x_tr[batch_idx], y_tr[batch_idx])
            g_u = list(grad(c_loss_avg, theta, create_graph=True))
            s_u = []
            for j, g_ in enumerate(g_u):
                var_t = torch.sum(g_ * g_va[j])
                s_u.append(list(grad(var_t, theta[j], create_graph=True))[0])

            # ---------------------------------------------------------------
            # compute s_u * H^-1, which are unique for each group
            # ---------------------------------------------------------------
            # Elementwise products
            elemwise_products = 0
            for grad_elem, v_elem in zip(g_u, s_u):
                elemwise_products += torch.sum(grad_elem * v_elem.detach())
            
            # Second backprop
            hv_s_u_H = grad(elemwise_products, theta, create_graph=True)

            # Take one-step lissa to approximate (Taylor expansion)
            s_u_h_inv = [ _v - _hv.detach() for _v, _hv in zip(s_u, hv_s_u_H)]

            # ---------------------------------------------------------------
            # compute group influence
            # ---------------------------------------------------------------
            phi = compute_group_influence(len(batch_idx), len(x_tr), s_va, s_u_h_inv, g_u)

            curriculum_score.append(phi.data.cpu().numpy())

        # get temperature scaled softmax of the scores
        curriculum_score = np.array(curriculum_score)

        # TODO schedule temperature scaling
        T = 0.5
        curriculum_prob = compute_curriculum_prob(curriculum_score, T)

        print("curriculum scores computed done.")


def compute_group_influence(num_u, num_all, s_va, s_u_h_inv, g_u):
    """Compute group-wise influence function.
    Args:
        num_u: number of training samples within this group;
        num_all: number of all training samples;
        s_va: g_va * H^-1, a list
        s_u_h_inv: s_u * H^-1, a list
        g_u: nabla L_u, a list with size of elements equals theta
    """
    # compute constant coefficients
    S, U = num_all, num_u
    p = U / S
    c1, c2 = p/(1-p), -1/(1-p) * U/S

    phi = 0
    for i in range(len(s_va)):
        t1 = torch.sum(s_va[i] * g_u[i])
        # compute hessian inverse vector product
        t2 = torch.sum(s_u_h_inv[i] * g_u[i])
        phi_ = c2 * t1 + c1 * c2 * (t1 - t2)
        phi = phi + phi_
    
    return phi


def cal_inverse_hvp_lissa(model,
    x_tr,
    y_tr,
    grad_val,
    theta,
    batch_size=128,
    damp=0.01,
    scale=10.0,
    recursion_depth=100,
    tol=1e-5,
    verbose=False,):
    """Cal group influence function of a batch of data by lissa.
    Return:
        h_estimate: the estimated g_va * h^-1
        g_va: the gradient w.r.t. val set, g_va
    """
    def _compute_diff(h0, h1):
        assert len(h0) == len(h1)
        diff_ratio = [1e8] * len(h0)
        for i in range(len(h0)):
            h0_ = h0[i].detach().cpu().numpy()
            h1_ = h1[i].detach().cpu().numpy()
            norm_0 = np.linalg.norm(h0_) 
            norm_1 = np.linalg.norm(h1_)
            abs_diff = abs(norm_0 - norm_1)
            diff_ratio[i] = abs_diff / norm_0

        return max(diff_ratio)

    model.eval()

    # start recurssively estimate the inv-hvp
    h_estimate = grad_val.copy()
    all_tr_idx = np.arange(len(x_tr))

    for i in range(recursion_depth):
        h_estimate_last = h_estimate
        # randomly select a batch from training data
        this_idx = np.random.choice(all_tr_idx, batch_size, replace=False)

        x_tr_this = x_tr[this_idx]
        y_tr_this = y_tr[this_idx]

        pred = model(x_tr_this)
        batch_loss = F.cross_entropy(pred, y_tr_this, reduction="mean")

        hv = hvp(batch_loss, theta, h_estimate)
        h_estimate = [ _v + (1 - damp) * _h_e - _hv.detach() / scale 
            for _v, _h_e, _hv in zip(grad_val, h_estimate, hv)]

        diff_ratio = _compute_diff(h_estimate, h_estimate_last)

        if i % 10 == 0 and verbose:
            print("[LISSA] diff:", diff_ratio)

        if diff_ratio <= tol:
            print("[LISSA] reach tolerance in epoch", int(i+1))
            break

    if i == recursion_depth-1:
        print("[LISSA] reach max recursion_depth, stop.")

    return h_estimate

def hvp(y, w, v):
    """Multiply the Hessians of y and w by v.
    Uses a backprop-like approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians.
    Example: if: y = 0.5 * w^T A x then hvp(y, w, v) returns and expression
    which evaluates to the same values as (A + A.t) v.
    Arguments:
        y: scalar/tensor, for example the output of the loss function
        w: list of torch tensors, tensors over which the Hessian
            should be constructed
        v: list of torch tensors, same shape as w,
            will be multiplied with the Hessian
    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.
    Raises:
        ValueError: `y` and `w` have a different length."""

    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem.detach())

    # Second backprop
    return_grads = grad(elemwise_products, w, create_graph=True)

    return return_grads

def evaluate_va_loss(model, x_va, y_va, batch_size=1000):
    va_loss = 0
    num_all_batch = int(np.ceil(len(x_va)/batch_size))
    for idx in range(num_all_batch):
        pred = model(x_va[idx*batch_size:(idx+1)*batch_size])
        va_loss = va_loss + F.cross_entropy(pred, y_va[idx*batch_size:(idx+1)*batch_size],
                reduction="sum")

    va_loss_avg = va_loss / len(x_va)
    return va_loss_avg

def compute_curriculum_prob(score, T):
    """
    Args:
        score: raw curriculum scores
        T: temperature for softmax
    """
    # debug minus?
    x = - (score - score.min()) / (score.max() - score.min())
    # x = (score - score.min()) / (score.max() - score.min())
    x_exp = np.exp(x/T)
    prob = x_exp / np.sum(x_exp)
    return prob

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
