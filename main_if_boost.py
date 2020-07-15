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
from utils import ExponentialScheduler, LinearScheduler, ExponentialDecayScheduler
from utils import save_model, load_model
from config import opt

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

    for epoch in range(num_epoch):
        total_loss = 0
        model.train()
        np.random.shuffle(sub_idx)

        for idx in range(num_all_tr_batch):
            batch_idx = sub_idx[idx*batch_size:(idx+1)*batch_size]
            x_batch = x_tr[batch_idx]
            y_batch = y_tr[batch_idx]

            pred = model(x_batch)
            loss = F.cross_entropy(pred, y_batch, 
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
        acc_va = eval_metric(pred_va, y_va)
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
    log_dir = os.path.join(opt.result_dir, "dclif_" + opt.model + "_"  + opt.data_name + "_" + opt.print_opt)
    ckpt_dir = os.path.join(os.path.join(log_dir, "ckpt"))
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    early_stop_ckpt_path = os.path.join(ckpt_dir, "best_va.pth")

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
    model = SimpleCNN(10)
    if opt.use_gpu:
        model.cuda()
    model.use_gpu = opt.use_gpu

    # set the last linear layer's weight and bias for calculating if
    theta = [model.dense.weight, model.dense.bias]

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
        3)

    # evaluate model on test set
    pred_te = predict(model, x_te)
    acc_te = eval_metric(pred_te, y_te)
    print("curriculum: {}, acc: {}".format(0, acc_te.item()))
    te_acc_list.append(acc_te.item())

    # calculate influence and do curriculum selection
    model.eval()

    # ---------------------------------------------------------------
    # compute s_va, g_va, which are same for all groups of training data
    # ---------------------------------------------------------------
    va_loss_avg = evaluate_va_loss(model, x_va, y_va)
    g_va = list(grad(va_loss_avg, theta, create_graph=True))
    grad_val = [g.detach() for g in g_va]

    s_va = cal_inverse_hvp_lissa(model, x_tr, y_tr, grad_val, theta,
        batch_size=512,
        damp=0.01,
        scale=25.0,
        recursion_depth=100,)
    
    # ---------------------------------------------------------------
    # start to compute influence score for each sample
    # ---------------------------------------------------------------
    all_tr_idx = np.arange(len(x_tr))
    phi_indiv = compute_individual_influence(model, x_tr, y_tr, s_va)

    # ---------------------------------------------------------------
    # do influence sampling to arrange the curriculum
    # ---------------------------------------------------------------
    
    # curriculum size
    num_all_curriculum = int(
        np.ceil(len(all_tr_idx)/
        opt.curriculum_size))
    
    curriculum_idx_list = build_curriculum_plan(
        y_tr,
        phi_indiv,
        num_class,
        opt.curriculum_size,
        num_all_curriculum,
        T = 1.0)

    # ------------------------------------------
    # 'boost' curriculum
    # ------------------------------------------
    boost_tr_idx = np.concatenate(curriculum_idx_list[:2])
    # boost_tr_idx = all_tr_idx
    va_acc = train(model,
            boost_tr_idx,
            x_tr, y_tr,
            x_va, y_va, 
            opt.num_epoch,
            opt.batch_size,
            1e-4,
            opt.weight_decay,
            early_stop_ckpt_path,
            5)
    pred_te = predict(model, x_te)
    acc_te = eval_metric(pred_te, y_te)
    print("curriculum: {}, acc: {}".format("boost", acc_te.item()))
    te_acc_list.append(acc_te.item())

    # ------------------------------------------
    # 'default' curriculum
    # ------------------------------------------
    curriculum_lr_list = np.array([opt.lr] * num_all_curriculum)
    curriculum_lr_list[-2:] = 1e-4
    # start training on curriculum plan
    # init model
    # model._initialize_weights()
    # tr_idx = []
    # for cidx in range(num_all_curriculum):
    #     tr_idx.extend(curriculum_idx_list[cidx])
    #     np.random.shuffle(tr_idx)

    #     va_acc = train(model,
    #         tr_idx,
    #         x_tr, y_tr,
    #         x_va, y_va, 
    #         opt.num_epoch,
    #         opt.batch_size,
    #         curriculum_lr_list[cidx],
    #         opt.weight_decay,
    #         early_stop_ckpt_path,
    #         5)
        
        # # evaluate model on test set
        # pred_te = predict(model, x_te)
        # acc_te = eval_metric(pred_te, y_te)
        # print("curriculum: {}, acc: {}".format(cidx+1, acc_te.item()))
        # te_acc_list.append(acc_te.item())

    print(te_acc_list)

def build_curriculum_plan(y, 
    phi,
    num_class,
    curriculum_size,
    num_all_curriculum,
    T=1.0):
    """Arrange curriculum indices by individual influence score phi.
    Need num_class to make sure the subsampled dataset to be balanced in temrs of label distribution.
    """
    # init local variables
    all_class = np.arange(num_class)
    remain_idx = np.arange(len(phi))
    curriculum_size_c = int(curriculum_size / num_class)
    rest_curriculum_size = curriculum_size - (num_class - 1) * curriculum_size_c
    curriculum_list = []

    # compute probabilities
    prob_pi = compute_prob_by_phi(phi, T)

    for i in range(num_all_curriculum-1):
        sub_idx = []

        remain_y = y[remain_idx].cpu()

        for c in all_class:
            remain_idx_c = remain_idx[remain_y == c]
            prob_pi_remain_c = prob_pi[remain_idx_c]

            if c != num_class-1:
                sub_idx_c = sampling(
                    remain_idx_c,
                    prob_pi_remain_c, 
                    curriculum_size_c)
            else:
                sub_idx_c = sampling(
                    remain_idx_c,
                    prob_pi_remain_c, 
                    rest_curriculum_size)
            
            sub_idx.extend(sub_idx_c)

        curriculum_list.append(sub_idx)

        # delete the all ready selected samples
        remain_idx = np.setdiff1d(remain_idx, sub_idx)

    curriculum_list.append(remain_idx)
    return curriculum_list

def compute_prob_by_phi(phi, T):
    sigma_phi = phi.std()
    avg_phi = phi.mean()
    # compute prob
    if sigma_phi == 0:
        # all phis are same
        prob_pi = np.array([0.5]*len(phi))
    else:
        phi_std = phi - avg_phi
        a_param = T / (1e-9 + phi.max() - phi.min())
        prob_pi = 1 / (1 + np.exp(a_param * phi_std))
        
    return prob_pi

def sampling_by_phi(all_idx, phi, T, ratio=0.5):
    """Args:
        all_idx: indices waited to be subsampled;
        phi: influence scores of all samples;
        T: hyper parameter indicates the 'temperature' of the sigmoid function;
        ratio: defined sample ratio;
    """
    assert ratio <= 1.0 and ratio >= 0
    num_target = int(len(all_idx) * ratio)
    sigma_phi = phi.std()
    avg_phi = phi.mean()
    # compute prob
    if sigma_phi == 0:
        # all phis are same
        prob_pi = np.array([0.5]*len(all_idx))
    else:
        phi_std = (phi - avg_phi)
        a_param = T / (1e-9 + phi.max() - phi.min())
        prob_pi = 1 / (1 + np.exp(a_param * phi_std))

    # do sampling
    sub_idx = sampling(prob_pi, ratio, num_target)
    return all_idx[sub_idx]

def sampling(all_idx, prob_pi, obj_sample_size):
    num_sample = prob_pi.shape[0]

    sb_idx = None
    iteration = 0
    while True:
        rand_prob = np.random.rand(num_sample)
        iter_idx = all_idx[rand_prob < prob_pi]
        if sb_idx is None:
            sb_idx = iter_idx
        else:
            new_idx = np.setdiff1d(iter_idx, sb_idx)
            diff_size = obj_sample_size - sb_idx.shape[0]
            if new_idx.shape[0] < diff_size:
                sb_idx = np.union1d(iter_idx, sb_idx)
            else:
                new_idx = np.random.choice(new_idx, diff_size, replace=False)
                sb_idx = np.union1d(sb_idx, new_idx)
        iteration += 1
        if sb_idx.shape[0] >= obj_sample_size:
            sb_idx = np.random.choice(sb_idx,obj_sample_size,replace=False)
            return sb_idx

        if iteration > 100:
            diff_size = obj_sample_size - sb_idx.shape[0]
            leave_idx = np.setdiff1d(all_idx, sb_idx)
            # left samples are sorted by their IF
            # leave_idx = leave_idx[np.argsort(prob_pi[leave_idx])[-diff_size:]]
            leave_idx = np.random.choice(leave_idx,diff_size,replace=False)
            sb_idx = np.union1d(sb_idx, leave_idx)
            return sb_idx

def compute_individual_influence(model, x_tr, y_tr, s_va):
    """Compute individual influence for spliting curriculum;
    """
    def one_hot_transform(y, num_class=10):
        one_hot_y = F.one_hot(y, num_classes=num_class)
        return one_hot_y.float()

    num_class = torch.unique(y_tr).shape[0]
    y_tr_oh = one_hot_transform(y_tr, num_class)

    batch_size = 1000
    num_all_batch = int(np.ceil(len(x_tr)/batch_size))

    if_list = []
    for idx in range(num_all_batch):
        x_batch = x_tr[batch_size*idx: batch_size*(idx+1)]
        y_batch = y_tr_oh[batch_size*idx: batch_size*(idx+1)]

        pred, x_h = model(x_batch, hidden=True)
        diff_pred = pred - y_batch

        x_h = torch.unsqueeze(x_h, 1)
        partial_J_theta = x_h * torch.unsqueeze(diff_pred, 2)
        partial_J_theta = partial_J_theta.view(-1, 
            partial_J_theta.shape[1] * partial_J_theta.shape[2]).detach()

        if model.use_gpu:
            identical_mat = torch.eye(num_class).cuda()
        else:
            identical_mat = torch.eye(num_class)

        partial_J_b = torch.mm(diff_pred, identical_mat)

        predicted_loss_diff = - (torch.mm(partial_J_theta, s_va[0].view(-1,1)) + 
            torch.mm(partial_J_b, s_va[1].view(-1,1)))
        
        if_list.append(predicted_loss_diff.view(-1).detach().cpu().numpy())

    pred_all_if = np.concatenate(if_list)

    return pred_all_if


def cal_inverse_hvp_lissa(model,
    x_tr,
    y_tr,
    grad_val,
    theta,
    batch_size=128,
    damp=0.01,
    scale=25.0,
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

def evaluate_va_loss(model, x_va, y_va, batch_size=512):
    model.eval()
    num_all_batch = int(np.ceil(len(x_va)/batch_size))
    all_idx = np.arange(len(x_va))

    pred_all = []
    for idx in range(num_all_batch):
        # print(idx)
        this_idx = all_idx[idx*batch_size:(idx+1)*batch_size]
        x_batch = x_va[this_idx]
        pred = model(x_batch)
        pred_all.append(pred)

    all_pred = torch.cat(pred_all, 0)
    va_loss_avg = F.cross_entropy(all_pred, y_va,reduction="mean")

    return va_loss_avg

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

def train_deprecated(**kwargs):
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

    model.use_gpu = opt.use_gpu

    # set the last linear layer's weight and bias for calculating if
    theta = [model.dense.weight, model.dense.bias]

    # compute for batch learning
    batch_size = opt.batch_size

    # initialize optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, weight_decay=opt.weight_decay)

    # debug
    T = 1.0
    te_acc_list = []
    print("start training")

    # debug
    ratio = 0.8
    num_all_train = 0
    all_tr_idx = np.arange(len(x_tr))
    for epoch in range(opt.num_epoch):
        total_loss = 0
        num_all_batch = int(np.ceil(len(all_tr_idx)/batch_size))
        np.random.shuffle(all_tr_idx)

        model.train()
        if epoch == 0:
            # the first epoch optimizes as usual
            for idx in range(num_all_batch):
                batch_idx = all_tr_idx[idx*batch_size:(idx+1)*batch_size]
                x_batch = x_tr[batch_idx]
                y_batch = y_tr[batch_idx]

                pred = model(x_batch)

                loss = F.cross_entropy(pred, y_batch, reduction="none")
                sum_loss = torch.sum(loss)

                avg_loss = torch.mean(loss)

                num_all_train += len(x_batch)
                optimizer.zero_grad()
                avg_loss.backward()
                optimizer.step()

                total_loss = total_loss + sum_loss.detach()
            
        else:
            print("temperature:", T)
            # execute learning from the designed curriculum
            for idx in range(num_all_batch):
                # sample the batch
                batch_idx = all_tr_idx[idx*batch_size:(idx+1)*batch_size]

                # get the influence score
                phi_batch = phi_indiv[batch_idx]
                
                # compute batch sampling prob and do sampling
                prob_batch = compute_prob_by_phi(phi_batch, T)
                
                sub_batch_idx = sampling_by_phi(batch_idx, phi_batch, T, ratio)

                # updating on the selected data
                x_batch = x_tr[sub_batch_idx]
                y_batch = y_tr[sub_batch_idx]

                pred = model(x_batch)
                loss = F.cross_entropy(pred, y_batch, reduction="none")
                sum_loss = torch.sum(loss)

                avg_loss = torch.mean(loss)
                num_all_train += len(x_batch)

                optimizer.zero_grad()
                avg_loss.backward()
                optimizer.step()

                total_loss = total_loss + sum_loss.detach()

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

        print("computing individual influence scores.")
        model.eval()

        # ---------------------------------------------------------------
        # compute s_va, g_va, which are same for all groups of training data
        # ---------------------------------------------------------------
        va_loss_avg = evaluate_va_loss(model, x_va, y_va)
        g_va = list(grad(va_loss_avg, theta, create_graph=True))
        grad_val = [g.detach() for g in g_va]

        s_va = cal_inverse_hvp_lissa(model, x_tr, y_tr, grad_val, theta,
            batch_size=512,
            damp=0.01,
            scale=25.0,
            recursion_depth=100,)
        
        # ---------------------------------------------------------------
        # start to compute influence score for each sample
        # ---------------------------------------------------------------
        all_tr_idx = np.arange(len(x_tr))
        phi_indiv = compute_individual_influence(
            model, x_tr, y_tr, s_va)

        # first sampling 25%

        print("individual influence computation done.")


if __name__ == '__main__':
    import fire
    fire.Fire()