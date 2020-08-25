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

from dataset import load_data

from utils import setup_seed, img_preprocess, impose_label_noise
from model import LeNet
from utils import save_model, load_model
from utils import predict, eval_metric, eval_metric_binary
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
    log_dir = os.path.join(opt.result_dir, "cltl_" + opt.model + "_"  + opt.data_name + "_" + opt.print_opt)
    print("output log dir", log_dir)

    ckpt_dir = os.path.join(os.path.join(log_dir, "ckpt"))
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # intermediate file for early stopping
    early_stop_ckpt_path = os.path.join(ckpt_dir, "best_va.pth")
    resnet50_ckpt_path = os.path.join(ckpt_dir, "resnet.pth")
    early_stop_stu_path = os.path.join(ckpt_dir, "best_va_stu_{}.pth".format(opt.bnn))

    output_result_path = os.path.join(log_dir, "cltl.result")

    # load data & preprocess
    x_tr, y_tr, x_va, y_va, x_te, y_te = load_data(opt.data_name)
    all_tr_idx = np.arange(len(x_tr))
    num_class = np.unique(y_va).shape[0]

    if opt.noise_ratio > 0:
        print("put noise in label, ratio = ", opt.noise_ratio)
        y_tr = impose_label_noise(y_tr, opt.noise_ratio)
    
    x_tr, y_tr = img_preprocess(x_tr, y_tr, opt.use_gpu)
    x_va, y_va = img_preprocess(x_va, y_va, opt.use_gpu)
    x_te, y_te = img_preprocess(x_te, y_te, opt.use_gpu)
    print("load data done")

    import torchvision.models as models
    resnet50 = models.resnet50(pretrained = True)
    resnet50.fc = nn.Linear(resnet50.fc.in_features, num_class, bias=True)

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

    if os.path.exists(os.path.join(log_dir, "tl_score.npy")):
        print("load precomputed difficulty scores.")
        score_list = np.load(os.path.join(log_dir, "tl_score.npy"))
    else:
        score_list = compute_tl_score(resnet50, x_tr, y_tr)
        np.save(os.path.join(log_dir,"tl_score.npy"), score_list)

    # load model
    _, in_channel, in_size, _ = x_tr.shape
    model = LeNet(num_class=num_class, in_size=in_size, in_channel=in_channel)
    if opt.use_gpu:
        model.cuda()
    model.use_gpu = opt.use_gpu

    # design difficulty by uncertainty difficulty
    curriculum_idx_list = one_step_pacing(y_tr, score_list, num_class, 0.2)

    te_acc_list = []

    # training on simple set
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
    te_acc_list.append(acc_te.item())
    print(te_acc_list)

    res_list = [str(_) for _ in [first_stage_acc.item(), acc_te.item()]]
    with open(output_result_path, "w") as f:
        f.write("\n".join(res_list) + "\n")

def compute_tl_score(large_net, x_tr, y_tr,):
    """Receive a large pretrained net, e.g., resnet50, get its penultimate features,
    for training a classifier, e.g., rbf-svm, to get the margin as the difficulty scores.
    """
    print("start computing transfer learning difficulty scores.")
    from sklearn import svm
    from sklearn.preprocessing import MinMaxScaler

    clf = svm.LinearSVC(verbose=True, max_iter=100)

    def hook(module, input, output):
        """return inputs of the last fc layer.
        """
        # print("hooker working.")
        return input
    
    # processing the x_tr to features with dim 2048, for training the rbf-svm.
    handle = large_net.fc.register_forward_hook(hook)
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
            x_ft = large_net(x_batch)
        x_tr_feat.append(x_ft[0])

    handle.remove()

    # fit a svm-rbf to get the scores
    x_tr_feat = torch.cat(x_tr_feat).cpu().numpy()
    y_tr_cpu = y_tr.cpu().numpy()
    scaler = MinMaxScaler()

    x_tr_feat = scaler.fit_transform(x_tr_feat)

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