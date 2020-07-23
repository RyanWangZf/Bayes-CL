# -*- coding: utf-8 -*-
"""Implementation of mentornet for curriculum learning [1].

[1] Jiang, L., Zhou, Z., Leung, T., Li, L. J., & Fei-Fei, L. (2018, July). Mentornet: Learning data-driven curriculum for very deep neural networks on corrupted labels. In International Conference on Machine Learning (pp. 2304-2313).
"""

import numpy as np
import pdb, os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from dataset import load_data

from utils import setup_seed, img_preprocess, impose_label_noise
from model import BNN
from utils import save_model, load_model
from utils import predict, eval_metric, eval_metric_binary
from utils import adjust_learning_rate
from config import opt

class mentorNet(nn.Module):
    def __init__(self, num_class, label_embedding_size=2, epoch_embedding_size=5, num_fc_nodes=20):
        """
        Args:
            label_embedding_size: the embedding size for the label feature.
            epoch_embedding_size: the embedding size for the epoch feature.
            num_fc_nodes: number of hidden nodes in the fc layer.
        """
        super(mentorNet, self).__init__()
        # initialize label and epoch embedding size
        self.label_embed = nn.Embedding(num_class, label_embedding_size)
        self.epoch_embed = nn.Embedding(100, epoch_embedding_size,)

        # bi-lstm
        self.lstm = nn.LSTM(input_size=2, hidden_size=1, bias=False, bidirectional=True)

        # fc layers
        self.fc1 = nn.Linear(11, num_fc_nodes)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(num_fc_nodes, 1)


    def forward(self, input_features):
        """Receive training features, training epoch percentage, and labels,
        output a weight for student training.
        Args:
            input_features: a [batch_size, 4] tensor. Each dimension corresponds to
            0: loss, 1: loss difference to the moving average, 2: label and 3: epoch,
            where epoch is an integer between 0 and 99 (the first and the last epoch).
        Returns:
            v: [batch_size, 1] weight vector.
        """
        # decode the input features
        loss, loss_diff, label, epoch_percent = input_features
        # epoch_percent_list = torch.repeat_interleave(epoch_percent, label.shape[0])
        
        # get label & epoch embeddings
        label_inputs = self.label_embed(label) # ?, 2
        epoch_inputs = self.epoch_embed(epoch_percent) # ?, 5

        # do bi-lstm for loss sequence inference
        lstm_inputs = torch.cat([loss.unsqueeze(1), loss_diff.unsqueeze(1)], axis=1)

        _,  out_state_lstm = self.lstm(lstm_inputs.unsqueeze(0))
        out_state_fw, out_state_bw = out_state_lstm # 2, ?, 1

        lstm_feat = torch.cat([out_state_fw.squeeze(2), out_state_bw.squeeze(2)], axis=0) # 2, 256

        feat = torch.cat([label_inputs, epoch_inputs, lstm_feat.T], axis=1) # ?, 11

        z = self.fc1(feat)
        z = self.tanh(z)
        z = self.fc2(z)
        out = torch.sigmoid(z)
        return out

def dump_student_feature(model,
    sub_idx,
    noise_index,
    x_tr, y_tr, 
    x_va, y_va, 
    num_epoch,
    batch_size,
    lr, 
    weight_decay,
    early_stop_ckpt_path,
    early_stop_tolerance=3,
    noise_ratio=0.2,
    ):
    """Learn a studentnet on corrupted label,
    then dump the feature for mentornet learning.
    Args:
        model: student model
    """

    # early stop
    best_va_acc = 0
    num_all_train = 0
    early_stop_counter = 0

    # num class
    num_class = torch.unique(y_va).shape[0]

    # train student model and dump features for mentornet learning
    _, best_epoch, _ = train(model, 
            sub_idx, 
            x_tr, y_tr, 
            x_va, y_va, 
            num_epoch,
            batch_size,
            lr,
            weight_decay,
            early_stop_ckpt_path,
            early_stop_tolerance)
    
    epoch_percent = np.ones(len(sub_idx)) * int(100 * best_epoch / num_epoch)
    corrupted_list = np.zeros(len(sub_idx))
    corrupted_list[noise_index] = 1 # tag the corrupted sample indices

    # forward studentnet for feature
    num_all_tr_batch = int(np.ceil(len(sub_idx) / batch_size))

    model.eval()

    loss_list = []
    for idx in range(num_all_tr_batch):
        batch_idx = sub_idx[idx*batch_size:(idx+1)*batch_size]
        x_batch = x_tr[batch_idx]
        y_batch = y_tr[batch_idx]

        with torch.no_grad():
            pred = model(x_batch)

        if num_class > 2:
            loss = F.cross_entropy(pred, y_batch,
                reduction="none")
        else:
            loss = F.binary_cross_entropy(pred[:,0], y_batch.float(), 
                reduction="none")
        
        loss_list.extend(loss.cpu().detach().numpy().tolist())

    # concat three features: sample_idx, epoch_percent, corrupted_tag, label, loss 
    all_stu_feat = np.stack([sub_idx, epoch_percent, corrupted_list, y_tr.cpu().detach().numpy().tolist(), loss_list], 1)

    return all_stu_feat

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

    # log it for training mentornet
    moving_avg_loss_percentile = 0

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

            batch_mv_avg_loss_percentile = np.percentile(loss.cpu().detach().numpy(), 0.7)

            if moving_avg_loss_percentile == 0:
                moving_avg_loss_percentile = batch_mv_avg_loss_percentile
            else:
                # exponential moving average
                moving_avg_loss_percentile = 0.9 * moving_avg_loss_percentile + 0.1 * batch_mv_avg_loss_percentile

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
            best_epoch = epoch

        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stop_tolerance:
            print("early stop on epoch {}, val acc {}".format(epoch, best_va_acc))
            # load model from the best checkpoint
            load_model(early_stop_ckpt_path, model)
            best_epoch = epoch - early_stop_tolerance
            break

    return best_va_acc, best_epoch, moving_avg_loss_percentile

def train_student(model, mentor,
    sub_idx,
    x_tr, y_tr,
    x_va, y_va,
    num_epoch,
    batch_size,
    lr,
    weight_decay,
    early_stop_ckpt_path,
    early_stop_tolerance,
    burn_in_epoch = 10,
    ):
    print("train studentnet.")

    mentor.eval()

    # early stop setup
    best_va_acc = 0
    early_stop_counter = 0

    # num class
    num_class = torch.unique(y_va).shape[0]
    num_all_tr_batch = int(np.ceil(len(sub_idx) / batch_size))

    # define weight variable for all training samples
    num_all_train = len(sub_idx)

    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)

    moving_avg_loss_percentile = 0
    adjust_learning_rate_flag = False
    for epoch in range(num_epoch):
        model.train()
        np.random.shuffle(sub_idx)

        epoch_percent = np.ones(len(sub_idx)) * int(100 * epoch / num_epoch)
        epoch_percent = torch.LongTensor(epoch_percent).to(x_tr.device)

        if epoch > (burn_in_epoch + 10) and not adjust_learning_rate_flag:
            print("adjust learning rate")
            adjust_learning_rate(optimizer, lr * 0.1)
            adjust_learning_rate_flag = True

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
            
            if epoch > burn_in_epoch:
                # burn-in process, dont use mentornet in first 20 epochs
                # output mentorNet weight
                batch_mv_avg_loss_percentile = np.percentile(loss.cpu().detach().numpy(), 0.75)

                if moving_avg_loss_percentile == 0:
                    moving_avg_loss_percentile = batch_mv_avg_loss_percentile
                else:
                    # exponential moving average
                    moving_avg_loss_percentile = 0.9 * moving_avg_loss_percentile + 0.1 * batch_mv_avg_loss_percentile

                batch_loss_diff = loss - moving_avg_loss_percentile
                batch_epoch_percent = epoch_percent[batch_idx]

                v_batch = mentor((loss, batch_loss_diff, y_batch, batch_epoch_percent))
                v_batch = v_batch.detach() # stop gradient

                # g_penalty = torch.sum(0.5 * 0.9 * v_batch ** 2 - (0.9 + 0.1) * v_batch)
                # g_penalty_noneg = torch.clamp(g_penalty, min=0) # non negative penalty term
                g_penalty_noneg = torch.mean(v_batch)

                weighted_loss = torch.sum(v_batch[:,0] * loss)

                all_loss = 1/len(batch_idx) * (weighted_loss + g_penalty_noneg)
                
                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()

            else:
                avg_loss = torch.mean(loss)
                optimizer.zero_grad()
                avg_loss.backward()
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


def main_train_mentornet(**kwargs):
    """Call for training the mentornet.
    """
    # pre-setup
    setup_seed(opt.seed)
    opt.parse(kwargs)
    log_dir = os.path.join(opt.result_dir, "mentornet_" + opt.model + "_"  + opt.data_name + "_" + opt.print_opt)
    ckpt_dir = os.path.join(os.path.join(log_dir, "ckpt"))
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # intermediate file for early stopping
    early_stop_ckpt_path = os.path.join(ckpt_dir, "best_va.pth")
    mentornet_path = os.path.join(log_dir, "mentornet.pth")

    # load data & preprocess
    x_tr, y_tr, x_va, y_va, x_te, y_te = load_data(opt.data_name)
    all_tr_idx = np.arange(len(x_tr))
    num_class = np.unique(y_va).shape[0]

    if opt.noise_ratio > 0:
        print("put noise in label, ratio = ", opt.noise_ratio)
        y_tr, noise_index = impose_label_noise(y_tr, opt.noise_ratio, return_noise_index=True)
    
    x_tr, y_tr = img_preprocess(x_tr, y_tr, opt.use_gpu)
    x_va, y_va = img_preprocess(x_va, y_va, opt.use_gpu)
    x_te, y_te = img_preprocess(x_te, y_te, opt.use_gpu)
    print("load data done")

    # dump student feature
    model = BNN(num_class=num_class)
    if opt.use_gpu:
        model.cuda()
    
    stu_feat_name = os.path.join(log_dir, "stu_feat.npy")

    if os.path.exists(stu_feat_name):
        stu_feat = np.load(stu_feat_name)
    else:
        stu_feat = dump_student_feature(model,
                all_tr_idx,
                noise_index,
                x_tr, y_tr,
                x_va, y_va,
                opt.num_epoch,
                opt.batch_size,
                opt.lr,
                opt.weight_decay,
                early_stop_ckpt_path,
                5)

        np.save(stu_feat_name, stu_feat)

    # train mentor net
    mentor = mentorNet(num_class)
    if opt.use_gpu:
        mentor.cuda()

    optm_mentor = optim.Adam(filter(lambda p: p.requires_grad, mentor.parameters()), 
        lr=1e-3, weight_decay=0)

    all_tr_idx = np.arange(len(stu_feat))
    batch_size = 32
    num_all_tr_batch = int(np.ceil(len(all_tr_idx) / batch_size))

    loss_p_percentile = np.percentile(stu_feat[:,-1], 70)
    loss_diff = stu_feat[:, -1] - loss_p_percentile

    loss_diff_feat = torch.FloatTensor(loss_diff).to(x_tr.device)
    loss_feat = torch.FloatTensor(stu_feat[:, -1]).to(x_tr.device)
    corrup_feat = torch.LongTensor(stu_feat[:,2]).to(x_tr.device)
    label_feat = torch.LongTensor(stu_feat[:, 3]).to(x_tr.device)
    epoch_feat = torch.LongTensor(stu_feat[:,1]).to(x_tr.device)

    for epoch in range(10):
        total_loss = 0
        mentor.train()
        np.random.shuffle(all_tr_idx)
        for idx in range(num_all_tr_batch):
            batch_idx = all_tr_idx[idx*batch_size:(idx+1)*batch_size]

            # get feat
            batch_loss_feat = loss_feat[batch_idx]
            batch_loss_diff = loss_diff_feat[batch_idx]
            batch_label_feat = label_feat[batch_idx]
            batch_epoch_feat = epoch_feat[batch_idx]

            # get target
            batch_corrup_feat = corrup_feat[batch_idx]

            # forward mentornet
            pred = mentor((batch_loss_feat, batch_loss_diff, batch_label_feat, batch_epoch_feat))
            
            # do xent loss
            loss = F.binary_cross_entropy(pred[:,0], batch_corrup_feat.float(), reduction="mean")
            
            optm_mentor.zero_grad()
            loss.backward()
            optm_mentor.step()

            total_loss = total_loss + loss.detach()

        print("train mentor, epoch: {}, loss: {}".format(epoch, total_loss))
    
    # save mentor model
    save_model(mentornet_path, mentor)

def main(**kwargs):
    """With the pretrained mentornet, train out student net.
    """
    # pre-setup
    setup_seed(opt.seed)
    opt.parse(kwargs)
    log_dir = os.path.join(opt.result_dir, "mentornet_" + opt.model + "_"  + opt.data_name + "_" + opt.print_opt)
    ckpt_dir = os.path.join(os.path.join(log_dir, "ckpt"))
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # intermediate file for early stopping
    early_stop_ckpt_path = os.path.join(ckpt_dir, "best_va.pth")
    mentornet_path = os.path.join(log_dir, "mentornet.pth")


    # load data & preprocess
    x_tr, y_tr, x_va, y_va, x_te, y_te = load_data(opt.data_name)
    all_tr_idx = np.arange(len(x_tr))
    num_class = np.unique(y_va).shape[0]

    if opt.noise_ratio > 0:
        print("put noise in label, ratio = ", opt.noise_ratio)
        y_tr, noise_index = impose_label_noise(y_tr, opt.noise_ratio, return_noise_index=True)
    
    x_tr, y_tr = img_preprocess(x_tr, y_tr, opt.use_gpu)
    x_va, y_va = img_preprocess(x_va, y_va, opt.use_gpu)
    x_te, y_te = img_preprocess(x_te, y_te, opt.use_gpu)
    print("load data done")

    # load mentornet
    mentor = mentorNet(num_class)
    if opt.use_gpu:
        mentor.cuda()
    load_model(mentornet_path, mentor)
    print("load mentor net done.")

    # init model
    model = BNN(num_class=num_class)
    if opt.use_gpu:
        model.cuda()

    # training
    best_va_acc = train_student(model, mentor, all_tr_idx, x_tr, y_tr, x_va, y_va, opt.num_epoch,
        opt.batch_size, opt.lr, opt.weight_decay, early_stop_ckpt_path, 5)
    
    # evaluate model on test set
    pred_te = predict(model, x_te)
    if num_class > 2:
        acc_te = eval_metric(pred_te, y_te)
    else:
        acc_te = eval_metric_binary(pred_te, y_te)

    print("curriculum: {}, acc: {}".format(0, acc_te.item()))
    return

if __name__ == "__main__":
    import fire
    fire.Fire()


