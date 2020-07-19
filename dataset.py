# -*- coding: utf-8 -*-
import pickle
import os
import numpy as np
import pdb


np.random.seed(2020)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data(data_name, class_list = None):
    if data_name == "cifar10":
        print("load from CIFAR-10.")
        return load_cifar10(class_list = class_list)

    if data_name == "cifar100":
        return

    if data_name == "":
        return

def select_from_one_class(x_tr, y_tr, select_class=0):
    all_idx = np.arange(len(x_tr))
    class_idx = all_idx[y_tr == select_class]
    return x_tr[class_idx], y_tr[class_idx]

def load_cifar10(dir="./data/cifar-10-python", class_list=None):
    val_ratio = 0.1

    # load training data
    tr_fnames = ["data_batch_"+str(i+1) for i in range(5)]
    te_fname = "test_batch"

    tr_fpath = [os.path.join(dir, _) for _ in tr_fnames]
    tr_batch_raw = [unpickle(path) for path in tr_fpath]
    te_fpath = os.path.join(dir, te_fname)

    features, labels = [], []
    for raw in tr_batch_raw:
        data = raw[b"data"]
        data_ = np.reshape(data, (-1, 3, 32, 32))
        features.append(data_)
        label = raw[b"labels"]
        labels.extend(label)

    features = np.concatenate(features)
    labels = np.array(labels)

    # select from one class
    if class_list is not None:
        feat_list, label_list = [], []
        for c in class_list:
            tr_feat, tr_label = select_from_one_class(features, labels, c)
            feat_list.append(tr_feat)
            label_list.append(tr_label)
        features = np.concatenate(feat_list)
        labels = np.concatenate(label_list)

    # split tr and va set
    val_size = int(val_ratio * len(features))
    all_idx = np.arange(len(labels))
    np.random.shuffle(all_idx)
    tr_features = features[all_idx[val_size:]]
    tr_labels = labels[all_idx[val_size:]]
    va_features = features[all_idx[:val_size]]
    va_labels = labels[all_idx[:val_size]]

    meta_fname = os.path.join(dir, "batches.meta")
    meta_data = unpickle(meta_fname)

    # load test data
    te_raw = unpickle(te_fpath)
    te_data, te_labels = te_raw[b"data"], te_raw[b"labels"]
    te_features = np.reshape(te_data, (-1, 3, 32, 32))
    te_labels = np.array(te_labels)
    
    # select from one class
    if class_list is not None:
        feat_list, label_list = [], []
        for c in class_list:
            tr_feat, tr_label = select_from_one_class(te_features, te_labels, c)
            feat_list.append(tr_feat)
            label_list.append(tr_label)
        te_features = np.concatenate(feat_list)
        te_labels = np.concatenate(label_list)

    return tr_features, tr_labels, va_features, va_labels, te_features, te_labels

def load_cifar100(dir="./data/cifar-100-python"):
    # TODO

    return


if __name__ == '__main__':
    x_tr, y_tr, x_va, y_va, x_te, y_te = load_cifar100()
    pdb.set_trace()
    pass

