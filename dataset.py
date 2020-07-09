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

def load_data(data_name):
    if data_name == "cifar10":
        print("load from CIFAR-10.")
        return load_cifar10()

    if data_name == "cifar100":
        return

    if data_name == "":
        return


def load_cifar10(dir="./data/cifar-10-python"):
    val_size = 5000

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

    # split tr and va set
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

    return tr_features, tr_labels, va_features, va_labels, te_features, te_labels

if __name__ == '__main__':
    x_tr, y_tr, x_va, y_va, x_te, y_te = load_cifar10()
    pdb.set_trace()
    pass

