# -*- coding: utf-8 -*-
"""Implementation of curriculumNet [1].
[1] Guo, S., Huang, W., Zhang, H., Zhuang, C., Dong, D., Scott, M. R., & Huang, D. (2018). Curriculumnet: Weakly supervised learning from large-scale web images. In Proceedings of the European Conference on Computer Vision (ECCV) (pp. 135-150).
"""

import time
import numpy as np
import pdb, os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from model import TextCNN, BertMLP
from utils import train, predict, eval_metric, eval_metric_binary
from utils import setup_seed, impose_label_noise, text_preprocess
from utils import save_model, load_model
from config import opt
from dataset import load_data

from utils import adjust_learning_rate
from config import opt

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.utils import check_array, check_consistent_length, gen_batches


def main(**kwargs):
    # pre-setup
    setup_seed(opt.seed)
    opt.parse(kwargs)
    log_dir = os.path.join(opt.result_dir, "clNet_" + opt.model + "_"  + opt.data_name + "_" + opt.print_opt)
    print("output log dir", log_dir)

    if not torch.cuda.is_available():
        print("[WARNING] do not find cuda device, change use_gpu=False!")
        opt.use_gpu = False

    ckpt_dir = os.path.join(os.path.join(log_dir, "ckpt"))
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # intermediate file for early stopping
    early_stop_ckpt_path = os.path.join(ckpt_dir, "best_va.pth")
    bert_ckpt_path = os.path.join(ckpt_dir, "bertmlp.pth")
    cluster_label_path = os.path.join(log_dir, "cluster_label.npy")

    # load data & preprocess
    x_tr, y_tr, x_va, y_va, x_te, y_te, vocab_size = load_data(opt.data_name, mode="onehot")
    x_tr_b, _, x_va_b, _, x_te_b, _, vocab_size = load_data(opt.data_name, mode="bert")
    
    all_tr_idx = np.arange(len(x_tr))
    num_class = np.unique(y_tr).shape[0]

    if opt.noise_ratio > 0:
        print("put noise in label, ratio = ", opt.noise_ratio)
        y_tr = impose_label_noise(y_tr, opt.noise_ratio)
    
    x_tr, y_tr = text_preprocess(x_tr, y_tr, opt.use_gpu)
    x_va, y_va = text_preprocess(x_va, y_va, opt.use_gpu)
    x_te, y_te = text_preprocess(x_te, y_te, opt.use_gpu)
    x_tr_b = text_preprocess(x_tr_b, None, opt.use_gpu, "bert")
    x_va_b = text_preprocess(x_va_b, None, opt.use_gpu, "bert")
    x_te_b = text_preprocess(x_te_b, None, opt.use_gpu, "bert")
    print("load data done")

    bert_mlp = BertMLP(num_class, x_tr_b.shape[1])
    if opt.use_gpu:
        bert_mlp.cuda()

    # finetune the bertmlp on this dataset
    if os.path.exists(bert_ckpt_path):
        print("load from finetuned resnet50 model.")
        load_model(bert_ckpt_path, bert_mlp)
    else:
        all_tr_idx = np.arange(len(x_tr))
        _ = train(bert_mlp, all_tr_idx, x_tr_b, y_tr, x_va_b, y_va,
            50, opt.batch_size, opt.lr, opt.weight_decay, bert_ckpt_path, 5)

    # extract bert mlp feature for clustering
    if os.path.exists(cluster_label_path):
        print("load precomputed cluster labels.")
        all_clustered_labels = np.load(cluster_label_path)
    
    else:
        # extract features and do clustering
        x_tr_feat = extract_model_feature(bert_mlp, x_tr_b)
        print("hidden feature extracted.")
        curriculum_clf = curriculumClustering(n_subsets=3, verbose=True)
        all_clustered_labels = curriculum_clf.fit_predict(x_tr_feat, y_tr.cpu().numpy())
        np.save(cluster_label_path, all_clustered_labels)
        print("cluster label computed.")

    # start curriculum learning for model
    model = TextCNN(vocab_size, num_class)
    if opt.use_gpu:
        model.cuda()
    model.use_gpu = opt.use_gpu

    # one_step_pacing
    num_simple_data = int(0.2 * len(x_tr))
    all_tr_idx = np.arange(len(x_tr))
    simple_sub_idx = all_tr_idx[all_clustered_labels == 0]
    simple_sub_idx = np.random.choice(simple_sub_idx, num_simple_data, replace=False)

    te_acc_list = []

    # training on simple set
    va_acc = train(model,
            simple_sub_idx,
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


def extract_model_feature(bert_mlp, x_tr):
    """Extract penultimate output features from resnet by doing inference.
    """
    bert_mlp.eval()
    all_tr_idx = np.arange(len(x_tr))
    batch_size = 256
    num_all_tr_batch = int(np.ceil(len(all_tr_idx) / batch_size))

    x_tr_feat = []
    for idx in range(num_all_tr_batch):
        batch_idx = all_tr_idx[idx*batch_size:(idx+1)*batch_size]
        x_batch = x_tr[batch_idx]
        with torch.no_grad():
            x_ft = bert_mlp.encoder(x_batch)
        x_tr_feat.append(x_ft)
    x_tr_feat = torch.cat(x_tr_feat).cpu().numpy()
    return x_tr_feat

#########################################################
# curriculum clustering toolbox
#########################################################

class curriculumClustering(BaseEstimator, ClusterMixin):
    """Perform Curriculum Clustering from vector array or distance matrix.
    The algorithm will cluster *each* category into N subsets using distribution density in an unsupervised manner.
    The subsets can be thought of stages in an educational curriculum, going from easiest to hardest learning material.
    For information, please see see CurriculumNet, a weakly supervised learning approach that leverages this technique.
    Parameters
    ----------
    X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
            array of shape (n_samples, n_samples)
        A feature array (the embedding space of the samples.)
    y : array-like, size=[n_samples]
        category labels (the curriculum will be learned for each of these categories into N subsets)
    verbose : bool, optional, default False
        Whether to print progress messages to stdout.
    density_t : float, optional
        The density threshold for neighbors to be clustered together
    n_subsets : int, optional (default = 3)
        The number of subsets to cluster each category into. For example, if set to 3, then the categories
        outputted will be assigned a label 0, 1, or 2. Where 0 contains the simplest (most similar) samples,
        1 contains middle level (somewhat similar samples), and 2 contains most complex (most diverse) samples.
    random_state : int, RandomState instance or None (default), optional
        The generator used to make random selections within the algorithm. Use an int to make the
        randomness deterministic.
    method : {'default', 'gaussian'}, optional
        The algorithm to be used to calculate local density values. The default algorithm
        follows the approach outlined in the scientific paper referenced below.
    dim_reduce : int, optional (default = 256)
        The dimensionality to reduce the feature vector to, prior to calculating distance.
        Lower dimension is more efficient, but degrades performance, and visa-versa.
    batch_max : int, optional (default = 500000)
        The maximum batch of feature vectors to process at one time (loaded into memory).
    calc_auxiliary : bool, optional (default = False)
        Provide auxiliary including delta centers and density centers.
        This can be useful information for visualization, and debugging, amongst other use-cases.
        The downside is the processing time significantly increases if turned on.
    Returns
    -------
    all_clustered_labels : array [n_samples]
        Clustered labels for each point. Labels are integers ordered from most simple to most complex.
        E.g. if curriculum subsets=3, then label=0 is simplest, labels=1 is harder, and label=n is hardest.
    auxiliary_info : list
        If calc_auxiliary is set to True, this list contains collected auxiliary information
        during the clustering process, including delta centers, which can be useful for visualization.
    References
    ----------
    S. Guo, W. Huang, H. Zhang, C. Zhuang, D. Dong, M. R. Scott, D. Huang,
    "CurriculumNet: Weakly Supervised Learning from Large-Scale Web Images".
    In: Proceedings of the European Conference on Computer Vision (ECCV),
    Munich, Germany, 2018. (arXiv:1808.01097)
    """

    def __init__(self, n_subsets=3, method='default', density_t=0.6, verbose=False,
                 dim_reduce=256, batch_max=500000, random_state=None, calc_auxiliary=False):
        self.n_subsets = n_subsets
        self.method = method
        self.density_t = density_t
        self.verbose = verbose
        self.output_labels = None
        self.random_state = random_state
        self.dim_reduce = dim_reduce
        self.batch_max = batch_max
        self.calc_auxiliary = calc_auxiliary

    def fit(self, X, y):
        """Perform curriculum clustering.
        Parameters
        ----------
        X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
                array of shape (n_samples, n_samples)
            A feature array (the embedding space of the samples.)
        y : array-like, size=[n_samples]
            category labels (the curriculum will be learned for each of these categories into N subsets)
        """
        X = check_array(X, accept_sparse='csr')
        check_consistent_length(X, y)
        self.output_labels, _ = self.cluster_curriculum_subsets(X, y, **self.get_params())
        return self

    def fit_predict(self, X, y=None):
        """Performs curriculum clustering on X and returns clustered labels (subsets).
        Parameters
        ----------
        X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
                array of shape (n_samples, n_samples)
            A feature array (the embedding space of the samples.)
        y : array-like, size=[n_samples]
            category labels (the curriculum will be learned for each of these categories into N subsets)
        Returns
        -------
        all_clustered_labels : array [n_samples]
            Clustered labels for each point. Labels are integers ordered from most simple to most complex.
            E.g. if curriculum subsets=3, then label=0 is simplest, labels=1 is harder, and label=n is hardest.
        auxiliary_info : list
            If calc_auxiliary is set to True, this list contains collected auxiliary information
            during the clustering process, including delta centers, which can be useful for visualization.
        """
        self.fit(X, y)
        return self.output_labels

    def cluster_curriculum_subsets(self, X, y, n_subsets=3, method="default", density_t=0.6,
        verbose=False, dim_reduce=256, batch_max=500000, random_state=None, calc_auxiliary=False):
        """Perform Curriculum Clustering from vector array or distance matrix.
        The algorithm will cluster each category into n subsets by analyzing distribution density.
        Parameters
        ----------
        X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
                array of shape (n_samples, n_samples)
            A feature array (the embedding space of the samples.)
        y : array-like, size=[n_samples]
            category labels (the curriculum will be learned for each of these categories into N subsets)
        verbose : bool, optional, default False
            Whether to print progress messages to stdout.
        density_t : float, optional
            The density threshold for neighbors to be clustered together into a subset
        n_subsets : int, optional (default = 3)
            The number of subsets to cluster each category into. For example, if set to 3, then the categories
            outputted will be assigned a label 0, 1, or 2. Where 0 contains the simplest (most similar) samples,
            1 contains middle level (somewhat similar samples), and 2 contains most complex (most diverse) samples.
        random_state : int, RandomState instance or None (default), optional
            The generator used to make random selections within the algorithm. Use an int to make the
            randomness deterministic.
        method : {'default', 'gaussian'}, optional
            The algorithm to be used to calculate local density values. The default algorithm
            follows the approach outlined in the scientific paper referenced below.
        dim_reduce : int, optional (default = 256)
            The dimensionality to reduce the feature vector to, prior to calculating distance.
            Lower dimension is more efficient, but degrades performance, and visa-versa.
        batch_max : int, optional (default = 500000)
            The maximum batch of feature vectors to process at one time (loaded into memory).
        calc_auxiliary : bool, optional (default = False)
            Provide auxiliary including delta centers and density centers.
            This can be useful information for visualization, and debugging, amongst other use-cases.
            The downside is the processing time significantly increases if turned on.
        Returns
        -------
        all_clustered_labels : array [n_samples]
            Clustered labels for each point. Labels are integers ordered from most simple to most complex.
            E.g. if curriculum subsets=3, then label=0 is simplest, labels=1 is harder, and label=n is hardest.
        auxiliary_info : list
            If calc_auxiliary is set to True, this list contains collected auxiliary information
            during the clustering process, including delta centers, which can be useful for visualization.
        References
        ----------
        S. Guo, W. Huang, H. Zhang, C. Zhuang, D. Dong, M. R. Scott, D. Huang,
        "CurriculumNet: Weakly Supervised Learning from Large-Scale Web Images".
        In: Proceedings of the European Conference on Computer Vision (ECCV),
        Munich, Germany, 2018. (arXiv:1808.01097)
        """
        if not density_t > 0.0:
            raise ValueError("density_thresh must be positive.")
        X = check_array(X, accept_sparse='csr')
        check_consistent_length(X, y)

        unique_categories = set(y)
        t0 = None
        pca = None
        auxiliary_info = []
        if X.shape[1] > dim_reduce:
            pca = PCA(n_components=dim_reduce, copy=False, random_state=random_state)

        # Initialize all labels as negative one which represents un-clustered 'noise'.
        # Post-condition: after clustering, there should be no negatives in the label output.
        all_clustered_labels = np.full(len(y), -1, dtype=np.intp)

        for cluster_idx, current_category in enumerate(unique_categories):
            print("cluster idx {}, category idx {}".format(cluster_idx, current_category))
            if verbose:
                t0 = time.time()

            # Collect the "learning material" for this particular category
            dist_list = [i for i, label in enumerate(y) if label == current_category]

            for batch_range in gen_batches(len(dist_list), batch_size=batch_max):
                batch_dist_list = dist_list[batch_range]

                # Load data subset
                subset_vectors = np.zeros((len(batch_dist_list), X.shape[1]), dtype=np.float32)
                for subset_idx, global_idx in enumerate(batch_dist_list):
                    subset_vectors[subset_idx, :] = X[global_idx, :]

                # Calc distances
                if pca:
                    subset_vectors = pca.fit_transform(subset_vectors)
                m = np.dot(subset_vectors, np.transpose(subset_vectors))
                t = np.square(subset_vectors).sum(axis=1)
                distance = np.sqrt(np.abs(-2 * m + t + np.transpose(np.array([t]))))

                # Calc densities
                if method == 'gaussian':
                    densities = np.zeros((len(subset_vectors)), dtype=np.float32)
                    distance = distance / np.max(distance)
                    for i in range(len(subset_vectors)):
                        densities[i] = np.sum(1 / np.sqrt(2 * np.pi) * np.exp((-1) * np.power(distance[i], 2) / 2.0))
                else:
                    densities = np.zeros((len(subset_vectors)), dtype=np.float32)
                    flat_distance = distance.reshape(distance.shape[0] * distance.shape[1])
                    dist_cutoff = np.sort(flat_distance)[int(distance.shape[0] * distance.shape[1] * density_t)]
                    for i in range(len(batch_dist_list)):
                        densities[i] = len(np.where(distance[i] < dist_cutoff)[0]) - 1  # remove itself
                if len(densities) < n_subsets:
                    raise ValueError("Cannot cluster into {} subsets due to lack of density diversification,"
                                    " please try a smaller n_subset number.".format(n_subsets))

                # Optionally, calc auxiliary info
                if calc_auxiliary:
                    # Calculate deltas
                    deltas = np.zeros((len(subset_vectors)), dtype=np.float32)
                    densities_sort_idx = np.argsort(densities)
                    for i in range(len(densities_sort_idx) - 1):
                        larger = densities_sort_idx[i + 1:]
                        larger = larger[np.where(larger != densities_sort_idx[i])]  # remove itself
                        deltas[i] = np.min(distance[densities_sort_idx[i], larger])

                    # Find the centers and package
                    center_id = np.argmax(densities)
                    center_delta = np.max(distance[np.argmax(densities)])
                    center_density = densities[center_id]
                    auxiliary_info.append((center_id, center_delta, center_density))

                model = KMeans(n_clusters=n_subsets, random_state=random_state)
                model.fit(densities.reshape(len(densities), 1))
                clusters = [densities[np.where(model.labels_ == i)] for i in range(n_subsets)]
                n_clusters_made = len(set([k for j in clusters for k in j]))
                if n_clusters_made < n_subsets:
                    raise ValueError("Cannot cluster into {} subsets, please try a smaller n_subset number, such as {}.".
                                    format(n_subsets, n_clusters_made))

                cluster_mins = [np.min(c) for c in clusters]
                bound = np.sort(np.array(cluster_mins))

                # Distribute into curriculum subsets, and package into global adjusted returnable array, optionally aux too
                other_bounds = range(n_subsets - 1)
                for i in range(len(densities)):

                    # Check if the most 'clean'
                    if densities[i] >= bound[n_subsets - 1]:
                        all_clustered_labels[batch_dist_list[i]] = 0
                    # Else, check the others
                    else:
                        for j in other_bounds:
                            if bound[j] <= densities[i] < bound[j + 1]:
                                all_clustered_labels[batch_dist_list[i]] = len(bound) - j - 1

            if verbose:
                print("Clustering {} of {} categories into {} curriculum subsets ({:.2f} secs).".format(
                    cluster_idx + 1, len(unique_categories), n_subsets, time.time() - t0))

        if (all_clustered_labels > 0).all():
            raise ValueError("A clustering error occurred: incomplete labels detected.")

        return all_clustered_labels, auxiliary_info

if __name__ == "__main__":
    import fire
    fire.Fire()