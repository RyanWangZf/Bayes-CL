# -*- coding: utf-8 -*-
import os, pdb
import numpy as np
np.random.seed(2020)
from collections import deque
from itertools import islice

from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

# max word settings for different dataset
MAX_WORD_DICT = {
    "20ng":230,
    "R8":70,
    "R52": 70,
    "ohsumed":140,
    "mr":30,
}

"""
1. Use fastText (average embedding + MLP) as the base student model.
2. Use BERT + MLP as the transfer model.
"""
def fill_zero(sen, max_word):
    length = len(sen)
    if length >= max_word:
        # clip
        return sen[:max_word]
    
    else:
        # fill
        sen = np.concatenate((sen, [0] * (max_word - length)))
        return sen

def load_data(name="20ng", data_dir = "./data", mode="onehot"):
    assert name in ["20ng", "R8", "ohsumed", "R52", "mr"]
    print("load dataset:", name)
    # SETUPS
    max_word = MAX_WORD_DICT[name]
    val_ratio = 0.1
    te_ratio = 0.3

    corpus_dir = os.path.join(data_dir, "corpus")
    target_dir = os.path.join(data_dir, "{}_processed".format(name))
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    assert mode in ["onehot", "bert"]
    # target filename
    onehot_filename = os.path.join(target_dir, "{}_onehot.npy".format(name))
    bert_filename = os.path.join(target_dir, "{}_bert.npy".format(name))
    label_ar_filename = os.path.join(target_dir, "{}_label.npy".format(name))
    idx_filename = os.path.join(target_dir, "{}_idx.npy".format(name))

    # raw filename
    feat_filename = "{}.clean.txt".format(name)
    feat_filename = os.path.join(corpus_dir, feat_filename)
    label_filename = os.path.join(data_dir, "{}.txt".format(name))

    if os.path.exists(idx_filename):
        idx_map = np.load(idx_filename, allow_pickle=True).item()
        tr_idx = idx_map["tr"]
        te_idx = idx_map["te"]
        va_idx = idx_map["va"]
    
    else:
        print("split tr va te")
        fin = open(label_filename, "r", encoding="utf-8")
        y_lines = fin.readlines()
        fin.close()
        all_idx = np.arange(len(y_lines))
        np.random.shuffle(all_idx)
        te_size, va_size = int(len(y_lines)*te_ratio), int(len(y_lines)*val_ratio)
        te_idx = all_idx[:te_size]
        va_idx = all_idx[te_size:te_size+va_size]
        tr_idx = all_idx[te_size+va_size:]
        np.save(idx_filename, {"tr":tr_idx, "te":te_idx, "va":va_idx})

    # build vocaburary list
    with open(os.path.join(corpus_dir, "{}_vocab.txt".format(name))) as f:
        vocab_list = f.readlines()
    vocab_dict = {vocab_list[k].strip():k for k in range(len(vocab_list))}
    vocab_size = len(vocab_dict.keys())

    if mode == "onehot":
        if os.path.exists(onehot_filename):
            print("load preprocessed features.")
            # load pre-processed features
            x_feat_ar = np.load(onehot_filename)
            y_label_ar = np.load(label_ar_filename)
        else:            
            with open(feat_filename, "r", encoding="utf-8") as fin:
                x_content = fin.readlines()
            
            vectorizer = CountVectorizer(vocabulary=vocab_dict)
            X = vectorizer.fit_transform(x_content) # a sparse csr matrix
            x_feat = []
            for i in range(len(x_content)):
                _, indices = X[i].nonzero()
                # fill zero, plus one means 0 is specifically for NULL word.
                indices = fill_zero(indices+1, max_word)
                x_feat.append(indices)

            x_feat_ar = np.array(x_feat)
            np.save(onehot_filename, x_feat_ar)

            fin = open(label_filename, "r", encoding="utf-8")
            y_lines = fin.readlines()
            y_lines = [_.strip().split()[-1] for _ in y_lines]
            fin.close()

            # get mapped labels
            classes = np.sort(np.unique(y_lines))
            class_dict = {classes[i]:i for i in range(len(classes))}
            map_func = np.vectorize(lambda x: class_dict[x])
            y_label_ar = map_func(y_lines)
            np.save(label_ar_filename, y_label_ar)
            print("done preprocessing.")
    
    elif mode == "bert":
        # transformers:BERT to extract embeddings
        from transformers import BertTokenizer, BertModel

        if os.path.exists(bert_filename):
            print("load preprocessed bert features.")
            # load pre-processed features
            x_feat_ar = np.load(bert_filename)
            y_label_ar = np.load(label_ar_filename)
        else:
            import torch
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            max_token_size = 512
            stride = 128
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
            model.eval()
            model.to(device)

            with open(feat_filename, "r", encoding="utf-8") as fin:
                x_content = fin.readlines()
            
            feat_list = []
            for i, text in enumerate(x_content):
                if i % 100 == 0:
                    print("line:", i)
                # print("line:", i)

                # split text into sentences
                # marked_text = "[CLS]" + "[SEP]".join(text.strip().split("\\"))
                marked_text = "[CLS] " + text + " [SEP]"
                tokenized_text = tokenizer.tokenize(marked_text)
                if len(tokenized_text) > max_token_size:
                    sen_embedding = []
                    start_offset = 0
                    while start_offset < len(tokenized_text):
                        length = len(tokenized_text) - start_offset
                        if length > max_token_size:
                            length = max_token_size
                        tokens = tokenized_text[start_offset:start_offset+length]
                        indexed_tokens = tokenizer.convert_tokens_to_ids(list(tokens))
                        segments_ids = [1] * len(indexed_tokens)
                        tokens_tensor = torch.LongTensor([indexed_tokens]).to(device)
                        segments_tensors = torch.LongTensor([segments_ids]).to(device)
                        with torch.no_grad():
                            outputs = model(tokens_tensor, segments_tensors)

                        hidden_states = outputs[2]
                        token_vecs = hidden_states[-2][0] # # words, 768
                        sen_embedding.append(torch.mean(token_vecs, dim=0).unsqueeze(0)) # 768
                        if start_offset + length == len(tokenized_text):
                            break
                        start_offset += min(length, stride)
                    
                    feat_list.append(torch.cat(sen_embedding,0).mean(0).unsqueeze(0))

                else:
                    indexed_tokens = tokenizer.convert_tokens_to_ids(list(tokenized_text))
                    segments_ids = [1] * len(indexed_tokens)
                    tokens_tensor = torch.LongTensor([indexed_tokens]).to(device)
                    segments_tensors = torch.LongTensor([segments_ids]).to(device)
                    with torch.no_grad():
                        outputs = model(tokens_tensor, segments_tensors)

                    hidden_states = outputs[2]
                    token_vecs = hidden_states[-2][0] # # words, 768
                    sen_embedding = torch.mean(token_vecs, dim=0) # 768
            
                    feat_list.append(sen_embedding.unsqueeze(0))

            feat_ts = torch.cat(feat_list, 0)
            x_feat_ar = feat_ts.cpu().numpy()
            np.save(bert_filename, x_feat_ar)

            fin = open(label_filename, "r", encoding="utf-8")
            y_lines = fin.readlines()
            y_lines = [_.strip().split()[-1] for _ in y_lines]
            fin.close()

            # get mapped labels
            classes = np.sort(np.unique(y_lines))
            class_dict = {classes[i]:i for i in range(len(classes))}
            map_func = np.vectorize(lambda x: class_dict[x])
            y_label_ar = map_func(y_lines)
            np.save(label_ar_filename, y_label_ar)
            print("done preprocessing.")

    x_tr, x_va, x_te = x_feat_ar[tr_idx], x_feat_ar[va_idx], x_feat_ar[te_idx]
    y_tr, y_va, y_te = y_label_ar[tr_idx], y_label_ar[va_idx], y_label_ar[te_idx]

    return x_tr, y_tr, x_va, y_va, x_te, y_te, vocab_size

def get_id2word_func(data_name="mr"):

    vocab_path = os.path.join("./data/corpus/","{}_vocab.txt".format(data_name))
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_list = [x.strip() for x in f.readlines()]
    vocab_list = ["\n"] + vocab_list
    id2word_func = np.vectorize(lambda x: vocab_list[x])
    return id2word_func

if __name__ == "__main__":
    x_tr, y_tr, x_va, y_va, x_te, y_te, vocab_size = load_data(name="R52", mode="onehot")
    pdb.set_trace()
    pass