import torch
import numpy as np
import pdb
import torchvision.transforms

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def img_preprocess(x, y=None, use_gpu=False):
    x = torch.Tensor(x) / 255.0
    # x = x - 0.5 # [-0.5, 0.5]

    if use_gpu:
        x = x.cuda()

    if y is not None:
        y = torch.LongTensor(y)
        if use_gpu:
            y = y.cuda()

        return x, y

    else:
        return x

def impose_label_noise(y, noise_ratio, return_noise_index=False):
    max_y, min_y = y.max(), y.min()
    all_idx = np.arange(len(y))
    np.random.shuffle(all_idx)
    
    num_noise_sample = int(noise_ratio * len(y))
    noise_idx = all_idx[:num_noise_sample]    
    noise_y = np.random.randint(int(min_y), int(max_y) + 1, [num_noise_sample])
    y[noise_idx] = noise_y
    
    if return_noise_index:
        return y, noise_idx
    else:
        return y

def save_model(ckpt_path, model):
    torch.save(model.state_dict(), ckpt_path)
    return

def load_model(ckpt_path, model):
    model.load_state_dict(torch.load(ckpt_path))
    return

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

def eval_metric_binary(pred, y):
    pred_label = np.ones(len(pred))
    y_label = y.detach().cpu().numpy()
    pred_prob = pred.flatten().cpu().detach().numpy()
    pred_label[pred_prob < 0.5] = 0.0
    acc = torch.Tensor(y_label == pred_label).float().mean()
    return acc

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class ExponentialScheduler(object):
    def __init__(self, init_t, max_t):
        """Args:
            init_t: initial value
            max_t: max value at last
        """
        self.max_t = max_t
        self.init_t = init_t

    def step(self, t):
        return np.minimum(self.max_t, (1.05) ** (t) * self.init_t)

class ExponentialDecayScheduler(object):
    def __init__(self, init_t, min_t):
        """Args:
            init_t: initial value
            max_t: max value at last
        """
        self.min_t = min_t
        self.init_t = init_t

    def step(self, t):
        return np.maximum(self.min_t, (0.95) ** (t) * self.init_t)



class LinearScheduler(object):
    def __init__(self, init_t, max_t, num_epoch):
        self.init_t = init_t
        self.max_t = max_t
        self.num_epoch = num_epoch

    def step(self, t):
        return self.init_t + (self.max_t - self.init_t) / self.num_epoch * t