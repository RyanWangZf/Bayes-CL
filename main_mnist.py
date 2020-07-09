import numpy as np
import pdb, os

import torch
import torch.nn.functional as F
from torch import nn

from torch.autograd import grad

def load_mnist(validation_size = 5000):
    import gzip
    def _read32(bytestream):
        dt = np.dtype(np.uint32).newbyteorder(">")
        return np.frombuffer(bytestream.read(4),dtype=dt)[0]

    def extract_images(f):
        print("Extracting",f.name)
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = _read32(bytestream)
            num_images = _read32(bytestream)
            rows = _read32(bytestream)
            cols = _read32(bytestream)
            buf = bytestream.read(rows * cols * num_images)
            data = np.frombuffer(buf,dtype=np.uint8)
            data = data.reshape(num_images,rows,cols,1)
            return data
    
    def extract_labels(f):
        print('Extracting', f.name)
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = _read32(bytestream)
            num_items = _read32(bytestream)
            buf = bytestream.read(num_items)
            labels = np.frombuffer(buf, dtype=np.uint8)
            return labels

    data_dir = "./data"
    TRAIN_IMAGES = os.path.join(data_dir,'train-images-idx3-ubyte.gz')
    with open(TRAIN_IMAGES,"rb") as f:
        train_images = extract_images(f)

    TRAIN_LABELS =  os.path.join(data_dir,'train-labels-idx1-ubyte.gz')
    with open(TRAIN_LABELS,"rb") as f:
        train_labels = extract_labels(f)

    TEST_IMAGES =  os.path.join(data_dir,'t10k-images-idx3-ubyte.gz')
    with open(TEST_IMAGES,"rb") as f:
        test_images = extract_images(f)

    TEST_LABELS =  os.path.join(data_dir,'t10k-labels-idx1-ubyte.gz')
    with open(TEST_LABELS,"rb") as f:
        test_labels = extract_labels(f)

    # split train and val
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    return train_images, train_labels, train_images[:validation_size], train_labels[:validation_size],\
         test_images, test_labels

def get_model_param_dict(model):
    params = {}
    for name,param in model.named_parameters():
        params[name] = param

    return params


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()

    def forward(self,input):
        return input.view(input.size(0),-1)

class SimpleCNN(nn.Module):
    def __init__(self, num_class=10):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d((2,2))

        self.conv2 = nn.Conv2d(16,8, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(8)

        # name: "dense.weight", "dense.bias"
        self.dense = nn.Linear(3*3*8, num_class) # 
        self.flatten = Flatten()
        self.softmax = nn.functional.softmax

        self.early_stopper = None

    def forward(self, inputs, hidden=False):
        is_training = self.training

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.flatten(x)
        out = self.dense(x)
        out = self.softmax(out, dim=1)

        if hidden:
            return out, x
        else:
            return out

    def predict(self, x, batch_size=1000, onehot=False):
        self.eval()
        total_batch = compute_num_batch(x,batch_size)
        pred_result = []

        for idx in range(total_batch):
            batch_x = x[idx*batch_size:(idx+1)*batch_size]
            pred = self.forward(batch_x)
            if onehot:
                pred_result.append(pred)
            else:
                pred_result.append(torch.argmax(pred,1))

        preds = torch.cat(pred_result)
        return preds

def one_hot_transform(y, num_class=10):
    one_hot_y = nn.functional.one_hot(y, num_classes=num_class)
    return one_hot_y.float()

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

def cal_inverse_hvp_lissa(model, 
    x_tr, 
    y_tr, 
    x_va, 
    y_va,
    theta,
    damp=0.01,
    scale=25.0,
    recursion_depth=100,
    tol=1e-5,
    verbose=False,
    ):
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

    # get grad theta on validation set
    batch_size = 256
    va_loss = None
    num_all_batch = int(np.ceil(len(x_va)/batch_size))
    for idx in range(num_all_batch):
        pred = model(x_va[idx*batch_size:(idx+1)*batch_size])
        if va_loss is None:
            va_loss = F.cross_entropy(pred, y_va[idx*batch_size:(idx+1)*batch_size],
                reduction="sum")
        else:
            va_loss = va_loss + F.cross_entropy(pred, y_va[idx*batch_size:(idx+1)*batch_size],
                reduction="sum")

    va_loss_avg = va_loss / len(x_va)
    g_va = list(grad(va_loss_avg, theta, create_graph=True))
    grad_val = [g.detach() for g in g_va]

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

    return h_estimate, g_va

if __name__ == '__main__':
    # load data
    x_tr, y_tr, x_va, y_va, x_te, y_te = load_mnist(5000)

    x_tr = np.transpose(x_tr, (0,3,1,2))
    x_va = np.transpose(x_va, (0,3,1,2))
    x_te = np.transpose(x_te, (0,3,1,2))

    model = SimpleCNN(10)

    x_ts = torch.Tensor(x_tr / 255.0)
    y_ts = torch.LongTensor(y_tr)

    x_va_ts = torch.Tensor(x_va / 255.0)
    y_va_ts = torch.LongTensor(y_va)

    x_te_ts = torch.Tensor(x_te / 255.0)
    y_te_ts = torch.LongTensor(y_te)

    model_param_list = get_model_param_dict(model)
    theta = [model_param_list["dense.weight"], model_param_list["dense.bias"]]

    # ---------------------------------------------------------------
    # compute s_va, g_va, which are same for all groups of training data
    # ---------------------------------------------------------------
    s_va, g_va = cal_inverse_hvp_lissa(model, x_ts, y_ts, x_va_ts, y_va_ts, theta, verbose=True)

    # ---------------------------------------------------------------
    # compute g_u and s_u = s_va * H(u), which are unique for each group
    # ---------------------------------------------------------------

    # compute g_u
    x_u, y_u = x_ts[:10], y_ts[:10]
    pred = model(x_u)
    loss_u = F.cross_entropy(pred, y_u, reduction="mean")
    g_u = list(grad(loss_u, theta, create_graph=True))

    # compute s_u = s_va * H(u)
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
    S = len(x_tr)
    U = len(x_u)
    p = S / U
    c1 = p/(1-p)
    c2 = -1/(1-p) * U/S

    phi = 0
    for i in range(len(theta)):
        t1 = torch.sum(s_va[i] * g_u[i])
        # compute hessian inverse vector product
        t2 = torch.sum(s_u_h_inv[i] * g_u[i])
        print(t1, t2)

        phi_ = c2 * t1 + c1 * c2 * (t1 - t2)
        phi = phi + phi_

    print("phi:", phi)
    print("Almost done.")