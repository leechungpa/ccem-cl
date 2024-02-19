# The original code is from "https://github.com/krafton-ai/mini-batch-cl"

import os
import random

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


class Args:
    def __init__(
            self,
            N=4, d=8,
            lr_full=0.5, num_steps=1000, logging_step=1000,
            loss="s", loss_t=1, loss_b=1,
            verbos=False, use_3_layor=False,
            device="cuda"
            ):
        self.N = N
        self.d = d
        self.lr_full = lr_full
        self.num_steps = num_steps
        self.logging_step = logging_step
        self.loss = loss
        self.loss_t = loss_t
        self.loss_b = loss_b
        self.verbos = verbos
        self.device = device
        self.use_3_layor = use_3_layor
        
        if verbos:
            print("N", self.N)
            print("d", self.d)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print("Seeded everything: {}".format(seed))


def generate_gaussian_data(args):
    mean = np.zeros(args.d*2)
    cov = np.eye(args.d*2)

    x1 = np.random.multivariate_normal(mean, cov, args.N)
    x2 = np.random.multivariate_normal(mean, cov, args.N)

    if args.device == 'cuda':
        return torch.cuda.FloatTensor(x1), torch.cuda.FloatTensor(x2)
    else:
        return torch.FloatTensor(x1), torch.FloatTensor(x2)


def info_loss(u, v, loss_t):
    n = u.shape[0]
    logits = torch.exp(u @ v.T / loss_t)
    loss = -torch.log(logits/torch.sum(logits, dim=1)).diagonal(dim1=0).sum()
    return loss/n

def cl_loss(u, v, args):
    if args.loss == "i": # InfoNCE Loss (args.loss_t)
        return info_loss(u, v, args.loss_t) + info_loss(v, u, args.loss_t)
    elif args.loss == "s": # Sigmoid Loss (args.loss_t, args.loss_b)
        n = u.shape[0]
        mask = torch.ones(n,n, device=args.device) - 2*torch.eye(n, device=args.device)
        loss = torch.log(1 + torch.exp(mask*(u@v.T*args.loss_t -args.loss_b))).sum()
        return loss/n
    else:
        raise ValueError("loss must be i or s")


class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc1 = nn.Linear(args.d*2, args.d, bias=True)
        self.fc2 = nn.Linear(args.d, args.d, bias=True)
        self.fc3 = nn.Linear(args.d, args.d, bias=True)

        self.use_3_layor = args.use_3_layor

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.use_3_layor:
            x = F.relu(self.fc3(x))
        return x


def train(**kwargs):
    args = Args(**kwargs)

    x_u, x_v = generate_gaussian_data(args)

    model = MLP(args)

    if args.device == 'cuda':
        model.to('cuda:0')

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_full)

    for step in tqdm(range(args.num_steps), disable=~args.verbos):
        optimizer.zero_grad()
        u = model(x_u)
        v = model(x_v)
        u, v = F.normalize(u, p=2, dim=1), F.normalize(v, p=2, dim=1)
        loss = cl_loss(u, v, args)
        loss.backward()
        optimizer.step()

        if args.verbos:
            if (step % args.logging_step == 0) or (step == args.num_steps-1):
                with torch.no_grad(), torch.cuda.amp.autocast():
                    u,v = F.normalize(u, p=2, dim=1), F.normalize(v, p=2, dim=1)
                print(loss.item())

    z = (u @ v.T).detach().cpu()
    return z


if __name__ == '__main__':
    pass