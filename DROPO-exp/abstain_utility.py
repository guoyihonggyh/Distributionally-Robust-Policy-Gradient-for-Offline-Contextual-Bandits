from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import os
import copy


class log_gradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, prob, y_onehot):
        """
        In the forward pass we receive a context object and a Tensor containing the
        input; we must return a Tensor containing the output, and we can use the
        context object to cache objects for use in the backward pass.
        """
        ctx.save_for_backward(x, prob, y_onehot)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive the context object and a Tensor containing
        the gradient of the loss with respect to the output produced during the
        forward pass. We can retrieve cached data from the context object, and must
        compute and return the gradient of the loss with respect to the input to the
        forward function.
        """
        x, prob, y_onehot = ctx.saved_tensors
        # grad_x = grad_output.clone()
        # print((y_onehot - prob))
        grad_x = grad_output.clone()

        grad_x = grad_x * (prob - y_onehot)/grad_x.shape[0]
        # print(grad_x)
        return grad_x, None, None


def my_softmax(x):
    n = np.shape(x)[0]
    max_x, _ = torch.max(x, dim=1)
    max_x = torch.reshape(max_x, (n, 1))
    exp_x = torch.exp(x - max_x)
    p = exp_x / torch.reshape(torch.sum(exp_x, dim=1), (n, 1))
    p[p<10e-8] = 0
    return p

def predict_log_abstain(x, rho, n_class, target, alpha):
    n = np.shape(x)[0]
    pred = torch.zeros(n, dtype = torch.long)
    loss = 0

    max_v, max_indx = torch.topk(x, 2)
    for i in range(n):
        
        if max_v[i, 0] - max_v[i, 1] > rho:
            pred[i] = max_indx[i, 0]
        else:
            pred[i] = n_class

        if pred[i] != target[i] and pred[i]!=n_class:
            loss += 1
        elif pred[i]==n_class:
            loss += alpha

 
    return pred, loss


def predict_abstain(x, alpha, n_class, target):

    nc = n_class
    m = np.shape(x)[0]

    abs_loss = np.zeros(m)
    loss = 0
    r = np.zeros((m, nc))
    s = np.zeros((m, nc + 1))
    for j in range(m):
        m1 = -10e8
        m2 = -10e8
        id1 = 0
        id2 = 0
        for i in range(nc):
            if x[j,i] > m1:               # if tie choose the first occurance
                # move the current to 2nd
                m2 = m1
                id2 = id1
                # 1st best
                m1 = x[j, i]
                id1 = i
            elif x[j, i] > m2:          # if tie choose the first occurance
                # 2nd
                m2 = x[j, i]
                id2 = i
    
        l1 = m1
        l2 = m1 * (1-alpha) + m2 * alpha + alpha

        if l2 >= l1:                    # if tie choose 2 element 
            abs_loss[j] = l2
            # if id1==0:
                
            r[j, id1] = (1-alpha)
            r[j, id2] = alpha

            s[j, id1] = m1 - m2
            s[j, nc] = 1 - s[j, id1]
        else:
            abs_loss[j] = l1
            r[j, id1] = 1.0
            s[j, id1] = 1.0

    max_indx = np.argmax(s, axis = 1)
    
    for j in range(m):

        if max_indx[j] != target[j] and max_indx[j]!=n_class:
            loss += 1
        elif max_indx[j]==n_class:
            loss += alpha


    return loss, abs_loss, r, s