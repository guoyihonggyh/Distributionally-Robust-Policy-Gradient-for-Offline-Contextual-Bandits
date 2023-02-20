from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import os
import copy
import warnings
warnings.filterwarnings('ignore')

mean0 = 0
var0 = 1
n_class = 10
d = 1

def predict_minimax_gaussian(Myy, Myx, output):
    
    bs = np.shape(output)[0]
    x_1 = torch.cat((output, torch.ones((bs, 1))), 1)
    Myy = torch.tensor(Myy)
    Myx = torch.tensor(Myx)


    meanY = -(1.0/Myy)*((output.matmul(Myx).t()))
    
    varY = -1.0/(2.0*Myy)

    # meanY = torch.max(meanY, torch.zeros(meanY.shape))

    return meanY, varY

def predict_regression(weight, Myy, Myx, output, mean0, var0):
    '''
    # mean0 and var0: batch * 1 vector
    # output: batch * d
    # Myy: 1*1
    # Myx: d+1, 1
    # weight: batch * 1
    '''
    bs = np.shape(output)[0]
    x_1 = torch.cat((output, torch.ones((bs, 1))), 1)
    Myy = torch.tensor(Myy)
    Myx = torch.tensor(Myx)


    # print(x_1.matmul(Myx).t())
    # print(weight)


    meanY = (1.0/(2.0*weight*Myy+(1/var0)))*(-2.0*(torch.addcmul(torch.zeros(1), x_1.matmul(Myx).t(), weight))+(1/var0)*mean0)
    
    varY = 1.0/(2.0 *weight*Myy + 1.0/var0)

    # meanY = torch.max(meanY, torch.zeros(meanY.shape))

    return meanY, varY

    
def prediction_regression(kde, data, model, Myy, Myx):
    model.eval()
    with torch.no_grad(): 
        weight = kde(data)
        weight = weight
        # weight = 1.0
        output = model(data)
        d = np.shape(data)[0]
        meanY, varY = predict_regression(torch.tensor(weight), Myy, Myx, output, mean0, var0)
    
    return meanY, varY

def prediction_naive(data):
    return mean0, var0



def Theta_yy_gradient(y, policy,meanY, varY, i, true_action):
    '''
    policy --size: batch * n_class

    y_vec -- size: batch * n_class


    '''
    # extract the i_th action probability
    policy_i = policy[:, i]
    one_hot_i = true_action[:, i]
    
    grad =  2 *(torch.mean(policy_i * meanY*varY) + torch.mean(torch.mean(meanY) - torch.mean(y)* meanY*varY*one_hot_i))/policy.shape[0]
    
    return grad

def Theta_yx_gradient(x, y, policy,meanY, varY, i, true_action):
    '''
    policy --size: batch * n_class

    x -- size batch * n_feature_dim

    y_vec -- size: batch * n_class


    '''

    # extract the i_th action probability
    policy_i = policy[:, i]
    one_hot_i = true_action[:, i]
    # print(np.shape(policy_i *varY))
    
    grad =  2 *(torch.einsum('i,ik->k' , policy_i *varY, x) + torch.einsum('i,ik->k', (torch.mean(meanY) - torch.mean(y))*varY*one_hot_i, x))/x.shape[0]
    return grad

def Omega_gradient(x, policy, meanY, true_action):
   
    temp = torch.einsum('ij,ij->ij', (policy - true_action), meanY)
    # print(policy - true_action)
    # print(temp.shape[0])
    grad =  torch.einsum('ij,ij->j', temp, x)/temp.shape[0]
    # print(grad)
    return grad



def M_gradient(x, meanY, varY, y, Myy, Myx):

    # print(x)
    bs = np.shape(x)[0]
    d = np.shape(x)[1]
    x = np.reshape(x.detach().numpy(), (bs, d))
    y = np.reshape(y.detach().numpy(), (bs, ))
    meanY = np.reshape(meanY.detach().numpy(), (bs, ))
    varY = np.reshape(varY.detach().numpy(), (bs, 1))

    y_vec = np.concatenate((np.reshape(y, (bs, 1)), x, np.ones((bs, 1))), axis = 1)

    emp = np.einsum('ij,i->ij', y_vec, y)
   
    # print(np.shape(emp))
    y_mean_vec = np.concatenate((np.reshape(meanY, (bs, 1)), x, np.ones((bs, 1))), 1)
    # y_var_vec = np.concatenate((varY, np.zeros((bs,d+1 ))), 1)
    var_vec = np.concatenate((varY, np.zeros((bs, d+1))), 1)
    exp = np.einsum('ij,i->ij', y_mean_vec, meanY) + var_vec
  
    grad =  np.mean(exp, 0) - np.mean(emp, 0)

    return np.reshape(grad, (d+2, 1))


class delta_gradient(torch.autograd.Function):
    '''
      back-prop gradients to network
      the gradient for "phi(x)" -- representing network -- is (y-mean_y)* Myx[0:-1]
    
    '''

    @staticmethod
    def forward(ctx, x, prob, one_hot, meanY, omega):
        ctx.save_for_backward(x, prob, one_hot, meanY, omega)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        
        x, prob, one_hot, meanY, omega= ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_x = grad_x  * meanY *(prob - one_hot)*omega/grad_x.shape[0]
        # print("mean", meanY)
        # print("omage", omega)
        # print(grad_x)
        return grad_x, None, None, None, None

class Myx_gradient(torch.autograd.Function):
    '''
      back-prop gradients to network
      the gradient for "phi(x)" -- representing network -- is (y-mean_y)* Myx[0:-1]
    
    '''

    @staticmethod
    def forward(ctx, x, policy, varY, meanY, Myx, target, one_hot):
        
        ctx.save_for_backward(x, policy, varY, meanY, Myx, target, one_hot)
        return x

    @staticmethod
    def backward(ctx, grad_output): 
        x, policy, varY, meanY, Myx, target, one_hot = ctx.saved_tensors
        
        
        grad_x = grad_output.clone()


        grad_x = grad_x  * 2 * torch.sum(policy *varY)* Myx + torch.sum((torch.mean(meanY) - torch.mean(target))*varY*one_hot)*Myx/grad_x.shape[0]
        
        return grad_x, None, None, None,None, None, None,None


class regression_gradient(torch.autograd.Function):
    '''
      back-prop gradients to network
      the gradient for "phi(x)" -- representing network -- is (y-mean_y)* Myx[0:-1]
    
    '''

    @staticmethod
    def forward(ctx, x, M, y, meanY):
        ctx.save_for_backward(x, M, y, meanY)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        
        x, M, y, meanY= ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_x = grad_x  *(y - meanY)* M.reshape(-1, d)/grad_x.shape[0]
        
        return grad_x, None, None, None


class policy_gradient(torch.autograd.Function):
    '''
      back-prop gradients to network
      the gradient for "phi(x)" -- representing network -- is (y-mean_y)* Myx[0:-1]
    
    '''

    @staticmethod
    def forward(ctx, potential, prob, reward, action):
        ctx.save_for_backward(potential, prob, reward, action)
        return potential

    @staticmethod
    def backward(ctx, grad_output):
        
        potential, prob, reward, action= ctx.saved_tensors

        grad_x = grad_output.clone()

        grad_x = grad_x * torch.tensor(np.einsum('ij,i->ij', (prob - action), reward))

        return grad_x, None, None, None
