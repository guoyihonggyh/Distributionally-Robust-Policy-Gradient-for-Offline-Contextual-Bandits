from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data_set import WEIGHT_DATA_SET
from data import DATA, DATA_defined_prob, DATA_policy_gradient, DATA_fullaction, DATA_action, DATA_partial_logistic, DATA_partial_logistic_deep, DATA_partial_random, DATA_partial_action, DATA_learn_policy
from torchvision import transforms
import torchvision
import os
import copy
import torch.utils.data as data
from data_mnist import DATA_mnist
from data_cifar import DATA_CIFAR10 
import regression_utility as ru
import abstain_utility as au
from scipy import stats
import math
from scipy.stats import dirichlet
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import random
torch.set_default_tensor_type('torch.DoubleTensor')
mean0 = 0.6
var0 = 1
d = 1

class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.D_in = D_in
        self.H = H
        self.D_out = D_out
        self.model = torch.nn.Sequential(
            torch.nn.utils.spectral_norm(torch.nn.Linear(self.D_in, self.H)),
            torch.nn.ReLU(),
            torch.nn.utils.spectral_norm(torch.nn.Linear(self.H, self.H)),
            torch.nn.ReLU(),
            torch.nn.utils.spectral_norm(torch.nn.Linear(self.H, self.H)),
            torch.nn.ReLU(),
            torch.nn.utils.spectral_norm(torch.nn.Linear(self.H, self.D_out)),
            )

    def forward(self, x):

        x = x.view(-1, self.D_in)
        x = self.model(x.double())
        return x

def my_softmax(x):
    n = np.shape(x)[0]
    max_x, _ = torch.max(x, dim=1)
    max_x = torch.reshape(max_x, (n, 1))
    exp_x = torch.exp(x - max_x)
    p = exp_x / torch.reshape(torch.sum(exp_x, dim=1), (n, 1))
    p[p<10e-8] = 0
    return p


def train_regression(args, model, device, train_loader, optimizer, epoch, Myy, Myx, mean0, var0):
     
    model.train()
    lowerB = -1.0/(2*var0)
   
    grad_yy = np.empty([0])
    # change this if d changes
    grad_yx = np.empty([2, 0])

    lr2 = 1
    lr2 = lr2 * (10 / (10 + np.sqrt(epoch)))

    lr1 = 1
    lr1 = lr1 * (10 / (10 + np.sqrt(epoch)))
    grad_squ_1 = 0.00001
    grad_squ_2 = 0.00001
    for batch_idx, (data, target, weight) in enumerate(train_loader):
        
        data, target = data.to(device), target.to(device)
      
        optimizer.zero_grad()
        output = model(data)


        meanY, varY = ru.predict_regression(weight, Myy, Myx, output, mean0, var0)
        # print(varY)
        grad = ru.M_gradient(output, meanY, varY, target, Myy, Myx)

        grad_squ_1 = grad_squ_1 + grad[0]**2
        grad_squ_2 = grad_squ_2 + grad[1:]**2
        
        
        diff = lr1*(grad[0]) + 0.00000*Myy
        grad_yy = np.concatenate((grad_yy, grad[0]))
        grad_yx = np.concatenate((grad_yx, grad[1:]), 1)
        preM = Myy
        Myy = preM + lr1*(grad[0]/np.sqrt(grad_squ_1)) + 0.00000*Myy

        while Myy[0][0] < lowerB:
            Myy = Myy + np.abs(diff)/2

        Myx = Myx + lr2 *(grad[1:]/np.sqrt(grad_squ_2)) + 0.00000*Myx

        bs = np.shape(output)[0]
      
        output_last = ru.regression_gradient.apply(output, torch.tensor(Myx[0:-1]), torch.reshape(target, (bs, 1)), torch.reshape(meanY, (bs, 1)))
        output_last.backward(torch.ones(output_last.shape),retain_graph=True)
        

        optimizer.step()
    return Myy, Myx


def train_MSE(args, model, device, train_loader, optimizer, epoch):
    model.train()

    for _ in range(epoch):
        total_loss = 0
        for batch_idx, (data, target, weight) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)

            criterion = nn.MSELoss()
            # loss = args.mse_weight*criterion(output[target==1], target[target==1]) + (1-args.mse_weight)* criterion(output[target==0], target[target==0])
            loss = criterion(output, target)
            loss.backward()
            total_loss +=loss.detach()
            optimizer.step()
        for p in optimizer.param_groups:
            p['lr'] *= 0.9
    return model


def test_regression(args, model, Myy, Myx, device, test_loader, mean0, var0):
    model.eval()
    test_loss = 0
    y_prediction = np.empty([1, 0])
    y_var = np.empty([1, 0])
    with torch.no_grad():
        for data, target, weight in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            
            d = np.shape(data)[0]
            target = torch.reshape(target, (1, d))
            
            meanY, varY = ru.predict_regression(weight, Myy, Myx, output, mean0, var0)
            loss =  -np.log(1/(np.sqrt(varY)*np.sqrt(2*3.14)))+(target-meanY).pow(2)/(2*varY)
            criterion = nn.MSELoss()
            l2loss = criterion(meanY, target)
            test_loss += torch.sum(l2loss)
            y_prediction = np.concatenate((y_prediction, meanY), axis=1)
            y_var = np.concatenate((y_var, varY), axis = 1)
    test_loss /= len(test_loader.dataset)

    return y_prediction, y_var, test_loss



def train_validate_test(args, epoch, loss_type, device, use_cuda, train_model, train_loader, test_loader, validate_loader, n_class, lbd, testflag = True):
    
    if loss_type == 'regression':

        Myy = np.ones((1, 1))
        Myx = np.ones((d+1, 1))
        optimizer = optim.SGD(train_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=lbd)

        for epoch in range(1, epoch + 1):
            Myy, Myx = train_regression(args, train_model, device, train_loader, optimizer, epoch, Myy, Myx, mean0, var0) 
            meanY, varY, loss = test_regression(args, train_model, Myy, Myx, device, validate_loader, mean0, var0)
        
        if testflag == True:
            meanY, varY, loss = test_regression(args, train_model, Myy, Myx, device, test_loader, mean0, var0)
       
        return train_model, Myy, Myx, meanY, varY, loss
    
def round_value(a):
    if a>1:
        a = 1.0
    elif a<0:
        a = 0
    return a


def sample_action(prob, n_class):
    # sample action
    rand = np.random.uniform(0, 1, 1)
    action = 0
    threshold = 0
    
    for j in range(n_class):
    
        threshold = threshold + prob[j]
        
        if rand[0] < threshold:
            action = j
            break
    return action
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
         
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Covariate Shift')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs-training', type=int, default=40, metavar='N',
                        help='number of epochs in training (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_pg', type=float, default=0.001, metavar='LR',
                        help='learning rate for policy gradient (default: 0.001)')
    parser.add_argument('--filename', type=str, default='trainingdata.csv',
                        help='the file that contains training data')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--mode', type=int, default=1, metavar='N',
                        help='1, policy is a uniform default policy, 2 is a policy is a known policy trained from biased samples, 3 policy is an unknown policy uniform policy, 4 is an unknown biased policy ')
    parser.add_argument('--dataset', type=int, default=1, metavar='N',
                        help='1, uci, 2 mnist, 3 cifar10')
    parser.add_argument('--policy', type=int, default=0, metavar='N',
                        help='0, policy is a uniform default policy, 1 is a policy small shift policy, 2 policy is large policy, 3 is d uniform policy, 4 is tweak')
    parser.add_argument('--alpha', type=float, default=10, metavar='N',
                        help='direclect shift parameter')
    parser.add_argument('--rou', type=float, default=0.91, metavar='N',
                        help='tweak one shift parameter')
    parser.add_argument('--clip_weight', type=bool, default=False, metavar='N',
                        help='whether clip grad for robust regression')
    parser.add_argument('--epochs_policy_gradient', type=int, default=8, metavar='N',
                        help='epochs for policy gradient')
    parser.add_argument('--save-file-name', type=str, default='', metavar='N',
                        help='save regret filename')
    parser.add_argument('--lr-decay', type=float, default=0.9, metavar='N', help='dedcay rate of lr')
    parser.add_argument('--mse_weight', type=float, default=0.5, metavar='N', help='dedcay rate of lr')

    parser.add_argument('--simulation', type=int, default=5, metavar='N', help='interaction-num')
    parser.add_argument('--evaluate-batch', type=int, default=100, metavar='N', help='interaction-num')
    parser.add_argument('--train-mse', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=10, metavar='N', help='if use ips gradient')
    parser.add_argument('--set-seed', action='store_true', default=False)



    args = parser.parse_args()

    for simulation in range(args.simulation):
        if args.set_seed:
            setup_seed(args.seed + simulation)
        evaluate_flag = 0

        use_cuda = not args.no_cuda and torch.cuda.is_available()

        device = torch.device("cuda" if use_cuda else "cpu")

        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        if args.dataset == 3:

            all_data = DATA_CIFAR10(transform=transforms.Compose([
                            # transforms.RandomCrop(32, padding=4),
                            # transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]))
            n_class = 10
            n_dim = 32*32*3
        elif args.dataset == 2:
            all_data = DATA_mnist(args.filename)
            n_class = 10
            n_dim = 28*28
        elif args.dataset == 1:
            all_data = DATA(args.filename)
            n_dim = 64
            n_class = 10
        elif args.dataset == 4:
            train_data = DATA('data/covertype/train.csv')
            test_data = DATA('data/covertype/test.csv')
            m_train = len(train_data)
            m_test = len(test_data)


            rand_idx = np.random.permutation(m_train)
            train_data = data.Subset(train_data, rand_idx)

            n_dim = 54
            n_class = 7
        elif args.dataset == 5:
            all_data = DATA('adult_processed.csv')
            n_dim = 92
            n_class = 14

        n_nodes = 32
        policy_default = 1.0/n_class

        if args.dataset != 4:
            data_size = len(all_data)
            rand_idx = np.random.permutation(data_size)
            training_ratio = 0.6
            m_train = int(training_ratio*data_size)
            train_data = data.Subset(all_data, rand_idx[0: m_train])
            test_data = data.Subset(all_data, rand_idx[m_train: -1])
            m_test = data_size - m_train

        weight_st = np.ones(m_train)

        weighted_train = WEIGHT_DATA_SET(train_data, weight_st,args)


        weight_st = np.ones(m_test)
        weighted_test = WEIGHT_DATA_SET(test_data, weight_st,args)

        test_loader = data.DataLoader(weighted_test,
            batch_size=args.batch_size, shuffle=False, **kwargs)

        if args.policy == 1:
            prob_logging = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.46, 0.46]

        elif args.policy == 0:
            prob_logging = 1/n_class * np.ones(n_class)

        elif args.policy == 2:
            prob_logging = 0.01 * np.ones(n_class)
            prob_logging[-1] = 1 - 0.01 * (n_class - 1)
        elif args.policy == 3:
            if args.dataset == 5:
                if args.alpha == 1:
                    prob_logging = list(0.95 * np.array([6.44567468e-02, 1.52196220e-01, 1.36630291e-05, 4.30040770e-02,
                                                         1.89581051e-02, 1.15729274e-02, 2.46207099e-02, 5.06446425e-02,
                                                         6.03770831e-02, 9.24506849e-02, 6.49027231e-02, 1.38071608e-01,
                                                         2.73214821e-02, 2.51409327e-01])) + 0.05 * np.ones(14) / 14

                elif args.alpha == 0.1:
                    prob_logging = list(0.95 * np.array([4.22775824e-04, 1.01818000e-39, 1.23161752e-08, 1.33573819e-07,
                                                         2.56969191e-04, 4.45324241e-04, 3.39187131e-07, 6.31043231e-16,
                                                         4.25651585e-04, 7.90291778e-09, 2.88045068e-01, 2.43133827e-05,
                                                         7.10379404e-01, 5.25978522e-11]) + 0.05 * np.ones(14) / 14)

            else:
                if args.alpha == 0.25:
                    prob_logging = [0.0071749357348857516, 0.19454627471495628, 0.0009514408790603926,
                                    0.6293886731160462,
                                    0.0010388153588815011, 0.023119786448965396, 0.12066416986904867,
                                    0.00306732560246253,
                                    0.0001334710100312967, 0.019915107265661893]
                elif args.alpha == 0.5:
                    prob_logging = [0.15481991427596029, 0.2550228325964202, 0.01723563773125192, 0.07189458372634579,
                                    0.0356889192614843, 0.17166537031930112, 5.985327222609714e-05, 0.08455150628333191,
                                    0.056736328494532584, 0.1523250540391458]
                elif args.alpha == 1:
                    prob_logging = [0.05175540229331372, 0.09131980645552733, 0.23544978300069874, 0.08334005030799103,
                                    0.07433052705468039, 0.038457343721411644, 0.07410487306052077, 0.18744552101039014,
                                    0.04382682084849391, 0.11996987224697245]
                elif args.alpha == 0.1:
                    prob_logging = list(0.95 * np.array([1.28619732e-03, 3.61775298e-04, 4.10925937e-17, 1.54225408e-05,
                                                         1.67697250e-09, 6.34070241e-09, 3.66253847e-03, 6.58049540e-07,
                                                         6.59566673e-01, 3.35106727e-01]) + 0.005 * np.ones(10))


        elif args.policy == 4:
            prob_logging = np.ones((n_class))*(1-args.rou)/(n_class-1)
            prob_logging[n_class - 1]  = args.rou

        train_data_mse = DATA_defined_prob(train_data, n_class, n_dim, args,prob_logging)

        if args.mode is 1 or args.mode is 3:
            # train robust model
            # save models
            model_robust_list = []
            Myy_robust_list = []
            Myx_robust_list = []
            train_data_robust = train_data_mse
            print(len(train_data_robust))
            for i in range(n_class):
                ## generate training set for action i
                train_action_data = DATA_partial_action(train_data_robust, i)
                # train the regression model for predicting rewards
                train_size = len(train_action_data)
                weight_st = np.ones(train_size)

                weighted_train = WEIGHT_DATA_SET(train_action_data, weight_st,args)

                train_model = Net(n_dim,n_nodes, 1)
                train_model = train_model.to(device)

                validate_size = int(0.1*train_size)
                validate_size = 1 if validate_size else validate_size

                try:
                    train_loader = data.DataLoader(data.Subset(weighted_train, range(validate_size, train_size)),batch_size=args.batch_size, shuffle=True, **kwargs)
                    validate_loader = data.DataLoader(data.Subset(weighted_train, range(0, validate_size)),batch_size=args.batch_size, shuffle=True, **kwargs)
                    #
                    train_model, Myy, Myx, _, _, _ = train_validate_test(args, args.epochs_training, "regression", device, use_cuda, train_model,
                                                                         train_loader, test_loader , validate_loader, n_class, 0.000, testflag = False)
                    if not args.train_mse:
                        train_model, Myy, Myx, _, _, _ = train_validate_test(args, args.epochs_training, "regression",
                                                                             device, use_cuda, train_model,
                                                                             train_loader, test_loader, validate_loader,
                                                                             n_class, 0.000, testflag=False)

                    else:
                        optimizer1 = optim.Adam(train_model.parameters(), lr=args.lr)
                        train_model = train_MSE(args, train_model, device, train_loader, optimizer1,
                                                args.epochs_training)

                        total_loss = 0
                        for batch_idx0, (data0, target0, weight0) in enumerate(validate_loader):

                            output = train_model(data0)
                            criterion = nn.MSELoss()
                            loss = criterion(output, target0)
                            total_loss += loss.detach()

                except:
                    Myy = np.ones((1, 1))
                    Myx = np.ones((d + 1, 1))
                model_robust_list.append(train_model)

                if not args.train_mse:
                    Myy_robust_list.append(Myy)
                    Myx_robust_list.append(Myx)

        test_partial_data = train_data_mse
        # policy using policy model
        policy_learning_model = Net(n_dim,n_nodes, n_class)
        policy_learning_model = policy_learning_model.to(device)

        optimizer = optim.Adam(policy_learning_model.parameters(), lr=args.lr_pg, weight_decay=0)

        count = 0
        prob_action_pre = (1.0/n_class) *torch.ones((len(test_partial_data), n_class), dtype = torch.float64)
        action_pre = torch.ones(len(test_partial_data), dtype = torch.int64)
        regret_list = np.zeros([args.epochs_policy_gradient,int(len(test_partial_data)/args.batch_size)+ 1])
        test_regret_list = np.zeros([args.epochs_policy_gradient])
        loss_evaluate_result = []

        best_epoch = 1
        best_loss = 10000

        for epoch in range(1, args.epochs_policy_gradient + 1):
            regret = 0.0
            reward_training = torch.ones(len(test_partial_data), dtype = torch.float64)
            features_training = test_partial_data.get_features()
            action_training = test_partial_data.get_action()
            action_true = test_partial_data.get_action_true()
            reward_true = []
            total_loss = 0
            total_reward = 0
            total_reward_estimated = 0
            with torch.no_grad():
                for i in range(len(test_partial_data)):
                    features, action, reward, action_true_ = test_partial_data[i]
                    ##### get evaluation determinstic policy

                    action_policy = sample_action(prob_action_pre[i], n_class)
                    action_pre[i] = action_policy
                    if action_policy == action_true_:
                        true_reward = 1
                        reward_true.append(1)
                    else:
                        true_reward = 0
                        reward_true.append(0)
                    policy = prob_logging[action_policy]/prob_action_pre[i][action_policy]


                    if args.mode is 1 or args.mode is 3:

                        if not args.train_mse:
                            output = model_robust_list[action_policy](features)
                            meanY_robust, varY = ru.predict_regression(torch.tensor(1.0), Myy_robust_list[action_policy], Myx_robust_list[action_policy], output, mean0, var0)
                        else:
                            output = model_robust_list[action_policy](features)
                            meanY_robust = output
                            total_reward_estimated += meanY_robust
                            total_reward += reward
                            total_loss += (meanY_robust - true_reward) ** 2

                    if args.mode is 1:
                        reward_training[i] = meanY_robust

                    if args.mode is 2:
                        action_policy = action
                        action_pre[i] = action_policy
                        policy = prob_logging[action_policy] / prob_action_pre[i][action_policy]
                        reward_training[i] = (1.0/policy)*reward

                    if args.mode is 3:

                        if action_policy == action:
                            reward_training[i] = (1.0/policy)*(reward - meanY_robust) + meanY_robust
                        else:
                            reward_training[i] = meanY_robust

            policy_gradient_data = DATA_policy_gradient(features_training, reward_training, action_pre, action_true)
            # construct dataloader
            train_loader = data.DataLoader(policy_gradient_data,batch_size=args.batch_size, shuffle=True, **kwargs)
            optimizer.zero_grad()
            policy_learning_model.train()
            prob_action_next = torch.empty([0, n_class])
            for batch_idx, (feature_idx, action_idx, reward_idx, action_true_idx) in enumerate(train_loader):
                evaluate_flag += 1
                optimizer.zero_grad()
                bsize = len(feature_idx)
                output_learning = policy_learning_model(feature_idx)
                prob = my_softmax(output_learning)

                pi_mean = output_learning.max(1, keepdim=True)[1]

                action_onehot = torch.DoubleTensor(bsize,n_class)
                action_onehot.zero_()
                action_onehot.scatter_(1, action_idx.reshape(bsize, 1), 1)

                grad_policy =  ru.policy_gradient.apply(output_learning, prob, reward_idx, action_onehot)
                grad_policy.backward(torch.ones(grad_policy.shape),retain_graph=True)
                optimizer.step()

                prob = prob.detach().numpy()
                prob_action_next = np.concatenate((prob_action_next, prob), axis=0)

                correct = pi_mean.eq(action_true_idx.view_as(pi_mean)).sum().item()
                regret = float((bsize - correct)/bsize)
                regret_list[epoch-1][batch_idx] = regret
                if evaluate_flag == args.evaluate_batch:
                    evaluate_flag = 0
                    with torch.no_grad():
                        for feature, target, weight, in test_loader:
                            feature, target, weight = feature.to(device), target.to(device), weight.to(device)

                            output = policy_learning_model(feature)
                            y_prediction = output.max(1, keepdim=True)[1]

                            correct += y_prediction.eq(target.view_as(y_prediction)).sum().item()
                    test_loss = 1 - float(correct) / len(test_loader.dataset)
                    loss_evaluate_result.append(test_loss)

            prob_action_next = torch.empty([0, n_class])
            train_loader = data.DataLoader(policy_gradient_data,batch_size=args.batch_size*8, shuffle=False, **kwargs)
            for batch_idx, (feature_idx, action_idx, reward_idx, action_true_idx) in enumerate(train_loader):
                output_learning = policy_learning_model(feature_idx)
                prob = my_softmax(output_learning)
                prob = prob.detach().numpy()
                prob_action_next = np.concatenate((prob_action_next, prob), axis=0)
            prob_action_pre = torch.tensor(prob_action_next, dtype = torch.float64)
            prob_action_pre = prob_action_pre.reshape((len(test_partial_data), n_class))
            correct = 0
            with torch.no_grad():
                for feature, target, weight, in test_loader:
                    feature, target, weight = feature.to(device), target.to(device), weight.to(device)
                    output = policy_learning_model(feature)
                    y_prediction = output.max(1, keepdim=True)[1]
                    correct += y_prediction.eq(target.view_as(y_prediction)).sum().item()

            test_loss = 1 - float(correct)/len(test_loader.dataset)
            print('epoch:',epoch,',test_loss:', round(test_loss,4))
            test_regret_list[epoch-1] = test_loss
            if test_loss<best_loss:
                best_loss = test_loss
                best_epoch = epoch
            elif epoch - best_epoch>10:
                break
            for p in optimizer.param_groups:
                p['lr'] *= args.lr_decay
        if args.dataset == 1:
            dataset_name = 'uci_'
        elif args.dataset == 2:
            dataset_name = 'mnist_'
        elif args.dataset == 3:
            dataset_name = 'ciafr10_'
        elif args.dataset == 4:
            dataset_name = 'covertype_'
        else:
            dataset_name = 'not_impletemnet'

        if args.mode == 1:
            model_name = 'dm_'
        elif args.mode == 2:
            model_name = 'ips_'
        elif args.mode == 3:
            model_name = 'dr_'
        else:
            model_name = 'not_impletemnet'

        if args.policy == 0:
            policy_name = 'uniform_'
        elif args.policy == 1:
            policy_name = 'small_'
        elif args.policy == 2:
            policy_name = 'large_'
        elif args.policy == 3:
            policy_name = 'd_' + str(args.alpha) + '_'
        elif args.policy == 4:
            policy_name = 'tweak_' + str(args.rou) + '_'

        np.savetxt('train_regret/' + args.save_file_name+dataset_name + model_name  + policy_name + '_'  + str(simulation) + "_results_doubly_2.csv", np.array(regret_list), delimiter=",")
        np.savetxt('test_regret/' + args.save_file_name+dataset_name + model_name  + policy_name + '_' + str(simulation) +  "_results_doubly_2.csv", test_regret_list, delimiter=",")
        np.savetxt('test_regret/' + args.save_file_name+dataset_name + model_name  + policy_name  +  '_'  + str(simulation) +  "every_n_batch.csv", np.array(loss_evaluate_result), delimiter=",")
        print('Average loss: {:.4f}\n'.format(test_loss))
    
           

if __name__ == '__main__':
    main()

        


 
