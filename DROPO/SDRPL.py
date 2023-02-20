from __future__ import print_function
import argparse
from sympy import I
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


torch.set_default_tensor_type('torch.DoubleTensor')

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

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, x):
        return x



def round_value(a):
    if a>1:
        a = 1.0
    elif a<0:
        a = 0
    return a

def my_softmax(x):
    n = np.shape(x)[0]
    max_x, _ = torch.max(x, dim=1)
    max_x = torch.reshape(max_x, (n, 1))
    exp_x = torch.exp(x - max_x)
    p = exp_x / torch.reshape(torch.sum(exp_x, dim=1), (n, 1))
    p[p<10e-8] = 0
    return p
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
         
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Covariate Shift')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs-training', type=int, default=40 , metavar='N',
                        help='number of epochs in training (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_pg', type=float, default=0.0005, metavar='LR',
                        help='learning rate for policy gradient (default: 0.001)')
    parser.add_argument('--filename', type=str, default='optdigits.csv',
                        help='the file that contains training data')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--delta', type=float, default=0.1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--mode', type=int, default=1, metavar='N',
                        help='1, policy is a uniform default policy, 2 is a policy is a known policy trained from biased samples, 3 policy is an unknown policy uniform policy, 4 is an unknown biased policy ')
    parser.add_argument('--dataset', type=int, default=2, metavar='N',
                        help='1, uci, 2 mnist, 3 cifar10')
    parser.add_argument('--policy', type=int, default=4, metavar='N',
                        help='0, policy is a uniform default policy, 1 is a policy small shift policy, 2 policy is large policy, 3 is d uniform policy, 4 is tweak')
    parser.add_argument('--alpha', type=float, default=1, metavar='N',
                        help='direclect shift parameter')
    parser.add_argument('--rou', type=float, default=0.95, metavar='N',
                        help='tweak one shift parameter')
    parser.add_argument('--clip_weight', type=bool, default=True, metavar='N',
                        help='whether clip grad for robust regression')
    parser.add_argument('--weights_upper_bound', type=float, default=100, metavar='N',
                        help='upper bound for weight clip')
    parser.add_argument('--weights_lower_bound', type=float, default=0.0005, metavar='N',
                        help='lower bound for weight clip')
    parser.add_argument('--epochs_policy_gradient', type=int, default=12, metavar='N',
                        help='epochs for policy gradient')
    parser.add_argument('--save-file-name', type=str, default='', metavar='N',
                        help='save regret filename')
    parser.add_argument('--lr-decay', type=float, default=0.8, metavar='N', help='dedcay rate of lr')
    parser.add_argument('--simulation', type=int, default=10, metavar='N', help='interaction-num')
    parser.add_argument('--ALPHA', type=float, default=0.5, metavar='N', help='interaction-num')
    args = parser.parse_args()
    for simulation in range(args.simulation):
        
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
            len(m_train)
            m_test = len(test_data)
            n_dim = 54
            n_class = 7
        elif args.dataset == 5:
            all_data = DATA('adult_processed.csv')
            n_dim = 92
            n_class = 14

        # just for testing, add as an osption later
        # data_inarray = np.genfromtxt(args.filename, delimiter=',')
        # print(len(data_inarray))
        # n_dim = 64
        # n_class = 10
        n_nodes = 32
        policy_default = 1.0 / n_class
        # data_size = len(all_data)
        # print(np.shape(data_inarray))
        if args.dataset != 4:
            data_size = len(all_data)
            rand_idx = np.random.permutation(data_size)
            training_ratio = 0.6
            m_train = int(training_ratio * data_size)
            train_data = data.Subset(all_data, rand_idx[0: m_train])
            test_data = data.Subset(all_data, rand_idx[m_train: -1])
            m_test = data_size - m_train


        weight_st = np.ones(m_train)

        weighted_train = WEIGHT_DATA_SET(train_data, weight_st,args)


        # m_test = data_size - m_train
        weight_st = np.ones(m_test)
        weighted_test = WEIGHT_DATA_SET(test_data, weight_st,args)

        test_loader = data.DataLoader(weighted_test,
            batch_size=args.batch_size, shuffle=False, **kwargs)

        # prob_logging = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01 , 0.01, 0.46, 0.46]
        if args.policy == 1:
            prob_logging = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.46, 0.46]

        elif args.policy == 0:
            prob_logging = 0.1 * np.ones(n_class)

        elif args.policy == 2:
            # prob = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01 , 0.01, 0.01, 0.91]
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
            print(prob_logging)

        train_data_mse = DATA_defined_prob(train_data, n_class, n_dim, args, prob_logging)
        l = len(train_data_mse)
        print(l)
        # prob_logging = train_data_mse.logging_policy

        ## policy learning using full data
        test_partial_data = DATA_fullaction(train_data, n_class)

        policy_learning_model = Net(n_dim,n_nodes, n_class)
        policy_learning_model = policy_learning_model.to(device)
        myloss = nn.L1Loss()


        test_regret_list = np.zeros([args.epochs_policy_gradient])


        ALPHA = args.ALPHA
        context = torch.ones(l,n_dim)
        action,reward,action_true = torch.ones(l),torch.ones(l),torch.ones(l)


        for i in range(l):
            context[i], action[i], reward[i], action_true[i] = train_data_mse[i]
            
        context, action, reward, action_true = context.to(device), action.long().to(device), reward.to(device), action_true.to(device)
        prob_logging = torch.tensor(prob_logging).to(device)
        optimizer = torch.optim.Adam(policy_learning_model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs_policy_gradient + 1):

            print("epoch:",epoch)
            for p in range(10):
                optimizer.zero_grad()
                Prob = my_softmax(policy_learning_model(context))

                S = Prob[torch.arange(len(Prob)),action]/torch.tensor(prob_logging)[action]
                W = S * torch.exp(-reward.detach()/ALPHA)
                S = S.to(device)
                W = W.to(device)

                Sn = torch.sum(S.detach())/l
                WS = torch.sum(W)/(Sn*l)

                loss = WS
                if p %5 == 4:
                    print("Sn:",Sn)
                    print("loss:",loss)
                loss.backward()
                optimizer.step()
            for _ in range(1):
                Prob = my_softmax(policy_learning_model(context))
                S = Prob[torch.arange(len(Prob)),action]/torch.tensor(prob_logging)[action]
                W = S * torch.exp(reward.detach()/ALPHA)
                S = S.to(device)
                W = W.to(device)
                Sn = torch.sum(S.detach())/l
                WS = torch.sum(W)/(Sn*l)
                phi1 = -torch.dot(reward,W)/(ALPHA*l*Sn*WS)-torch.log(WS)-args.delta
                phi2 = torch.dot(reward,W)**2/(ALPHA**3*(l*Sn)**2*WS**2)-torch.dot(reward**2,W)/(ALPHA**3*l*Sn*WS)
                ALPHA = ALPHA - 0.001 * phi1.detach()/phi2.detach()
            print("WS:",WS,"reward*W:",torch.dot(reward,W),"Sn:",Sn)
            print("phi1:",phi1,"phi2:",phi2)
            print("ALPHA:",ALPHA)
            correct = 0
            with torch.no_grad():
                for feature, target, weight, in test_loader:
                    feature, target, weight = feature.to(device), target.to(device), weight.to(device)

                    output = policy_learning_model(feature)
                    y_prediction = output.max(1, keepdim=True)[1]

                    correct += y_prediction.eq(target.view_as(y_prediction)).sum().item()

            test_loss = 1 - float(correct)/len(test_loader.dataset)
            test_regret_list[epoch-1] = test_loss
            print('test_loss:',test_loss)
            for p in optimizer.param_groups:
                p['lr'] *= args.lr_decay

        if args.policy == 0:
            policy_name = 'uniform_'
        elif args.policy == 1:
            policy_name = 'small_'
        elif args.policy == 2:
            policy_name = 'large_'
        elif args.policy == 3:
            policy_name = 'd_'+ str(args.alpha) + '_'
        elif args.policy == 4:
            policy_name = 'tweak_' + str(args.rou) + '_'

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
        print('sinian0'+dataset_name+  policy_name +  '_'  + str(simulation) +".csv")
        np.savetxt('test_regret/'+'sinian0'+dataset_name+  policy_name +  '_'  + str(simulation) +".csv", np.array(test_regret_list), delimiter=",")
if __name__ == '__main__':
    main()
