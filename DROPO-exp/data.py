from __future__ import print_function
import torch.utils.data as data
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from scipy.stats import dirichlet
import torch.utils.data as data


def my_softmax(x):
    n = np.shape(x)[0]
    max_x, _ = torch.max(x, dim=1)
    max_x = torch.reshape(max_x, (n, 1))
    exp_x = torch.exp(x - max_x)
    p = exp_x / torch.reshape(torch.sum(exp_x, dim=1), (n, 1))
    p[p<10e-8] = 0
    return p

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


class DATA(data.Dataset):


    def __init__(self, folder, transform=None):
   
    
        self.read_file(folder)  
        self.transform = transform


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.tensor(self.data[index]), torch.tensor(self.labels[index]).long()

        return data, target

    def __len__(self):
        return len(self.data)

    def get_data(self):
        return self.data

    def get_labels(self):
        return self.labels 


    def read_file(self,folder):

        data = np.genfromtxt(folder, delimiter=',')
        self.data = data[:, 0:-1]
        self.labels = data[:, -1] - 1
        print(np.unique(self.labels))


class DATA_action(data.Dataset):
    ''' create action data from full data'''
    def __init__(self, dataset, ithclass, transform=None):
   
        #generate action dataset from full data
        
        rewards = np.zeros(len(dataset))
        for i in range(len(dataset)):
            data, label = dataset[i]

            if ithclass == label:
                rewards[i] = 0
            else:
                rewards[i] = 1
        
        self.rewards = rewards
        self.transform = transform
        self.dataset = dataset


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, _ =  self.dataset[index]
        rewards = torch.tensor(self.rewards[index])

        return data, rewards

    def __len__(self):
        return len(self.dataset)

class DATA_fullaction(data.Dataset):
    ''' create action data from full data'''
    def __init__(self, dataset, n_class, transform=None):
   
        #generate action dataset from full data
        self.n_class = n_class
        data_new = []
        action_new = []
        reward_new = []
        action_true = []
        for i in range(len(dataset)):
            data, label = dataset[i]

            for j in range(n_class):
                data_new.append(data)
                action_new.append(j)
                action_true.append(label)
                if j != label:
                    reward_new.append(0.0)
                else:
                    reward_new.append(1.0)


        self.rewards = reward_new
        self.action = action_new
        self.data = data_new
        self.action_true = action_true

    def get_new_evaluation_prob(self,policy):
        prob_action_next = torch.empty([0, self.n_class])
        with torch.no_grad():
            for data in self.new_data:
                output_learning = policy(data)
                prob = my_softmax(output_learning)
                prob = prob.detach().numpy()
                prob_action_next = np.concatenate((prob_action_next, prob), axis=0)
        prob_action_pre = torch.tensor(prob_action_next, dtype = torch.float64)
        prob_action_pre = prob_action_pre.reshape((len(self.new_data), self.n_class))
        return prob_action_pre


    def add_training_data(self,dataset):
        data_new = []
        action_new = []
        reward_new = []
        action_true = []
        for i in range(len(dataset)):
            data, label = dataset[i]
            for j in range(self.n_class):
                data_new.append(data)
                action_new.append(j)
                action_true.append(label)
                if j != label:
                    reward_new.append(0.0)
                else:
                    reward_new.append(1.0)
        self.new_data = data_new
        #
        # print(self.data.shape)
        #
        # self.rewards = np.concatenate((self.rewards,reward_new),axis=0)
        # self.action = np.concatenate((self.rewards,action_new),axis=0)
        # self.data = np.concatenate((self.rewards,data_new),axis=0)
        # self.action_true = np.concatenate((self.rewards,action_true),axis=0)

        self.rewards = self.rewards + reward_new
        self.action = self.action + action_new
        self.data = self.data + data_new
        self.action_true = self.action_true + action_true
        # print(data_new)
        # print(self.data.shape)

            # if len(i) < 10:
            #     print(i)





    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data  =  torch.tensor(self.data[index])
        action = torch.tensor(self.action[index])

        rewards = torch.tensor(self.rewards[index])

        action_true = torch.tensor(self.action_true[index])

        return data, action, rewards, action_true

    def __len__(self):
        return len(self.data)

    def get_features(self):
        return self.data

    def get_action(self):
        return self.action

    def get_reward(self):
        return self.rewards

    def get_action_true(self):
        return self.action_true


class DATA_policy_gradient(data.Dataset):
    ''' create action data from full data'''
    def __init__(self, features, rewards, actions, action_true, transform=None):
        
        self.rewards = rewards
        self.actions = actions
        self.features = features
        self.action_true = action_true


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data  =  torch.tensor(self.features[index])
        action = torch.tensor(self.actions[index])

        rewards = torch.tensor(self.rewards[index])
        action_true = torch.tensor(self.action_true[index])


        return data, action, rewards, action_true

    def __len__(self):
        return len(self.features)

class DATA_partial_random(data.Dataset):
    '''create partial data from full data'''
    def __init__(self, dataset, n_class, transform=None):
   
        #generate partial labeled dataset
        data1, _ = dataset[0] 
        actions = np.zeros(len(dataset))
        rewards = np.zeros(len(dataset))
        features = np.zeros((len(dataset), np.shape(data1)[0]))

        for i in range(len(dataset)):
            data, label = dataset[i]
            features[i] = data
            action = np.random.choice(n_class, 1)
            actions[i] = action[0]
            if action[0] == label:
                rewards[i] = 1
        self.features = features
        self.actions = actions
        self.rewards = rewards
        self.transform = transform
        self.dataset = dataset


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, label =  self.dataset[index]
        actions, rewards = torch.tensor(self.actions[index]).long(), torch.tensor(self.rewards[index])

        return data, actions, rewards

    def __len__(self):
        return len(self.dataset)

    def get_features(self):
        return self.features

    def get_action(self):
        return self.actions.astype(int)

    def get_reward(self):
        return self.rewards




class DATA_partial_action(data.Dataset):
    ''' create action data from partial data'''
    def __init__(self, dataset, ithclass, transform=None):
        m = len(dataset)
        # data_sample, _, _, _= dataset[0]
        # random policy cases
      
        data_sample, _, _, _= dataset[0]

       
        n = np.shape(data_sample)[0]
        # data_new = np.empty((0, n))
        # reward_new = np.empty((0, 1))
        data_new = []
        reward_new = []
        policy_new = np.empty([0])
        for i in range(len(dataset)):
            data, action, reward, _ = dataset[i]
            # random policy cases
            # data, action, reward = dataset[i]
            # print(data)
            # print(np.shape(data))
            if action == ithclass:
                # data_new = np.concatenate((data_new, data))
                # reward_new = np.concatenate((reward_new, reward))
                data_new.append(data)
                reward_new.append(reward)

                # policy_new = np.concatenate([policy_new, [policy]])

        self.data = data_new
       
        self.rewards = reward_new
        self.transform = transform
        # self.policy = policy_new
        self.dataset = dataset


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        
        data =  torch.tensor(self.data[index])

        rewards = torch.tensor(self.rewards[index])

        return data, rewards

    def __len__(self):
        return len(self.data)

    # def get_policy(self):
    #     return self.policy

class DATA_partial_logistic(data.Dataset):
    '''create partial data from full data

       when you have a csv of data and data is small enough
    '''
    def __init__(self, data, n_class, estimated_policy = False, transform=None):

        '''
        data has the last column as labels
        '''
   
        #generate partial labeled dataset
     
        actions = np.zeros(len(data))
        rewards = np.zeros(len(data))
        policy = np.zeros(len(data))
        
        # generate covariate shifted data
        pca = PCA(n_components=1)
        projected = pca.fit_transform(data[:, 0:-1])
        mean_proj = np.mean(projected)
        min_proj = np.min(projected)
        mu = (mean_proj - min_proj)/1.5+ min_proj
        var = (mean_proj - min_proj)
        ## sampling
        data_shift = []
        label_shift = []
        for i in range(len(data)):
            select_prob = gaussian(projected[i], mu, var)
            if select_prob > 1:
                select_prob = 1
            rand = np.random.uniform(0, 1, 1)
            
            if rand < select_prob:
                data_shift.append(data[i, 0:-1])
                label_shift.append(data[i, -1])
        # learn logistic model for sampling
        clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter = 1, multi_class='multinomial').fit(data_shift, label_shift)
        # sample y

        for i in range(len(data)):
            label = data[i, -1]
            # sample action
            rand = np.random.uniform(0, 1, 1)
        
            threshold = 0
            prob = clf.predict_proba(data[i, 0:-1].reshape(1, -1))
            # print(prob)
            for j in range(n_class):
            
                threshold = threshold + prob[0][j]
                
                if rand[0] < threshold:
                    action = j
                    actions[i] = action
                    if action != label:
                        rewards[i] = 1
                    policy[i] = prob[0][j]
                    break

        policy_estimate = np.zeros(len(data))
        if estimated_policy == True:
            # learn logistic model for estimating logging policy
            model = LogisticRegression(random_state=0, solver='lbfgs', max_iter =1, multi_class='multinomial').fit(data[:, 0:-1], actions)       
            logistic_policy = model.predict_proba(data[:, 0:-1])
            for i in range(len(data)):
                policy_estimate[i] = logistic_policy[0][int(actions[i])]
            # using robust classification?

            
        self.actions = actions
        self.rewards = rewards
        if estimated_policy == False:

            self.policy = policy
            print(policy)
            self.model = clf
        else:
            self.policy = policy_estimate
            self.model = model


        self.transform = transform
        self.data = data
        self.estimated_policy = estimated_policy
        self.unique_action =  np.unique(actions)

     



    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data =  torch.tensor(self.data[index, 0:-1])
        if self.estimated_policy == False:

            actions, rewards, policy = torch.tensor(self.actions[index]).long(), torch.tensor(self.rewards[index]), self.policy[index]
        else:
          
            actions, rewards, policy = torch.tensor(self.actions[index]).long(), torch.tensor(self.rewards[index]), self.policy[index]
            
        return data, actions, rewards, policy

    def __len__(self):
        return len(self.data)

    def get_model(self):
        return self.model



class DATA_partial_logistic_deep(data.Dataset):
    '''create partial data from full data

       when training using deep learning
    '''
    def __init__(self, data, n_class, model, estimated_policy = False, transform=None):

        '''
        data is another dataset class
        '''
   
        #generate partial labeled dataset
     
        actions = np.zeros(len(data))
        rewards = np.zeros(len(data))
        policy = np.zeros(len(data))
        
        # sample y

        for i in range(len(data)):
            # label = data[i, -1]
            # sample action
            data_sample, target = data[i]
            rand = np.random.uniform(0, 1, 1)
        
            threshold = 0

            prob = my_softmax(model(data_sample))
            # print(prob)
            
            for j in range(n_class):
            
                threshold = threshold + prob[0][j]
                
                if rand[0] < threshold:
                    action = j
                    actions[i] = action
                    if action == target:
                        rewards[i] = 1
                    # policy[i] = prob[0][j]
                    break
             
        # if estimated_policy == True:
        #     # learn logistic model for estimating logging policy
        #     model = LogisticRegression(random_state=0, solver='lbfgs', max_iter = 10, multi_class='multinomial').fit(data[:, 0:-1], actions)       
        #     logistic_policy = model.predict_proba(data[:, 0:-1])
        #     # using robust classification?

            
        self.actions = actions
        self.rewards = rewards
        # if estimated_policy == False:

        # self.policy = policy
        # else:
        #     self.policy = logistic_policy

        self.transform = transform
        self.data = data
        # self.estimated_policy = estimated_policy
        self.unique_action =  np.unique(actions)
     



    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data =  self.data[index][0]
        # if self.estimated_policy == False:

        actions, rewards = torch.tensor(self.actions[index]).long(), torch.tensor(self.rewards[index])
        # else:
          
            # actions, rewards, policy = torch.tensor(self.actions[index]).long(), torch.tensor(self.rewards[index]), self.policy[index][np.where(self.unique_action==int(self.actions[index]))]
            
        return data, actions, rewards

    def __len__(self):
        return len(self.data)

class DATA_online(data.Dataset):
    def __init__(self,args, data, n_class, n_dim,train_data_robust,transform=None):
        self.args = args
        self.dataset = data
        self.n_class = n_class
        self.n_dim = n_dim
        self.transform = transform

        self.interaction_start_index = 0
        self.interaction_num = args.interaction_num

        self.actions = train_data_robust.actions
        self.rewards = train_data_robust.rewards
        self.policy = train_data_robust.policy
        self.action_true = train_data_robust.action_true
        self.data = train_data_robust.data

        self.rand_idx = np.random.permutation((len(self.dataset)))
        self.ips_grad = False



    def sample_online_data(self):

        # start = self.interaction_start_index
        # end = min(self.interaction_start_index+self.interaction_num,len(self.dataset) - 1)
        # # :self.interaction_start_index+self.interaction_num]
        # if start >= len(self.dataset) - 1:
        #     return False
        #
        # data_new = data.Subset(self.dataset, self.rand_idx[start: end])
        # self.interaction_start_index = self.interaction_start_index + self.interaction_num
        data_new = data.Subset(self.dataset, np.random.permutation((len(self.dataset)))[: self.interaction_num])

        return data_new

    def interaction(self, trained_policy,logging_policy = None):
        data = self.sample_online_data()
        if data == False:
            return False
        self.new_data = data
        features = np.zeros((self.n_class * len(data), self.n_dim))
        actions = np.zeros(self.n_class * len(data))
        rewards = np.zeros(self.n_class * len(data))
        policy = np.zeros(self.n_class * len(data))
        action_true = np.zeros(self.n_class * len(data))
        self.add_logging_policy = np.zeros((self.n_class * len(data),self.n_class))


        for i in range(len(data)):
            data_sample, target = data[i]
            if logging_policy is not None:
                action_prob = list(logging_policy)
            else:
                rand_epsilon = np.random.uniform(0, 1, 1)
                if rand_epsilon>self.args.epsilon:
                    action_prob = self.my_softmax(trained_policy(data_sample))
                    action_prob = list(action_prob.squeeze(0).detach().numpy())

                else:
                    action_prob = list(np.ones(self.n_class)/self.n_class)

            for j in range(self.n_class):

                rand = np.random.uniform(0, 1, 1)

                threshold = 0

                features[self.n_class * i + j] = data_sample

                action_true[self.n_class * i + j] = target
                self.add_logging_policy[self.n_class * i + j] = action_prob
                for k in range(self.n_class):

                    threshold = threshold + action_prob[k]

                    if rand[0] < threshold:

                        actions[self.n_class * i + j] = k
                        policy[self.n_class * i + j] = action_prob[k]
                        if k == target:
                            rewards[self.n_class * i + j] = 1
                        # policy[i] = prob[0][j]
                        break
        self.new_data_add = self.new_data

        self.new_action = actions
        self.new_reward = rewards
        self.new_policy = policy
        self.new_action_true = action_true
        self.new_data = features
        self.ips_grad = True


        self.actions = np.concatenate((self.actions,actions),axis=0)
        self.rewards = np.concatenate((self.rewards,rewards),axis=0)
        self.policy = np.concatenate((self.policy,policy),axis=0)
        self.action_true = np.concatenate((self.action_true,action_true),axis=0)
        self.data = np.concatenate((self.data,features),axis=0)
        return True




    def my_softmax(self, x):
        n = np.shape(x)[0]
        max_x, _ = torch.max(x, dim=1)
        max_x = torch.reshape(max_x, (n, 1))
        exp_x = torch.exp(x - max_x)
        p = exp_x / torch.reshape(torch.sum(exp_x, dim=1), (n, 1))
        p[p < 10e-8] = 0
        return p

    def get_features(self):
        return self.data

    def get_action(self):
        return self.actions.astype(int)

    def get_reward(self):
        return self.rewards

    def get_action_true(self):
        return self.action_true.astype(int)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data = torch.tensor(self.data[index])

        actions, rewards = torch.tensor(self.actions[index]).long(), torch.tensor(self.rewards[index])

        policy, action_true = torch.tensor(self.policy[index]), torch.tensor(self.action_true[index]).long()

        return data, actions, rewards, action_true


class DATA_defined_prob(data.Dataset):
    '''create partial data from full data

       when training using deep learning
    '''
    def __init__(self, data, n_class, n_dim, args, prob, transform=None):

        '''
        data is another dataset class
        '''
   
        #generate partial labeled dataset

        features = np.zeros((n_class*len(data), n_dim))
        actions = np.zeros(n_class*len(data))
        rewards = np.zeros(n_class *len(data))
        policy = np.zeros(n_class *len(data))
        action_true = np.zeros(n_class *len(data))

        # self.logging_policy = np.zeros((len(data), n_class))
        # dirichlet_label_shift = False
        # tweak_shift = False
        #
        # if args.policy== 1:
        #     prob = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01 , 0.01, 0.46, 0.46]
        #     # prob = 0.01 * np.ones(n_class)
        #     # prob[-1] = 1 - 0.01 * (n_class - 2 )
        #     # prob[-2] = 1 - 0.01 * (n_class - 2)
        #     # for i in range(len(self.logging_policy)):
        #     #     self.logging_policy[i, :] = np.array(prob)
        # elif args.policy== 0:
        #     # prob = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 , 0.1, 0.1, 0.1]
        #     prob = 0.1 * np.ones(n_class)
        #     for i in range(len(self.logging_policy)):
        #         self.logging_policy[i, :] = np.array(prob)
        # elif args.policy == 2:
        #     # prob = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01 , 0.01, 0.01, 0.91]
        #     prob = 0.01 * np.ones(n_class)
        #     prob[-1] = 1 - 0.01 * (n_class - 1)
        #     for i in range(len(self.logging_policy)):
        #         self.logging_policy[i, :] = np.array(prob)
        # elif args.policy == 3:
        #     dirichlet_label_shift = True
        #     alpha = np.ones(n_class) * args.alpha
        # elif args.policy == 4:
        #     tweak_shift = True
        #     self.rou = args.rou

        
        # sample y

        for i in range(len(data)):
            data_sample, target = data[i]

            for j in range(n_class):

             
                rand = np.random.uniform(0, 1, 1)
            
                threshold = 0
                # print(prob)
               
                features[n_class *i+j] = data_sample
              
                action_true[n_class *i+j] = target
                for k in range(n_class):
                
                    threshold = threshold + prob[k]
                    
                    if rand[0] < threshold:
                     
                        actions[n_class *i+j] = k
                        policy[n_class *i+j] = prob[k]
                        if k == target:
                            rewards[n_class *i+j] = 1
                        # policy[i] = prob[0][j]
                        break
            
        self.actions = actions
        self.rewards = rewards
        self.policy = policy
        self.action_true = action_true
        self.transform = transform
        self.data = features
        # self.estimated_policy = estimated_policy
        self.unique_action =  np.unique(actions)
        self.logging_policy = np.array(prob)*np.ones((n_class * len(data),n_class))
        # print(self.logging_policy)
        print(self.unique_action)
     

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data =  torch.tensor(self.data[index])
    
        actions, rewards = torch.tensor(self.actions[index]).long(), torch.tensor(self.rewards[index])
           
        policy, action_true =  torch.tensor(self.policy[index]), torch.tensor(self.action_true[index]).long()
           
        return data, actions, rewards, action_true

    def __len__(self):
        return len(self.data)


    def get_features(self):
        return self.data

    def get_action(self):
        return self.actions.astype(int)

    def get_reward(self):
        return self.rewards

    def get_action_true(self):
        return self.action_true.astype(int)

# convert to learning sampling policies

class DATA_learn_policy(data.Dataset):


    def __init__(self, data):
   
    
        self.data = data 


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target, _, = self.data[index]

        return data, target

    def __len__(self):
        return len(self.data)





        
       






