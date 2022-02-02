import sys, os
import numpy as np
import math
import random
import itertools
import copy

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import torch


class CustomDataset(Dataset):
    """Custom Dataset.

    Attributes:
        x: A PyTorch tensor for x features of data.
        y: A PyTorch tensor for y features (true labels) of data.
        z: A PyTorch tensor for z features (sensitive attributes) of data.
    """
    def __init__(self, x_tensor, y_tensor, z_tensor):
        """Initializes the dataset with torch tensors."""
        
        self.x = x_tensor
        self.y = y_tensor
        self.z = z_tensor
        
    def __getitem__(self, index):
        """Returns the selected data based on the index information."""
        
        return (self.x[index], self.y[index], self.z[index])

    def __len__(self):
        """Returns the length of data."""
        
        return len(self.x)
class CustomDataset1(Dataset):
    """Custom Dataset.

    Attributes:
        x: A PyTorch tensor for x features of data.
        y: A PyTorch tensor for y features (true labels) of data.
        z1: A PyTorch tensor for z1 features (sensitive attributes) of data.
        z2: A PyTorch tensor for z2 features (sensitive attributes) of data.
        z3: A PyTorch tensor for z3 features (sensitive attributes) of data.
    """
    def __init__(self, x_tensor, y_tensor, z1_tensor, z2_tensor, z3_tensor):
        """Initializes the dataset with torch tensors."""
        
        self.x = x_tensor
        self.y = y_tensor
        self.z1 = z1_tensor
        self.z2 = z2_tensor
        self.z3 = z3_tensor
        
    def __getitem__(self, index):
        """Returns the selected data based on the index information."""
        
        return (self.x[index], self.y[index], self.z1[index], self.z2[index], self.z3[index])

    def __len__(self):
        """Returns the length of data."""
        
        return len(self.x)    
class FairRobust(Sampler):
    """FairRobust (Sampler in DataLoader).
    
    This class is for implementing the lambda adjustment and batch selection of FairBatch [Roh et al., ICLR 2021] with robust training.

    Attributes:
        model: A model containing the intermediate states of the training.
        x_, y_, z_data: Tensor-based train data.
        alpha: A positive number for step size that used in the lambda adjustment.
        fairness_type: A string indicating the target fairness type 
                       among original, demographic parity (dp), equal opportunity (eqopp), and equalized odds (eqodds).
        replacement: A boolean indicating whether a batch consists of data with or without replacement.
        N: An integer counting the size of data.
        batch_size: An integer for the size of a batch.
        batch_num: An integer for total number of batches in an epoch.
        y_, z_item: Lists that contains the unique values of the y_data and z_data, respectively.
        yz_tuple: Lists for pairs of y_item and z_item.
        y_, z_, yz_mask: Dictionaries utilizing as array masks.
        y_, z_, yz_index: Dictionaries containing the indexes of each class.
        y_, z_, yz_len: Dictionaries containing the length information.
        clean_index: A list that contains the data indexes of selected samples.
        clean_y_, clean_z_, clean_yz_index: Dictionaries containing the indexes of each class in the selected set.
        clean_y_, clean_z_, clean_yz_len: Dictionaries containing the length information in the selected set.
        S: A dictionary containing the default size of each class in a batch.
        lb1, lb2: (0~1) real numbers indicating the lambda values for fairness [Roh et al., ICLR 2021].
        tau: (0~1) real number indicating the clean ratio of the data.
        warm_start: An integer for warm-start period.

        
    """
    def __init__(self, model, x_tensor, y_tensor, z_tensor, target_fairness, parameters, replacement = False, seed = 0):
        """Initializes FairBatch."""
        
        self.model = model
        
        np.random.seed(seed)
        random.seed(seed)
        
        self.x_data = x_tensor
        self.y_data = y_tensor
        self.z_data = z_tensor
        
        
        self.alpha = parameters.alpha
        self.fairness_type = target_fairness
        
        self.replacement = replacement
        
        self.N = len(z_tensor)
        
        self.batch_size = parameters.batch_size
        self.batch_num = int(len(self.y_data) / self.batch_size)
        
        # Takes the unique values of the tensors
        self.z_item = list(set(z_tensor.tolist()))
        self.y_item = list(set(y_tensor.tolist()))
        
        self.yz_tuple = list(itertools.product(self.y_item, self.z_item))
        
        # Makes masks
        self.z_mask = {}
        self.y_mask = {}
        self.yz_mask = {}
        
        for tmp_z in self.z_item:
            self.z_mask[tmp_z] = (self.z_data == tmp_z)
            
        for tmp_y in self.y_item:
            self.y_mask[tmp_y] = (self.y_data == tmp_y)
            
        for tmp_yz in self.yz_tuple:
            self.yz_mask[tmp_yz] = (self.y_data == tmp_yz[0]) & (self.z_data == tmp_yz[1])
        

        # Finds the index
        self.z_index = {}
        self.y_index = {}
        self.yz_index = {}
        
        for tmp_z in self.z_item:
            self.z_index[tmp_z] = (self.z_mask[tmp_z] == 1).nonzero().squeeze()
            
        for tmp_y in self.y_item:
            self.y_index[tmp_y] = (self.y_mask[tmp_y] == 1).nonzero().squeeze()
        
        for tmp_yz in self.yz_tuple:
            self.yz_index[tmp_yz] = (self.yz_mask[tmp_yz] == 1).nonzero().squeeze()
            
        self.entire_index = torch.FloatTensor([i for i in range(len(self.y_data))])
            
        # Length information
        self.z_len = {}
        self.y_len = {}
        self.yz_len = {}
        
        for tmp_z in self.z_item:
            self.z_len[tmp_z] = len(self.z_index[tmp_z])
            
        for tmp_y in self.y_item:
            self.y_len[tmp_y] = len(self.y_index[tmp_y])
            
        for tmp_yz in self.yz_tuple:
            self.yz_len[tmp_yz] = len(self.yz_index[tmp_yz])

        # Default batch size
        self.S = {}
        
        for tmp_yz in self.yz_tuple:
            self.S[tmp_yz] = self.batch_size * (self.yz_len[tmp_yz])/self.N

        
        self.lb1 = (self.S[1,1])/(self.S[1,1]+(self.S[1,0]))
        self.lb2 = (self.S[-1,1])/(self.S[-1,1]+(self.S[-1,0]))
        
        # For cleanselection parameters
        self.tau = parameters.tau # Clean ratio
        self.warm_start = parameters.warm_start
    
        self.count_epoch = 0
        
            
        # Clean sample selection
        self.clean_index = np.arange(0,len(self.y_data))
        
        # Finds the index
        self.clean_z_index = {}
        self.clean_y_index = {}
        self.clean_yz_index = {}
        
        for tmp_z in self.z_item:
            self.clean_z_index[tmp_z] = (self.z_mask[tmp_z] == 1)[self.clean_index].nonzero().squeeze()
            
        for tmp_y in self.y_item:
            self.clean_y_index[tmp_y] = (self.y_mask[tmp_y] == 1)[self.clean_index].nonzero().squeeze()
        
        for tmp_yz in self.yz_tuple:
            self.clean_yz_index[tmp_yz] = (self.yz_mask[tmp_yz] == 1)[self.clean_index].nonzero().squeeze()
        
        
       # Length information
        self.clean_z_len = {}
        self.clean_y_len = {}
        self.clean_yz_len = {}
        
        for tmp_z in self.z_item:
            self.clean_z_len[tmp_z] = len(self.clean_z_index[tmp_z])
            
        for tmp_y in self.y_item:
            self.clean_y_len[tmp_y] = len(self.clean_y_index[tmp_y])
            
        for tmp_yz in self.yz_tuple:
            self.clean_yz_len[tmp_yz] = len(self.clean_yz_index[tmp_yz])
 
      
    def get_logit(self):
        """Runs forward pass of the intermediate model with the training data.
        
        Returns:
            Outputs (logits) of the model.

        """
        
        self.model.eval()
        logit = self.model(self.x_data)
        
        return logit
    
    
    def adjust_lambda(self, logit):
        """Adjusts the lambda values using FairBatch [Roh et al., ICLR 2021].
        See our paper for algorithm details.
        
        Args: 
            logit: A torch tensor that contains the intermediate model's output on the training data.
        
        """
        
        criterion = torch.nn.BCELoss(reduction = 'none')
        
        
        if self.fairness_type == 'eqopp':
            
            yhat_yz = {}
            yhat_y = {}
                        
            eo_loss = criterion ((F.tanh(logit.squeeze())+1)/2, (self.y_data.squeeze()+1)/2)
            
            for tmp_yz in self.yz_tuple:
                yhat_yz[tmp_yz] = float(torch.sum(eo_loss[self.clean_yz_index[tmp_yz]])) / self.clean_yz_len[tmp_yz]
                
            for tmp_y in self.y_item:
                yhat_y[tmp_y] = float(torch.sum(eo_loss[self.clean_y_index[tmp_y]])) / self.clean_y_len[tmp_y]
            
            # lb1 * loss_z1 + (1-lb1) * loss_z0
            
            if yhat_yz[(1, 1)] > yhat_yz[(1, 0)]:
                self.lb1 += self.alpha
            else:
                self.lb1 -= self.alpha
                
            if self.lb1 < 0:
                self.lb1 = 0
            elif self.lb1 > 1:
                self.lb1 = 1 
                
        elif self.fairness_type == 'eqodds':
            
            yhat_yz = {}
            yhat_y = {}
                        
            eo_loss = criterion ((F.tanh(logit.squeeze())+1)/2, (self.y_data.squeeze()+1)/2)
            
            for tmp_yz in self.yz_tuple:
                yhat_yz[tmp_yz] = float(torch.sum(eo_loss[self.clean_yz_index[tmp_yz]])) / (self.clean_yz_len[tmp_yz]+1)
                
            for tmp_y in self.y_item:
                yhat_y[tmp_y] = float(torch.sum(eo_loss[self.clean_y_index[tmp_y]])) / (self.clean_y_len[tmp_y]+1)
            
            y1_diff = abs(yhat_yz[(1, 1)] - yhat_yz[(1, 0)])
            y0_diff = abs(yhat_yz[(-1, 1)] - yhat_yz[(-1, 0)])
            
            # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
            # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0
            
            if y1_diff > y0_diff:
                if yhat_yz[(1, 1)] > yhat_yz[(1, 0)]:
                    self.lb1 += self.alpha
                else:
                    self.lb1 -= self.alpha
            else:
                if yhat_yz[(-1, 1)] > yhat_yz[(-1, 0)]:
                    self.lb2 += self.alpha
                else:
                    self.lb2 -= self.alpha
                    
                
            if self.lb1 < 0:
                self.lb1 = 0
            elif self.lb1 > 1:
                self.lb1 = 1
                
            if self.lb2 < 0:
                self.lb2 = 0
            elif self.lb2 > 1:
                self.lb2 = 1
                
        elif self.fairness_type == 'dp':
            yhat_yz = {}
            yhat_y = {}
            
            ones_array = np.ones(len(self.y_data))
            ones_tensor = torch.FloatTensor(ones_array).cuda()
            dp_loss = criterion((F.tanh(logit.squeeze())+1)/2, ones_tensor.squeeze()) # Note that ones tensor puts as the true label
            
            for tmp_yz in self.yz_tuple:
                yhat_yz[tmp_yz] = float(torch.sum(dp_loss[self.clean_yz_index[tmp_yz]])) / self.clean_z_len[tmp_yz[1]]
                    
            
            y1_diff = abs(yhat_yz[(1, 1)] - yhat_yz[(1, 0)])
            y0_diff = abs(yhat_yz[(-1, 1)] - yhat_yz[(-1, 0)])
            
            # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
            # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0
            
            if y1_diff > y0_diff:
                if yhat_yz[(1, 1)] > yhat_yz[(1, 0)]:
                    self.lb1 += self.alpha
                else:
                    self.lb1 -= self.alpha
            else:
                if yhat_yz[(-1, 1)] > yhat_yz[(-1, 0)]: 
                    self.lb2 -= self.alpha
                else:
                    self.lb2 += self.alpha
                    
            if self.lb1 < 0:
                self.lb1 = 0
            elif self.lb1 > 1:
                self.lb1 = 1
                
            if self.lb2 < 0:
                self.lb2 = 0
            elif self.lb2 > 1:
                self.lb2 = 1

    def select_fair_robust_sample(self):
        """Selects fair and robust samples and adjusts the lambda values for fairness. 
        See our paper for algorithm details.
        
        Returns:
            Indexes that indicate the data.
            
        """
        
        logit = self.get_logit()
        
        self.adjust_lambda(logit)

        criterion = torch.nn.BCELoss(reduction = 'none')
        
        loss = criterion ((F.tanh(logit.squeeze())+1)/2, (self.y_data.squeeze()+1)/2)
        profit = torch.max(loss)-loss
        
        current_weight_sum = {}
        
        lb_ratio = {}
        
        for tmp_yz in self.yz_tuple:
            if tmp_yz == (1,1):
                lb_ratio[tmp_yz] = self.lb1
            elif tmp_yz == (1,0): 
                lb_ratio[tmp_yz] = 1-self.lb1
            elif tmp_yz == (-1,1):
                lb_ratio[tmp_yz] = self.lb2
            elif tmp_yz == (-1,0):
                lb_ratio[tmp_yz] = 1-self.lb2
            
            current_weight_sum[tmp_yz] = 0
        
        # Greedy-based algorithm
        
        (_, sorted_index) = torch.topk(profit, len(profit), largest=True, sorted=True)
        
        clean_index = []
        
        total_selected = 0
        
        desired_size = int(self.tau * len(self.y_data))
        
        for j in sorted_index:
            tmp_y = self.y_data[j].item()
            tmp_z = self.z_data[j].item()
            current_weight_list = list(current_weight_sum.values())
            
            if total_selected >= desired_size:
                break
            if all(i < desired_size for i in current_weight_list):
                clean_index.append(j)
                
                current_weight_sum[(tmp_y, tmp_z)] += 2 - lb_ratio[(tmp_y, tmp_z)]
                current_weight_sum[(tmp_y, 1-tmp_z)] += 1 - lb_ratio[(tmp_y, 1-tmp_z)]
                current_weight_sum[(tmp_y * -1, tmp_z)] += 1
                current_weight_sum[(tmp_y * -1, 1-tmp_z)] += 1
                
                total_selected += 1        
        
        clean_index = torch.LongTensor(clean_index).cuda()
        
        self.batch_num = int(len(clean_index)/self.batch_size)
        
        # Update the variables
        self.clean_index = clean_index
        
        for tmp_z in self.z_item:
            combined = torch.cat((self.z_index[tmp_z], self.clean_index))
            uniques, counts = combined.unique(return_counts=True)
            intersection = uniques[counts > 1]
            
            self.clean_z_index[tmp_z] = intersection 
            
        for tmp_y in self.y_item:
            combined = torch.cat((self.y_index[tmp_y], self.clean_index))
            uniques, counts = combined.unique(return_counts=True)
            intersection = uniques[counts > 1]
            
            self.clean_y_index[tmp_y] = intersection
        
        for tmp_yz in self.yz_tuple:
            combined = torch.cat((self.yz_index[tmp_yz], self.clean_index))
            uniques, counts = combined.unique(return_counts=True)
            intersection = uniques[counts > 1]
            
            self.clean_yz_index[tmp_yz] = intersection
        
        
        for tmp_z in self.z_item:
            self.clean_z_len[tmp_z] = len(self.clean_z_index[tmp_z])
            
        for tmp_y in self.y_item:
            self.clean_y_len[tmp_y] = len(self.clean_y_index[tmp_y])
        
        for tmp_yz in self.yz_tuple:
            self.clean_yz_len[tmp_yz] = len(self.clean_yz_index[tmp_yz])
            
        
        return clean_index
        
    
    def select_batch_replacement(self, batch_size, full_index, batch_num, replacement = False, weight = None):
        """Selects a certain number of batches based on the given batch size.
        
        Args: 
            batch_size: An integer for the data size in a batch.
            full_index: An array containing the candidate data indices.
            batch_num: An integer indicating the number of batches.
            replacement: A boolean indicating whether a batch consists of data with or without replacement.
        
        Returns:
            Indexes that indicate the data.
            
        """
        
        select_index = []
        
        if replacement == True:
            for _ in range(batch_num):
                if weight == None:
                    weight_norm = weight/torch.sum(weight)
                    select_index.append(np.random.choice(full_index, batch_size, replace = False, p = weight_norm))
                else:
                    select_index.append(np.random.choice(full_index, batch_size, replace = False))
        else:
            tmp_index = full_index.detach().cpu().numpy().copy()
            random.shuffle(tmp_index)
            
            start_idx = 0
            for i in range(batch_num):
                if start_idx + batch_size > len(full_index):
                    select_index.append(np.concatenate((tmp_index[start_idx:], tmp_index[ : batch_size - (len(full_index)-start_idx)])))
                    
                    start_idx = len(full_index)-start_idx
                else:

                    select_index.append(tmp_index[start_idx:start_idx + batch_size])
                    start_idx += batch_size
            
        return select_index


    
    def decide_fair_batch_size(self):
        """Calculates each class size based on the lambda values (lb1 and lb2) for fairness.
        
        Returns:
            Each class size for fairness.
            
        """
        
        each_size = {}

        for tmp_yz in self.yz_tuple:
            self.S[tmp_yz] = self.batch_size * (self.clean_yz_len[tmp_yz])/len(self.clean_index)

        # Based on the updated lambdas, determine the size of each class in a batch
        if self.fairness_type == 'eqopp':
            # lb1 * loss_z1 + (1-lb1) * loss_z0

            each_size[(1,1)] = round(self.lb1 * (self.S[(1,1)] + self.S[(1,0)]))
            each_size[(1,0)] = round((1-self.lb1) * (self.S[(1,1)] + self.S[(1,0)]))
            each_size[(-1,1)] = round(self.S[(-1,1)])
            each_size[(-1,0)] = round(self.S[(-1,0)])

        elif self.fairness_type == 'eqodds':
            # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
            # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0

            each_size[(1,1)] = round(self.lb1 * (self.S[(1,1)] + self.S[(1,0)]))
            each_size[(1,0)] = round((1-self.lb1) * (self.S[(1,1)] + self.S[(1,0)]))
            each_size[(-1,1)] = round(self.lb2 * (self.S[(-1,1)] + self.S[(-1,0)]))
            each_size[(-1,0)] = round((1-self.lb2) * (self.S[(-1,1)] + self.S[(-1,0)]))

        elif self.fairness_type == 'dp':
            # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
            # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0

            each_size[(1,1)] = round(self.lb1 * (self.S[(1,1)] + self.S[(1,0)]))
            each_size[(1,0)] = round((1-self.lb1) * (self.S[(1,1)] + self.S[(1,0)]))
            each_size[(-1,1)] = round(self.lb2 * (self.S[(-1,1)] + self.S[(-1,0)]))
            each_size[(-1,0)] = round((1-self.lb2) * (self.S[(-1,1)] + self.S[(-1,0)]))
        
        return each_size
        
        
    
    def __iter__(self):
        """Iters the full process of fair and robust sample selection for serving the batches to training.
        
        Returns:
            Indexes that indicate the data in each batch.
            
        """
        self.count_epoch += 1
        
        if self.count_epoch > self.warm_start:

            _ = self.select_fair_robust_sample()


            each_size = self.decide_fair_batch_size()

            # Get the indices for each class
            sort_index_y_1_z_1 = self.select_batch_replacement(each_size[(1, 1)], self.clean_yz_index[(1,1)], self.batch_num, self.replacement)
            sort_index_y_0_z_1 = self.select_batch_replacement(each_size[(-1, 1)], self.clean_yz_index[(-1,1)], self.batch_num, self.replacement)
            sort_index_y_1_z_0 = self.select_batch_replacement(each_size[(1, 0)], self.clean_yz_index[(1,0)], self.batch_num, self.replacement)
            sort_index_y_0_z_0 = self.select_batch_replacement(each_size[(-1, 0)], self.clean_yz_index[(-1,0)], self.batch_num, self.replacement)

            for i in range(self.batch_num):
                key_in_fairbatch = sort_index_y_0_z_0[i].copy()
                key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_1_z_0[i].copy()))
                key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_0_z_1[i].copy()))
                key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_1_z_1[i].copy()))

                random.shuffle(key_in_fairbatch)

                yield key_in_fairbatch

        else:
            entire_index = torch.FloatTensor([i for i in range(len(self.y_data))])

            sort_index = self.select_batch_replacement(self.batch_size, entire_index, self.batch_num, self.replacement)

            for i in range(self.batch_num):
                yield sort_index[i]
        
                               

    def __len__(self):
        """Returns the length of data."""
        
        return len(self.y_data)
class FairRobust1(Sampler):
    """FairRobust (Sampler in DataLoader).
    
    This class is for implementing the lambda adjustment and batch selection of FairBatch [Roh et al., ICLR 2021] with robust training.

    Attributes:
        model: A model containing the intermediate states of the training.
        x_, y_, z_data: Tensor-based train data.
        alpha: A positive number for step size that used in the lambda adjustment.
        fairness_type: A string indicating the target fairness type 
                       among original, demographic parity (dp), equal opportunity (eqopp), and equalized odds (eqodds).
        replacement: A boolean indicating whether a batch consists of data with or without replacement.
        N: An integer counting the size of data.
        batch_size: An integer for the size of a batch.
        batch_num: An integer for total number of batches in an epoch.
        y_, z_item: Lists that contains the unique values of the y_data and z_data, respectively.
        yz_tuple: Lists for pairs of y_item and z_item.
        y_, z_, yz_mask: Dictionaries utilizing as array masks.
        y_, z_, yz_index: Dictionaries containing the indexes of each class.
        y_, z_, yz_len: Dictionaries containing the length information.
        clean_index: A list that contains the data indexes of selected samples.
        clean_y_, clean_z_, clean_yz_index: Dictionaries containing the indexes of each class in the selected set.
        clean_y_, clean_z_, clean_yz_len: Dictionaries containing the length information in the selected set.
        S: A dictionary containing the default size of each class in a batch.
        lb1, lb2: (0~1) real numbers indicating the lambda values for fairness [Roh et al., ICLR 2021].
        tau: (0~1) real number indicating the clean ratio of the data.
        warm_start: An integer for warm-start period.

        
    """
    def __init__(self, model, x_tensor, y_tensor, z1_tensor, z2_tensor, z3_tensor, target_fairness, parameters, replacement = False, seed = 0):
        """Initializes FairBatch."""
        
        self.model = model
        
        np.random.seed(seed)
        random.seed(seed)
        
        self.x_data = x_tensor
        self.y_data = y_tensor
        self.z1_data = z1_tensor
        self.z2_data = z2_tensor
        self.z3_data = z3_tensor
        
        #alpha: how to modify rate of each category
        self.alpha = parameters.alpha
        self.fairness_type = target_fairness
        
        self.replacement = replacement
        # N: number of elements
        self.N = len(z1_tensor)
        
        self.batch_size = parameters.batch_size
        self.batch_num = int(len(self.y_data) / self.batch_size)
        
        # Takes the unique values of the tensors
        self.z1_item = list(set(z1_tensor.tolist()))
        self.z2_item = list(set(z2_tensor.tolist()))
        self.z3_item = list(set(z3_tensor.tolist()))
        self.y_item = list(set(y_tensor.tolist()))
        # yz_tuple: (-1,0), (-1,1), (1,1), (1,0)
        self.yz1_tuple = list(itertools.product(self.y_item, self.z1_item))
        self.yz2_tuple = list(itertools.product(self.y_item, self.z2_item))
        self.yz3_tuple = list(itertools.product(self.y_item, self.z3_item))
        
        # Makes masks
        #z_mask: z = 1 or z = 0 (list)
        self.z1_mask = {}
        self.z2_mask = {}
        self.z3_mask = {}
        #z_mask: y = 1 or y = -1 (list)
        self.y_mask = {}
        self.yz1_mask = {}
        self.yz2_mask = {}
        self.yz3_mask = {}
        
        for tmp_z1 in self.z1_item:
            self.z1_mask[tmp_z1] = (self.z1_data == tmp_z1)
        
        for tmp_z2 in self.z2_item:
            self.z2_mask[tmp_z2] = (self.z2_data == tmp_z2)
        
        for tmp_z3 in self.z3_item:
            self.z3_mask[tmp_z3] = (self.z3_data == tmp_z3)
            
        for tmp_y in self.y_item:
            self.y_mask[tmp_y] = (self.y_data == tmp_y)
            
        for tmp_yz1 in self.yz1_tuple:
            self.yz1_mask[tmp_yz1] = (self.y_data == tmp_yz1[0]) & (self.z1_data == tmp_yz1[1])
        for tmp_yz2 in self.yz2_tuple:
            self.yz2_mask[tmp_yz2] = (self.y_data == tmp_yz2[0]) & (self.z2_data == tmp_yz2[1])
        for tmp_yz3 in self.yz3_tuple:
            self.yz3_mask[tmp_yz3] = (self.y_data == tmp_yz3[0]) & (self.z3_data == tmp_yz3[1])
        

        # Finds the index
        self.z1_index = {}
        self.z2_index = {}
        self.z3_index = {}
        self.y_index = {}
        self.yz1_index = {}
        self.yz2_index = {}
        self.yz3_index = {}
        #index of z=1, z=0
        for tmp_z1 in self.z1_item:
            self.z1_index[tmp_z1] = (self.z1_mask[tmp_z1] == 1).nonzero().squeeze()
        for tmp_z2 in self.z2_item:
            self.z2_index[tmp_z2] = (self.z2_mask[tmp_z2] == 1).nonzero().squeeze()
        for tmp_z3 in self.z3_item:
            self.z3_index[tmp_z3] = (self.z3_mask[tmp_z3] == 1).nonzero().squeeze()
        #index of y=-1, y=0
        for tmp_y in self.y_item:
            self.y_index[tmp_y] = (self.y_mask[tmp_y] == 1).nonzero().squeeze()
        #index of (1,1); (-1,1); (1,0); (-1,0)
        for tmp_yz1 in self.yz1_tuple:
            self.yz1_index[tmp_yz1] = (self.yz1_mask[tmp_yz1] == 1).nonzero().squeeze()
        for tmp_yz2 in self.yz2_tuple:
            self.yz2_index[tmp_yz2] = (self.yz2_mask[tmp_yz2] == 1).nonzero().squeeze()
        for tmp_yz3 in self.yz3_tuple:
            self.yz3_index[tmp_yz3] = (self.yz3_mask[tmp_yz3] == 1).nonzero().squeeze()
            
        self.entire_index = torch.FloatTensor([i for i in range(len(self.y_data))])
            
        # Length information
        self.z1_len = {}
        self.z2_len = {}
        self.z3_len = {}
        self.y_len = {}
        self.yz1_len = {}
        self.yz2_len = {}
        self.yz3_len = {}
        
        for tmp_z1 in self.z1_item:
            self.z1_len[tmp_z1] = len(self.z1_index[tmp_z1])
        for tmp_z2 in self.z2_item:
            self.z2_len[tmp_z2] = len(self.z2_index[tmp_z2])
        for tmp_z3 in self.z3_item:
            self.z3_len[tmp_z3] = len(self.z3_index[tmp_z3])
            
        for tmp_y in self.y_item:
            self.y_len[tmp_y] = len(self.y_index[tmp_y])
            
        for tmp_yz1 in self.yz1_tuple:
            self.yz1_len[tmp_yz1] = len(self.yz1_index[tmp_yz1])
        for tmp_yz2 in self.yz2_tuple:
            self.yz2_len[tmp_yz2] = len(self.yz2_index[tmp_yz2])
        for tmp_yz3 in self.yz3_tuple:
            self.yz3_len[tmp_yz3] = len(self.yz3_index[tmp_yz3])

        # Default batch size
        self.S1 = {}
        self.S2 = {}
        self.S3 = {}
        
        # len of each (y,z)
        for tmp_yz1 in self.yz1_tuple:
            self.S1[tmp_yz1] = self.batch_size * (self.yz1_len[tmp_yz1])/self.N
        for tmp_yz2 in self.yz2_tuple:
            self.S2[tmp_yz2] = self.batch_size * (self.yz2_len[tmp_yz2])/self.N
        for tmp_yz3 in self.yz3_tuple:
            self.S3[tmp_yz3] = self.batch_size * (self.yz3_len[tmp_yz3])/self.N

        # lambda = (y=1,z=1)/(y=1)
        self.z1_lb1 = (self.S1[1,1])/(self.S1[1,1]+(self.S1[1,0]))
        # (y=-1,z=1)/(y=-1)
        self.z1_lb2 = (self.S1[-1,1])/(self.S1[-1,1]+(self.S1[-1,0]))
        
        # lambda = (y=1,z=1)/(y=1)
        self.z2_lb1 = (self.S2[1,1])/(self.S2[1,1]+(self.S2[1,0]))
        # (y=-1,z=1)/(y=-1)
        self.z2_lb2 = (self.S2[-1,1])/(self.S2[-1,1]+(self.S2[-1,0]))
        
        # lambda = (y=1,z=1)/(y=1)
        self.z3_lb1 = (self.S3[1,1])/(self.S3[1,1]+(self.S3[1,0]))
        # (y=-1,z=1)/(y=-1)
        self.z3_lb2 = (self.S3[-1,1])/(self.S3[-1,1]+(self.S3[-1,0]))
        
        # For cleanselection parameters
        self.tau = parameters.tau # Clean ratio
        self.warm_start = parameters.warm_start
    
        self.count_epoch = 0
        
            
        # Clean sample selection
        self.clean_index = np.arange(0,len(self.y_data))
        
        # Finds the index
        self.clean_z1_index = {}
        self.clean_z2_index = {}
        self.clean_z3_index = {}
        self.clean_y_index = {}
        self.clean_yz1_index = {}
        self.clean_yz2_index = {}
        self.clean_yz3_index = {}
        # index of each z values in list
        for tmp_z1 in self.z1_item:
            self.clean_z1_index[tmp_z1] = (self.z1_mask[tmp_z1] == 1)[self.clean_index].nonzero().squeeze()
        for tmp_z2 in self.z2_item:
            self.clean_z2_index[tmp_z2] = (self.z2_mask[tmp_z2] == 1)[self.clean_index].nonzero().squeeze()
        for tmp_z3 in self.z3_item:
            self.clean_z3_index[tmp_z3] = (self.z3_mask[tmp_z3] == 1)[self.clean_index].nonzero().squeeze()
        # index of each y values in list
        for tmp_y in self.y_item:
            self.clean_y_index[tmp_y] = (self.y_mask[tmp_y] == 1)[self.clean_index].nonzero().squeeze()
        # index of each yz values in list
        for tmp_yz1 in self.yz1_tuple:
            self.clean_yz1_index[tmp_yz1] = (self.yz1_mask[tmp_yz1] == 1)[self.clean_index].nonzero().squeeze()
        for tmp_yz2 in self.yz2_tuple:
            self.clean_yz2_index[tmp_yz2] = (self.yz2_mask[tmp_yz2] == 1)[self.clean_index].nonzero().squeeze()
        for tmp_yz3 in self.yz3_tuple:
            self.clean_yz3_index[tmp_yz3] = (self.yz3_mask[tmp_yz3] == 1)[self.clean_index].nonzero().squeeze()
        
        
       # Length information
        self.clean_z1_len = {}
        self.clean_z2_len = {}
        self.clean_z3_len = {}
        self.clean_y_len = {}
        self.clean_yz1_len = {}
        self.clean_yz2_len = {}
        self.clean_yz3_len = {}
        
        for tmp_z1 in self.z1_item:
            self.clean_z1_len[tmp_z1] = len(self.clean_z1_index[tmp_z1])
        for tmp_z2 in self.z2_item:
            self.clean_z2_len[tmp_z2] = len(self.clean_z2_index[tmp_z2])
        for tmp_z3 in self.z3_item:
            self.clean_z3_len[tmp_z3] = len(self.clean_z3_index[tmp_z3])
            
        for tmp_y in self.y_item:
            self.clean_y_len[tmp_y] = len(self.clean_y_index[tmp_y])
            
        for tmp_yz1 in self.yz1_tuple:
            self.clean_yz1_len[tmp_yz1] = len(self.clean_yz1_index[tmp_yz1])
        for tmp_yz2 in self.yz2_tuple:
            self.clean_yz2_len[tmp_yz2] = len(self.clean_yz2_index[tmp_yz2])
        for tmp_yz3 in self.yz3_tuple:
            self.clean_yz3_len[tmp_yz3] = len(self.clean_yz3_index[tmp_yz3])
 
      
    def get_logit(self):
        """Runs forward pass of the intermediate model with the training data.
        
        Returns:
            Outputs (logits) of the model.

        """
        
        self.model.eval()
        logit = self.model(self.x_data)
        
        return logit
    
    
    def adjust_lambda(self, logit):
        """Adjusts the lambda values using FairBatch [Roh et al., ICLR 2021].
        See our paper for algorithm details.
        
        Args: 
            logit: A torch tensor that contains the intermediate model's output on the training data.
        
        """
        
        criterion = torch.nn.BCELoss(reduction = 'none')
        
        
        if self.fairness_type == 'eqopp':
            
            yhat_yz1 = {}
            yhat_yz2 = {}
            yhat_yz3 = {}
            yhat_y = {}
                        
            eo_loss = criterion ((F.tanh(logit.squeeze())+1)/2, (self.y_data.squeeze()+1)/2)
            
            for tmp_yz1 in self.yz1_tuple:
                yhat_yz1[tmp_yz1] = float(torch.sum(eo_loss[self.clean_yz1_index[tmp_yz1]])) / self.clean_yz1_len[tmp_yz1]
            for tmp_yz2 in self.yz2_tuple:
                yhat_yz2[tmp_yz2] = float(torch.sum(eo_loss[self.clean_yz2_index[tmp_yz2]])) / self.clean_yz2_len[tmp_yz2]
            for tmp_yz3 in self.yz3_tuple:
                yhat_yz3[tmp_yz3] = float(torch.sum(eo_loss[self.clean_yz3_index[tmp_yz3]])) / self.clean_yz3_len[tmp_yz3]
                
            for tmp_y in self.y_item:
                yhat_y[tmp_y] = float(torch.sum(eo_loss[self.clean_y_index[tmp_y]])) / self.clean_y_len[tmp_y]
            
            # lb1 * loss_z1 + (1-lb1) * loss_z0
            if yhat_yz1[(1, 1)] > yhat_yz1[(1, 0)]:
                self.z1_lb1 += self.alpha
            else:
                self.z1_lb1 -= self.alpha
                
            if yhat_yz2[(1, 1)] > yhat_yz2[(1, 0)]:
                self.z2_lb1 += self.alpha
            else:
                self.z2_lb1 -= self.alpha
                
            if yhat_yz3[(1, 1)] > yhat_yz3[(1, 0)]:
                self.z3_lb1 += self.alpha
            else:
                self.z3_lb1 -= self.alpha
                
            if self.z1_lb1 < 0:
                self.z1_lb1 = 0
            elif self.z1_lb1 > 1:
                self.z1_lb1 = 1 
                
            if self.z2_lb1 < 0:
                self.z2_lb1 = 0
            elif self.z2_lb1 > 1:
                self.z2_lb1 = 1 
                
            if self.z3_lb1 < 0:
                self.z3_lb1 = 0
            elif self.z3_lb1 > 1:
                self.z3_lb1 = 1     
        elif self.fairness_type == 'eqodds':
            
            yhat_yz1 = {}
            yhat_yz2 = {}
            yhat_yz3 = {}
            yhat_y = {}
                        
            eo_loss = criterion ((F.tanh(logit.squeeze())+1)/2, (self.y_data.squeeze()+1)/2)
            
            for tmp_yz1 in self.yz1_tuple:
                yhat_yz1[tmp_yz1] = float(torch.sum(eo_loss[self.clean_yz1_index[tmp_yz1]])) / (self.clean_yz1_len[tmp_yz1]+1)
            
            for tmp_yz2 in self.yz2_tuple:
                yhat_yz2[tmp_yz2] = float(torch.sum(eo_loss[self.clean_yz2_index[tmp_yz2]])) / (self.clean_yz2_len[tmp_yz2]+1)
                
            for tmp_yz3 in self.yz3_tuple:
                yhat_yz3[tmp_yz3] = float(torch.sum(eo_loss[self.clean_yz3_index[tmp_yz3]])) / (self.clean_yz3_len[tmp_yz3]+1)
            
                
            for tmp_y in self.y_item:
                yhat_y[tmp_y] = float(torch.sum(eo_loss[self.clean_y_index[tmp_y]])) / (self.clean_y_len[tmp_y]+1)
            
            y1_diff_z1 = abs(yhat_yz1[(1, 1)] - yhat_yz1[(1, 0)])
            y0_diff_z1 = abs(yhat_yz1[(-1, 1)] - yhat_yz1[(-1, 0)])
            
            # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
            # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0
            
            if y1_diff_z1 > y0_diff_z1:
                if yhat_yz1[(1, 1)] > yhat_yz1[(1, 0)]:
                    self.z1_lb1 += self.alpha
                else:
                    self.z1_lb1 -= self.alpha
            else:
                if yhat_yz1[(-1, 1)] > yhat_yz1[(-1, 0)]:
                    self.z1_lb2 += self.alpha
                else:
                    self.z1_lb2 -= self.alpha
            
            y1_diff_z2 = abs(yhat_yz2[(1, 1)] - yhat_yz2[(1, 0)])
            y0_diff_z2 = abs(yhat_yz2[(-1, 1)] - yhat_yz2[(-1, 0)])
            
            # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
            # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0
            
            if y1_diff_z2 > y0_diff_z2:
                if yhat_yz2[(1, 1)] > yhat_yz2[(1, 0)]:
                    self.z2_lb1 += self.alpha
                else:
                    self.z2_lb1 -= self.alpha
            else:
                if yhat_yz2[(-1, 1)] > yhat_yz2[(-1, 0)]:
                    self.z2_lb2 += self.alpha
                else:
                    self.z2_lb2 -= self.alpha
                    
            y1_diff_z3 = abs(yhat_yz3[(1, 1)] - yhat_yz3[(1, 0)])
            y0_diff_z3 = abs(yhat_yz3[(-1, 1)] - yhat_yz3[(-1, 0)])
            
            # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
            # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0
            
            if y1_diff_z3 > y0_diff_z3:
                if yhat_yz3[(1, 1)] > yhat_yz3[(1, 0)]:
                    self.z3_lb1 += self.alpha
                else:
                    self.z3_lb1 -= self.alpha
            else:
                if yhat_yz3[(-1, 1)] > yhat_yz3[(-1, 0)]:
                    self.z3_lb2 += self.alpha
                else:
                    self.z3_lb2 -= self.alpha
                    
                
            if self.z1_lb1 < 0:
                self.z1_lb1 = 0
            elif self.z1_lb1 > 1:
                self.z1_lb1 = 1 
                
            if self.z2_lb1 < 0:
                self.z2_lb1 = 0
            elif self.z2_lb1 > 1:
                self.z2_lb1 = 1 
                
            if self.z3_lb1 < 0:
                self.z3_lb1 = 0
            elif self.z3_lb1 > 1:
                self.z3_lb1 = 1   
                
        elif self.fairness_type == 'dp':
            yhat_yz1 = {}
            yhat_yz2 = {}
            yhat_yz3 = {}
            yhat_y = {}
            
            ones_array = np.ones(len(self.y_data))
            ones_tensor = torch.FloatTensor(ones_array).cuda()
            dp_loss = criterion((F.tanh(logit.squeeze())+1)/2, ones_tensor.squeeze()) # Note that ones tensor puts as the true label
            
            for tmp_yz1 in self.yz1_tuple:
                yhat_yz1[tmp_yz1] = float(torch.sum(dp_loss[self.clean_yz1_index[tmp_yz1]])) / self.clean_z1_len[tmp_yz1[1]]
            
            for tmp_yz2 in self.yz2_tuple:
                yhat_yz2[tmp_yz2] = float(torch.sum(dp_loss[self.clean_yz2_index[tmp_yz2]])) / self.clean_z2_len[tmp_yz2[1]]
            
            for tmp_yz3 in self.yz3_tuple:
                yhat_yz3[tmp_yz3] = float(torch.sum(dp_loss[self.clean_yz3_index[tmp_yz3]])) / self.clean_z3_len[tmp_yz3[1]]
                    
            
            y1_diff_z1 = abs(yhat_yz1[(1, 1)] - yhat_yz1[(1, 0)])
            y0_diff_z1 = abs(yhat_yz1[(-1, 1)] - yhat_yz1[(-1, 0)])
            
            # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
            # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0
            
            if y1_diff_z1 > y0_diff_z1:
                if yhat_yz1[(1, 1)] > yhat_yz1[(1, 0)]:
                    self.z1_lb1 += self.alpha
                else:
                    self.z1_lb1 -= self.alpha
            else:
                if yhat_yz1[(-1, 1)] > yhat_yz1[(-1, 0)]:
                    self.z1_lb2 += self.alpha
                else:
                    self.z1_lb2 -= self.alpha
            
            y1_diff_z2 = abs(yhat_yz2[(1, 1)] - yhat_yz2[(1, 0)])
            y0_diff_z2 = abs(yhat_yz2[(-1, 1)] - yhat_yz2[(-1, 0)])
            
            # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
            # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0
            
            if y1_diff_z2 > y0_diff_z2:
                if yhat_yz2[(1, 1)] > yhat_yz2[(1, 0)]:
                    self.z2_lb1 += self.alpha
                else:
                    self.z2_lb1 -= self.alpha
            else:
                if yhat_yz2[(-1, 1)] > yhat_yz2[(-1, 0)]:
                    self.z2_lb2 += self.alpha
                else:
                    self.z2_lb2 -= self.alpha
                    
            y1_diff_z3 = abs(yhat_yz3[(1, 1)] - yhat_yz3[(1, 0)])
            y0_diff_z3 = abs(yhat_yz3[(-1, 1)] - yhat_yz3[(-1, 0)])
            
            # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
            # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0
            
            if y1_diff_z3 > y0_diff_z3:
                if yhat_yz3[(1, 1)] > yhat_yz3[(1, 0)]:
                    self.z3_lb1 += self.alpha
                else:
                    self.z3_lb1 -= self.alpha
            else:
                if yhat_yz3[(-1, 1)] > yhat_yz3[(-1, 0)]:
                    self.z3_lb2 += self.alpha
                else:
                    self.z3_lb2 -= self.alpha
                    
            if self.z1_lb1 < 0:
                self.z1_lb1 = 0
            elif self.z1_lb1 > 1:
                self.z1_lb1 = 1 
                
            if self.z2_lb1 < 0:
                self.z2_lb1 = 0
            elif self.z2_lb1 > 1:
                self.z2_lb1 = 1 
                
            if self.z3_lb1 < 0:
                self.z3_lb1 = 0
            elif self.z3_lb1 > 1:
                self.z3_lb1 = 1   

    def select_fair_robust_sample(self):
        """Selects fair and robust samples and adjusts the lambda values for fairness. 
        See our paper for algorithm details.
        
        Returns:
            Indexes that indicate the data.
            
        """
        
        logit = self.get_logit()
        
        self.adjust_lambda(logit)

        criterion = torch.nn.BCELoss(reduction = 'none')
        
        loss = criterion ((F.tanh(logit.squeeze())+1)/2, (self.y_data.squeeze()+1)/2)
        profit = torch.max(loss)-loss
        
        current_weight_sum_z1 = {}
        current_weight_sum_z2 = {}
        current_weight_sum_z3 = {}
        
        lb_ratio_z1 = {}
        lb_ratio_z2 = {}
        lb_ratio_z3 = {}
        
        for tmp_yz1 in self.yz1_tuple:
            if tmp_yz1 == (1,1):
                lb_ratio_z1[tmp_yz1] = self.z1_lb1
            elif tmp_yz1 == (1,0): 
                lb_ratio_z1[tmp_yz1] = 1-self.z1_lb1
            elif tmp_yz1 == (-1,1):
                lb_ratio_z1[tmp_yz1] = self.z1_lb2
            elif tmp_yz1 == (-1,0):
                lb_ratio_z1[tmp_yz1] = 1-self.z1_lb2
            
            current_weight_sum_z1[tmp_yz1] = 0
        
        for tmp_yz2 in self.yz2_tuple:
            if tmp_yz2 == (1,1):
                lb_ratio_z2[tmp_yz2] = self.z2_lb1
            elif tmp_yz2 == (1,0): 
                lb_ratio_z2[tmp_yz2] = 1-self.z2_lb1
            elif tmp_yz2 == (-1,1):
                lb_ratio_z2[tmp_yz2] = self.z2_lb2
            elif tmp_yz2 == (-1,0):
                lb_ratio_z2[tmp_yz2] = 1-self.z2_lb2
            
            current_weight_sum_z2[tmp_yz2] = 0
        
        for tmp_yz3 in self.yz3_tuple:
            if tmp_yz3 == (1,1):
                lb_ratio_z3[tmp_yz3] = self.z3_lb1
            elif tmp_yz3 == (1,0): 
                lb_ratio_z3[tmp_yz3] = 1-self.z3_lb1
            elif tmp_yz3 == (-1,1):
                lb_ratio_z3[tmp_yz3] = self.z3_lb2
            elif tmp_yz3 == (-1,0):
                lb_ratio_z3[tmp_yz3] = 1-self.z3_lb2
            
            current_weight_sum_z3[tmp_yz3] = 0
        
        # Greedy-based algorithm
        
        (_, sorted_index) = torch.topk(profit, len(profit), largest=True, sorted=True)
        
        clean_index = []
        
        total_selected = 0
        
        desired_size = int(self.tau * len(self.y_data))
        
        for j in sorted_index:
            tmp_y = self.y_data[j].item()
            tmp_z1 = self.z1_data[j].item()
            tmp_z2 = self.z2_data[j].item()
            tmp_z3 = self.z3_data[j].item()
            current_weight_list_z1 = list(current_weight_sum_z1.values())
            current_weight_list_z2 = list(current_weight_sum_z2.values())
            current_weight_list_z3 = list(current_weight_sum_z3.values())
            if total_selected >= desired_size:
                break
            if all(i < desired_size for i in current_weight_list_z1) and all(i < desired_size for i in current_weight_list_z2) and all(i < desired_size for i in current_weight_list_z3):
                clean_index.append(j)
                
                current_weight_sum_z1[(tmp_y, tmp_z1)] += 2 - lb_ratio_z1[(tmp_y, tmp_z1)]
                current_weight_sum_z1[(tmp_y, 1-tmp_z1)] += 1 - lb_ratio_z1[(tmp_y, 1-tmp_z1)]
                current_weight_sum_z1[(tmp_y * -1, tmp_z1)] += 1
                current_weight_sum_z1[(tmp_y * -1, 1-tmp_z1)] += 1
                
                current_weight_sum_z2[(tmp_y, tmp_z2)] += 2 - lb_ratio_z2[(tmp_y, tmp_z2)]
                current_weight_sum_z2[(tmp_y, 1-tmp_z2)] += 1 - lb_ratio_z2[(tmp_y, 1-tmp_z2)]
                current_weight_sum_z2[(tmp_y * -1, tmp_z2)] += 1
                current_weight_sum_z2[(tmp_y * -1, 1-tmp_z2)] += 1
                
                current_weight_sum_z3[(tmp_y, tmp_z3)] += 2 - lb_ratio_z3[(tmp_y, tmp_z3)]
                current_weight_sum_z3[(tmp_y, 1-tmp_z3)] += 1 - lb_ratio_z3[(tmp_y, 1-tmp_z3)]
                current_weight_sum_z3[(tmp_y * -1, tmp_z3)] += 1
                current_weight_sum_z3[(tmp_y * -1, 1-tmp_z3)] += 1
                
                total_selected += 1        
        
        clean_index = torch.LongTensor(clean_index).cuda()
        
        self.batch_num = int(len(clean_index)/self.batch_size)
        
        # Update the variables
        self.clean_index = clean_index
        
        for tmp_z1 in self.z1_item:
            combined = torch.cat((self.z1_index[tmp_z1], self.clean_index))
            uniques, counts = combined.unique(return_counts=True)
            intersection = uniques[counts > 1]
            
            self.clean_z1_index[tmp_z1] = intersection 
        
        for tmp_z2 in self.z2_item:
            combined = torch.cat((self.z2_index[tmp_z2], self.clean_index))
            uniques, counts = combined.unique(return_counts=True)
            intersection = uniques[counts > 1]
            
            self.clean_z2_index[tmp_z2] = intersection 
        
        for tmp_z3 in self.z3_item:
            combined = torch.cat((self.z3_index[tmp_z3], self.clean_index))
            uniques, counts = combined.unique(return_counts=True)
            intersection = uniques[counts > 1]
            
            self.clean_z3_index[tmp_z3] = intersection 
            
        for tmp_y in self.y_item:
            combined = torch.cat((self.y_index[tmp_y], self.clean_index))
            uniques, counts = combined.unique(return_counts=True)
            intersection = uniques[counts > 1]
            
            self.clean_y_index[tmp_y] = intersection
        
        for tmp_yz1 in self.yz1_tuple:
            combined = torch.cat((self.yz1_index[tmp_yz1], self.clean_index))
            uniques, counts = combined.unique(return_counts=True)
            intersection = uniques[counts > 1]
            
            self.clean_yz1_index[tmp_yz1] = intersection
            
        for tmp_yz2 in self.yz2_tuple:
            combined = torch.cat((self.yz2_index[tmp_yz2], self.clean_index))
            uniques, counts = combined.unique(return_counts=True)
            intersection = uniques[counts > 1]
            
            self.clean_yz2_index[tmp_yz2] = intersection
            
        for tmp_yz3 in self.yz3_tuple:
            combined = torch.cat((self.yz3_index[tmp_yz3], self.clean_index))
            uniques, counts = combined.unique(return_counts=True)
            intersection = uniques[counts > 1]
            
            self.clean_yz3_index[tmp_yz3] = intersection
        
        
        for tmp_z1 in self.z1_item:
            self.clean_z1_len[tmp_z1] = len(self.clean_z1_index[tmp_z1])
        
        for tmp_z2 in self.z2_item:
            self.clean_z2_len[tmp_z2] = len(self.clean_z2_index[tmp_z2])
        
        for tmp_z3 in self.z3_item:
            self.clean_z3_len[tmp_z3] = len(self.clean_z3_index[tmp_z3])
            
        for tmp_y in self.y_item:
            self.clean_y_len[tmp_y] = len(self.clean_y_index[tmp_y])
        
        for tmp_yz1 in self.yz1_tuple:
            self.clean_yz1_len[tmp_yz1] = len(self.clean_yz1_index[tmp_yz1])
        
        for tmp_yz2 in self.yz2_tuple:
            self.clean_yz2_len[tmp_yz2] = len(self.clean_yz2_index[tmp_yz2])
        
        for tmp_yz3 in self.yz2_tuple:
            self.clean_yz3_len[tmp_yz3] = len(self.clean_yz3_index[tmp_yz3])
            
        
        return clean_index
        
    
    def select_batch_replacement(self, batch_size, full_index, batch_num, replacement = False, weight = None):
        """Selects a certain number of batches based on the given batch size.
        
        Args: 
            batch_size: An integer for the data size in a batch.
            full_index: An array containing the candidate data indices.
            batch_num: An integer indicating the number of batches.
            replacement: A boolean indicating whether a batch consists of data with or without replacement.
        
        Returns:
            Indexes that indicate the data.
            
        """
        
        select_index = []
        
        if replacement == True:
            for _ in range(batch_num):
                if weight == None:
                    weight_norm = weight/torch.sum(weight)
                    select_index.append(np.random.choice(full_index, batch_size, replace = False, p = weight_norm))
                else:
                    select_index.append(np.random.choice(full_index, batch_size, replace = False))
        else:
            tmp_index = full_index.detach().cpu().numpy().copy()
            random.shuffle(tmp_index)
            
            start_idx = 0
            for i in range(batch_num):
                if start_idx + batch_size > len(full_index):
                    select_index.append(np.concatenate((tmp_index[start_idx:], tmp_index[ : batch_size - (len(full_index)-start_idx)])))
                    
                    start_idx = len(full_index)-start_idx
                else:

                    select_index.append(tmp_index[start_idx:start_idx + batch_size])
                    start_idx += batch_size
            
        return select_index


    
    def decide_fair_batch_size(self):
        """Calculates each class size based on the lambda values (lb1 and lb2) for fairness.
        
        Returns:
            Each class size for fairness.
            
        """
        
        each_size_z1 = {}
        each_size_z2 = {}
        each_size_z3 = {}

        for tmp_yz1 in self.yz1_tuple:
            self.S1[tmp_yz1] = self.batch_size * (self.clean_yz1_len[tmp_yz1])/len(self.clean_index)
            
        for tmp_yz2 in self.yz2_tuple:
            self.S2[tmp_yz2] = self.batch_size * (self.clean_yz2_len[tmp_yz2])/len(self.clean_index)
        
        for tmp_yz3 in self.yz3_tuple:
            self.S3[tmp_yz3] = self.batch_size * (self.clean_yz3_len[tmp_yz3])/len(self.clean_index)

        # Based on the updated lambdas, determine the size of each class in a batch
        if self.fairness_type == 'eqopp':
            # lb1 * loss_z1 + (1-lb1) * loss_z0

            each_size_z1[(1,1)] = round(self.z1_lb1 * (self.S1[(1,1)] + self.S1[(1,0)]))
            each_size_z1[(1,0)] = round((1-self.z1_lb1) * (self.S1[(1,1)] + self.S1[(1,0)]))
            each_size_z1[(-1,1)] = round(self.S1[(-1,1)])
            each_size_z1[(-1,0)] = round(self.S1[(-1,0)])
            
            each_size_z2[(1,1)] = round(self.z2_lb1 * (self.S2[(1,1)] + self.S2[(1,0)]))
            each_size_z2[(1,0)] = round((1-self.z2_lb1) * (self.S2[(1,1)] + self.S2[(1,0)]))
            each_size_z2[(-1,1)] = round(self.S2[(-1,1)])
            each_size_z2[(-1,0)] = round(self.S2[(-1,0)])
            
            each_size_z3[(1,1)] = round(self.z3_lb1 * (self.S3[(1,1)] + self.S3[(1,0)]))
            each_size_z3[(1,0)] = round((1-self.z3_lb1) * (self.S3[(1,1)] + self.S3[(1,0)]))
            each_size_z3[(-1,1)] = round(self.S3[(-1,1)])
            each_size_z3[(-1,0)] = round(self.S3[(-1,0)])

        elif self.fairness_type == 'eqodds':
            # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
            # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0

            each_size_z1[(1,1)] = round(self.z1_lb1 * (self.S1[(1,1)] + self.S1[(1,0)]))
            each_size_z1[(1,0)] = round((1-self.z1_lb1) * (self.S1[(1,1)] + self.S1[(1,0)]))
            each_size_z1[(-1,1)] = round(self.z1_lb2 * (self.S1[(-1,1)] + self.S1[(-1,0)]))
            each_size_z1[(-1,0)] = round((1-self.z1_lb2) * (self.S1[(-1,1)] + self.S1[(-1,0)]))
            
            each_size_z2[(1,1)] = round(self.z2_lb1 * (self.S2[(1,1)] + self.S2[(1,0)]))
            each_size_z2[(1,0)] = round((1-self.z2_lb1) * (self.S2[(1,1)] + self.S2[(1,0)]))
            each_size_z2[(-1,1)] = round(self.z2_lb2 * (self.S2[(-1,1)] + self.S2[(-1,0)]))
            each_size_z2[(-1,0)] = round((1-self.z2_lb2) * (self.S2[(-1,1)] + self.S2[(-1,0)]))
            
            each_size_z3[(1,1)] = round(self.z3_lb1 * (self.S3[(1,1)] + self.S3[(1,0)]))
            each_size_z3[(1,0)] = round((1-self.z3_lb1) * (self.S3[(1,1)] + self.S3[(1,0)]))
            each_size_z3[(-1,1)] = round(self.z3_lb2 * (self.S3[(-1,1)] + self.S3[(-1,0)]))
            each_size_z3[(-1,0)] = round((1-self.z3_lb2) * (self.S3[(-1,1)] + self.S3[(-1,0)]))

        elif self.fairness_type == 'dp':
            # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
            # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0

            each_size_z1[(1,1)] = round(self.z1_lb1 * (self.S1[(1,1)] + self.S1[(1,0)]))
            each_size_z1[(1,0)] = round((1-self.z1_lb1) * (self.S1[(1,1)] + self.S1[(1,0)]))
            each_size_z1[(-1,1)] = round(self.z1_lb2 * (self.S1[(-1,1)] + self.S1[(-1,0)]))
            each_size_z1[(-1,0)] = round((1-self.z1_lb2) * (self.S1[(-1,1)] + self.S1[(-1,0)]))
            
            each_size_z2[(1,1)] = round(self.z2_lb1 * (self.S2[(1,1)] + self.S2[(1,0)]))
            each_size_z2[(1,0)] = round((1-self.z2_lb1) * (self.S2[(1,1)] + self.S2[(1,0)]))
            each_size_z2[(-1,1)] = round(self.z2_lb2 * (self.S2[(-1,1)] + self.S2[(-1,0)]))
            each_size_z2[(-1,0)] = round((1-self.z2_lb2) * (self.S2[(-1,1)] + self.S2[(-1,0)]))
            
            each_size_z3[(1,1)] = round(self.z3_lb1 * (self.S3[(1,1)] + self.S3[(1,0)]))
            each_size_z3[(1,0)] = round((1-self.z3_lb1) * (self.S3[(1,1)] + self.S3[(1,0)]))
            each_size_z3[(-1,1)] = round(self.z3_lb2 * (self.S3[(-1,1)] + self.S3[(-1,0)]))
            each_size_z3[(-1,0)] = round((1-self.z3_lb2) * (self.S3[(-1,1)] + self.S3[(-1,0)]))
        
        return each_size_z1, each_size_z2, each_size_z3
        
        
    
    def __iter__(self):
        """Iters the full process of fair and robust sample selection for serving the batches to training.
        
        Returns:
            Indexes that indicate the data in each batch.
            
        """
        self.count_epoch += 1
        
        if self.count_epoch > self.warm_start:

            _ = self.select_fair_robust_sample()


            each_size_z1, each_size_z2, each_size_z3 = self.decide_fair_batch_size()

            # Get the indices for each class
            sort_index_y_1_z1_1 = self.select_batch_replacement(each_size_z1[(1, 1)], self.clean_yz1_index[(1,1)], self.batch_num, self.replacement)
            sort_index_y_0_z1_1 = self.select_batch_replacement(each_size_z1[(-1, 1)], self.clean_yz1_index[(-1,1)], self.batch_num, self.replacement)
            sort_index_y_1_z1_0 = self.select_batch_replacement(each_size_z1[(1, 0)], self.clean_yz1_index[(1,0)], self.batch_num, self.replacement)
            sort_index_y_0_z1_0 = self.select_batch_replacement(each_size_z1[(-1, 0)], self.clean_yz1_index[(-1,0)], self.batch_num, self.replacement)
            
            sort_index_y_1_z2_1 = self.select_batch_replacement(each_size_z2[(1, 1)], self.clean_yz2_index[(1,1)], self.batch_num, self.replacement)
            sort_index_y_0_z2_1 = self.select_batch_replacement(each_size_z2[(-1, 1)], self.clean_yz2_index[(-1,1)], self.batch_num, self.replacement)
            sort_index_y_1_z2_0 = self.select_batch_replacement(each_size_z2[(1, 0)], self.clean_yz2_index[(1,0)], self.batch_num, self.replacement)
            sort_index_y_0_z2_0 = self.select_batch_replacement(each_size_z2[(-1, 0)], self.clean_yz2_index[(-1,0)], self.batch_num, self.replacement)
            
            sort_index_y_1_z3_1 = self.select_batch_replacement(each_size_z3[(1, 1)], self.clean_yz3_index[(1,1)], self.batch_num, self.replacement)
            sort_index_y_0_z3_1 = self.select_batch_replacement(each_size_z3[(-1, 1)], self.clean_yz3_index[(-1,1)], self.batch_num, self.replacement)
            sort_index_y_1_z3_0 = self.select_batch_replacement(each_size_z3[(1, 0)], self.clean_yz3_index[(1,0)], self.batch_num, self.replacement)
            sort_index_y_0_z3_0 = self.select_batch_replacement(each_size_z3[(-1, 0)], self.clean_yz3_index[(-1,0)], self.batch_num, self.replacement)

            for i in range(self.batch_num):
                key_in_fairbatch_z1 = sort_index_y_0_z1_0[i].copy()
                key_in_fairbatch_z1 = np.hstack((key_in_fairbatch_z1, sort_index_y_1_z1_0[i].copy()))
                key_in_fairbatch_z1 = np.hstack((key_in_fairbatch_z1, sort_index_y_0_z1_1[i].copy()))
                key_in_fairbatch_z1 = np.hstack((key_in_fairbatch_z1, sort_index_y_1_z1_1[i].copy()))

                random.shuffle(key_in_fairbatch_z1)

                yield key_in_fairbatch_z1
                
                key_in_fairbatch_z2 = sort_index_y_0_z2_0[i].copy()
                key_in_fairbatch_z2 = np.hstack((key_in_fairbatch_z2, sort_index_y_1_z2_0[i].copy()))
                key_in_fairbatch_z2 = np.hstack((key_in_fairbatch_z2, sort_index_y_0_z2_1[i].copy()))
                key_in_fairbatch_z2 = np.hstack((key_in_fairbatch_z2, sort_index_y_1_z2_1[i].copy()))

                random.shuffle(key_in_fairbatch_z2)

                yield key_in_fairbatch_z2
                
                key_in_fairbatch_z3 = sort_index_y_0_z3_0[i].copy()
                key_in_fairbatch_z3 = np.hstack((key_in_fairbatch_z3, sort_index_y_1_z3_0[i].copy()))
                key_in_fairbatch_z3 = np.hstack((key_in_fairbatch_z3, sort_index_y_0_z3_1[i].copy()))
                key_in_fairbatch_z3 = np.hstack((key_in_fairbatch_z3, sort_index_y_1_z3_1[i].copy()))

                random.shuffle(key_in_fairbatch_z3)

                yield key_in_fairbatch_z3

        else:
            entire_index = torch.FloatTensor([i for i in range(len(self.y_data))])

            sort_index = self.select_batch_replacement(self.batch_size, entire_index, self.batch_num, self.replacement)

            for i in range(self.batch_num):
                yield sort_index[i]
        
                               

    def __len__(self):
        """Returns the length of data."""
        
        return len(self.y_data)

class FairRobust2(Sampler):
    """FairRobust (Sampler in DataLoader).
    
    This class is for implementing the lambda adjustment and batch selection of FairBatch [Roh et al., ICLR 2021] with robust training.

    Attributes:
        model: A model containing the intermediate states of the training.
        x_, y_, z_data: Tensor-based train data.
        alpha: A positive number for step size that used in the lambda adjustment.
        fairness_type: A string indicating the target fairness type 
                       among original, demographic parity (dp), equal opportunity (eqopp), and equalized odds (eqodds).
        replacement: A boolean indicating whether a batch consists of data with or without replacement.
        N: An integer counting the size of data.
        batch_size: An integer for the size of a batch.
        batch_num: An integer for total number of batches in an epoch.
        y_, z_item: Lists that contains the unique values of the y_data and z_data, respectively.
        yz_tuple: Lists for pairs of y_item and z_item.
        y_, z_, yz_mask: Dictionaries utilizing as array masks.
        y_, z_, yz_index: Dictionaries containing the indexes of each class.
        y_, z_, yz_len: Dictionaries containing the length information.
        clean_index: A list that contains the data indexes of selected samples.
        clean_y_, clean_z_, clean_yz_index: Dictionaries containing the indexes of each class in the selected set.
        clean_y_, clean_z_, clean_yz_len: Dictionaries containing the length information in the selected set.
        S: A dictionary containing the default size of each class in a batch.
        lb1, lb2: (0~1) real numbers indicating the lambda values for fairness [Roh et al., ICLR 2021].
        tau: (0~1) real number indicating the clean ratio of the data.
        warm_start: An integer for warm-start period.

        
    """
    def __init__(self, model, x_tensor, y_tensor, z1_tensor, z2_tensor, z3_tensor, target_fairness, parameters, replacement = False, seed = 0):
        """Initializes FairBatch."""
        
        self.model = model
        
        np.random.seed(seed)
        random.seed(seed)
        
        self.x_data = x_tensor
        self.y_data = y_tensor
        self.z1_data = z1_tensor
        self.z2_data = z2_tensor
        self.z3_data = z3_tensor
        
        #alpha: how to modify rate of each category
        self.alpha = parameters.alpha
        self.fairness_type = target_fairness
        
        self.replacement = replacement
        # N: number of elements
        self.N = len(z1_tensor)
        
        self.batch_size = parameters.batch_size
        self.batch_num = int(len(self.y_data) / self.batch_size)
        
        # Takes the unique values of the tensors
        self.z_item = [tuple((0,0,0)), tuple((0,0,1)), tuple((0,1,0)), tuple((0,1,1)), tuple((1,0,0)), tuple((1,0,1)), tuple((1,1,0)), tuple((1,1,1))]
        self.y_item = list(set(y_tensor.tolist()))
        # yz_tuple: (-1,0), (-1,1), (1,1), (1,0)
        self.yz_tuple = list(itertools.product(self.y_item, self.z_item))
        
        # Makes masks
        #z_mask: z = 1 or z = 0 (list)
        self.z_mask = {}
        #z_mask: y = 1 or y = -1 (list)
        self.y_mask = {}
        self.yz_mask = {}
        
        for tmp_z in self.z_item:
            self.z_mask[tmp_z] = (self.z1_data == tmp_z[0])&(self.z2_data == tmp_z[1])&(self.z3_data == tmp_z[2])
            
        for tmp_y in self.y_item:
            self.y_mask[tmp_y] = (self.y_data == tmp_y)
            
        for tmp_yz in self.yz_tuple:
            self.yz_mask[tmp_yz] = (self.y_data == tmp_yz[0]) & (self.z1_data == tmp_yz[1][0])&(self.z2_data == tmp_yz[1][1])&(self.z3_data == tmp_yz[1][2])
        

        # Finds the index
        self.z_index = {}
        self.y_index = {}
        self.yz_index = {}
        #index of z=1, z=0
        for tmp_z in self.z_item:
            self.z_index[tmp_z] = (self.z_mask[tmp_z] == 1).nonzero().squeeze()
        #index of y=-1, y=0
        for tmp_y in self.y_item:
            self.y_index[tmp_y] = (self.y_mask[tmp_y] == 1).nonzero().squeeze()
        #index of (1,1); (-1,1); (1,0); (-1,0)
        for tmp_yz in self.yz_tuple:
            self.yz_index[tmp_yz] = (self.yz_mask[tmp_yz] == 1).nonzero().squeeze()
            
        self.entire_index = torch.FloatTensor([i for i in range(len(self.y_data))])
            
        # Length information
        self.z_len = {}
        self.y_len = {}
        self.yz_len = {}
        
        for tmp_z in self.z_item:
            self.z_len[tmp_z] = len(self.z_index[tmp_z])
            
        for tmp_y in self.y_item:
            self.y_len[tmp_y] = len(self.y_index[tmp_y])
            
        for tmp_yz in self.yz_tuple:
            self.yz_len[tmp_yz] = len(self.yz_index[tmp_yz])

        # Default batch size
        self.S = {}
        
        # len of each (y,z)
        for tmp_yz in self.yz_tuple:
            self.S[tmp_yz] = self.batch_size * (self.yz_len[tmp_yz])/self.N
            

        # lambda = (y=1,z=1)/(y=1)
        self.lb1_1 = (self.S[1.0, (0, 0, 0)])/(self.y_len[1])
        self.lb1_2 = (self.S[1.0, (0, 0, 1)])/(self.y_len[1])
        self.lb1_3 = (self.S[1.0, (0, 1, 0)])/(self.y_len[1])
        self.lb1_4 = (self.S[1.0, (0, 1, 1)])/(self.y_len[1])
        self.lb1_5 = (self.S[1.0, (1, 0, 0)])/(self.y_len[1])
        self.lb1_6 = (self.S[1.0, (1, 0, 1)])/(self.y_len[1])
        self.lb1_7 = (self.S[1.0, (1, 1, 0)])/(self.y_len[1])
        
        
        # (y=-1,z=1)/(y=-1)
        self.lb2_1 = (self.S[-1.0, (0, 0, 0)])/(self.y_len[-1])
        self.lb2_2 = (self.S[-1.0, (0, 0, 1)])/(self.y_len[-1])
        self.lb2_3 = (self.S[-1.0, (0, 1, 0)])/(self.y_len[-1])
        self.lb2_4 = (self.S[-1.0, (0, 1, 1)])/(self.y_len[-1])
        self.lb2_5 = (self.S[-1.0, (1, 0, 0)])/(self.y_len[-1])
        self.lb2_6 = (self.S[-1.0, (1, 0, 1)])/(self.y_len[-1])
        self.lb2_7 = (self.S[-1.0, (1, 1, 0)])/(self.y_len[-1])
        
        # For cleanselection parameters
        self.tau = parameters.tau # Clean ratio
        self.warm_start = parameters.warm_start
    
        self.count_epoch = 0
        
            
        # Clean sample selection
        self.clean_index = np.arange(0,len(self.y_data))
        
        # Finds the index
        self.clean_z_index = {}
        self.clean_y_index = {}
        self.clean_yz_index = {}
        # index of each z values in list
        for tmp_z in self.z_item:
            self.clean_z_index[tmp_z] = (self.z_mask[tmp_z] == 1)[self.clean_index].nonzero().squeeze()
        # index of each y values in list
        for tmp_y in self.y_item:
            self.clean_y_index[tmp_y] = (self.y_mask[tmp_y] == 1)[self.clean_index].nonzero().squeeze()
        # index of each yz values in list
        for tmp_yz in self.yz_tuple:
            self.clean_yz_index[tmp_yz] = (self.yz_mask[tmp_yz] == 1)[self.clean_index].nonzero().squeeze()
        
       # Length information
        self.clean_z_len = {}
        self.clean_y_len = {}
        self.clean_yz_len = {}
        
        for tmp_z in self.z_item:
            
            self.clean_z_len[tmp_z] = len(self.clean_z_index[tmp_z])
            
        for tmp_y in self.y_item:
            
            self.clean_y_len[tmp_y] = len(self.clean_y_index[tmp_y])
            
        for tmp_yz in self.yz_tuple:
            
            self.clean_yz_len[tmp_yz] = len(self.clean_yz_index[tmp_yz])
 
      
    def get_logit(self):
        """Runs forward pass of the intermediate model with the training data.
        
        Returns:
            Outputs (logits) of the model.

        """
        
        self.model.eval()
        logit = self.model(self.x_data)
        
        return logit
    
    
    def adjust_lambda(self, logit):
        """Adjusts the lambda values using FairBatch [Roh et al., ICLR 2021].
        See our paper for algorithm details.
        
        Args: 
            logit: A torch tensor that contains the intermediate model's output on the training data.
        
        """
        
        criterion = torch.nn.BCELoss(reduction = 'none')
        
        
        if self.fairness_type == 'eqopp':
            yhat_yz = {}
            yhat_y = {}
                        
            eo_loss = criterion ((F.tanh(logit.squeeze())+1)/2, (self.y_data.squeeze()+1)/2)
            
            for tmp_yz in self.yz_tuple:
                
                yhat_yz[tmp_yz] = float(torch.sum(eo_loss[self.clean_yz_index[tmp_yz]])) / self.clean_yz_len[tmp_yz]
                
            for tmp_y in self.y_item:
                yhat_y[tmp_y] = float(torch.sum(eo_loss[self.clean_y_index[tmp_y]])) / self.clean_y_len[tmp_y]
            
            # lb1 * loss_z1 + (1-lb1) * loss_z0
            summ_1 = 0
            for tmp_yz in self.yz_tuple:
                if tmp_yz[0] == 1:
                    summ_1 += yhat_yz[tmp_yz]
            avg = summ_1/8

            if yhat_yz[(1.0, (0, 0, 0))] > avg:
                self.lb1_1 += self.alpha
                if self.lb1_1 > 1:
                    self.lb1_1 = 1
            else:
                self.lb1_1 -= self.alpha
                if self.lb1_1 < 0:
                    self.lb1_1 = 0
            
            if yhat_yz[(1.0, (0, 0, 1))] > avg:
                self.lb1_2 += self.alpha
                if self.lb1_2 > 1:
                    self.lb1_2 = 1
            else:
                self.lb1_2 -= self.alpha
                if self.lb1_2 < 0:
                    self.lb1_2 = 0
            
            if yhat_yz[(1.0, (0, 1, 0))] > avg:
                
                self.lb1_3 += self.alpha
                if self.lb1_3 > 1:
                    self.lb1_3 = 1
            else:
                self.lb1_3 -= self.alpha
                
                if self.lb1_3 < 0:
                    self.lb1_3 = 0
            
            if yhat_yz[(1.0, (0, 1, 1))] > avg:
                self.lb1_4 += self.alpha
                if self.lb1_4 > 1:
                    self.lb1_4 = 1
            else:
                self.lb1_4 -= self.alpha
                if self.lb1_4 < 0:
                    self.lb1_4 = 0
            
            if yhat_yz[(1.0, (1, 0, 0))] > avg:
                self.lb1_5 += self.alpha
                if self.lb1_5 > 1:
                    self.lb1_5 = 1
            else:
                self.lb1_5 -= self.alpha
                if self.lb1_5 < 0:
                    self.lb1_5 = 0
            
            if yhat_yz[(1.0, (1, 0, 1))] > avg:
                self.lb1_6 += self.alpha
                if self.lb1_6 > 1:
                    self.lb1_6 = 1
            else:
                self.lb1_6 -= self.alpha
                if self.lb1_6 < 0:
                    self.lb1_6 = 0
            
            if yhat_yz[(1.0, (1, 1, 0))] > avg:
                self.lb1_7 += self.alpha
                if self.lb1_7 > 1:
                    self.lb1_7 = 1
            else:
                self.lb1_7 -= self.alpha
                if self.lb1_7 < 0:
                    self.lb1_7 = 0
                
            if self.lb1_1 + self.lb1_2 + self.lb1_3 + self.lb1_4 + self.lb1_5 + self.lb1_6 + self.lb1_7 > 1:
                minus = (self.lb1_1 + self.lb1_2 + self.lb1_3 + self.lb1_4 + self.lb1_5 + self.lb1_6 + self.lb1_7 - 1)/7
                self.lb1_1 -= minus
                self.lb1_2 -= minus
                self.lb1_3 -= minus
                self.lb1_4 -= minus
                self.lb1_5 -= minus
                self.lb1_6 -= minus
                self.lb1_7 -= minus
            
            elif self.lb1_1 + self.lb1_2 + self.lb1_3 + self.lb1_4 + self.lb1_5 + self.lb1_6 + self.lb1_7 < 0:
                plus = (self.lb1_1 + self.lb1_2 + self.lb1_3 + self.lb1_4 + self.lb1_5 + self.lb1_6 + self.lb1_7 - 0)/7
                self.lb1_1 -= plus
                self.lb1_2 -= plus
                self.lb1_3 -= plus
                self.lb1_4 -= plus
                self.lb1_5 -= plus
                self.lb1_6 -= plus
                self.lb1_7 -= plus
                
        elif self.fairness_type == 'eqodds':
            
            yhat_yz = {}
            yhat_y = {}
                        
            eo_loss = criterion ((F.tanh(logit.squeeze())+1)/2, (self.y_data.squeeze()+1)/2)
            
            for tmp_yz in self.yz_tuple:
                yhat_yz[tmp_yz] = float(torch.sum(eo_loss[self.clean_yz_index[tmp_yz]])) / (self.clean_yz_len[tmp_yz]+1)
                
                
            for tmp_y in self.y_item:
                yhat_y[tmp_y] = float(torch.sum(eo_loss[self.clean_y_index[tmp_y]])) / (self.clean_y_len[tmp_y]+1)
            summ_1 = 0
            
            for tmp_yz in self.yz_tuple:
                if tmp_yz[0] == 1:
                    
                    summ_1 += yhat_yz[tmp_yz]
            avg_1 = summ_1/8
            summ_2 = 0
            for tmp_yz in self.yz_tuple:
                if tmp_yz[0] == -1:
                    summ_2 += yhat_yz[tmp_yz]
            avg_2 = summ_2/8
            std_1 = 0
            for tmp_yz in self.yz_tuple:
                if tmp_yz[0] == 1:
                    std_1 += abs(yhat_yz[tmp_yz]-avg_1)
            std_1 /= 8
            std_2 = 0
            for tmp_yz in self.yz_tuple:
                if tmp_yz[0] == -1:
                    std_2 += abs(yhat_yz[tmp_yz]-avg_2)
            std_2 /= 8
#             y1_diff = abs(yhat_yz[(1, 1)] - yhat_yz[(1, 0)])
#             y0_diff = abs(yhat_yz[(-1, 1)] - yhat_yz[(-1, 0)])
            
            # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
            # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0
            
            if std_1 > std_2:
                if yhat_yz[(1.0, (0, 0, 0))] > avg_1:
                    self.lb1_1 += self.alpha
                    if self.lb1_1 > 1:
                        self.lb1_1 = 1
                else:
                    self.lb1_1 -= self.alpha
                    if self.lb1_1 < 0:
                        self.lb1_1 = 0

                if yhat_yz[(1.0, (0, 0, 1))] > avg_1:
                    self.lb1_2 += self.alpha
                    if self.lb1_2 > 1:
                        self.lb1_2 = 1
                else:
                    self.lb1_2 -= self.alpha
                    if self.lb1_2 < 0:
                        self.lb1_2 = 0

                if yhat_yz[(1.0, (0, 1, 0))] > avg_1:
                    self.lb1_3 += self.alpha
                    if self.lb1_3 > 1:
                        self.lb1_3 = 1
                else:
                    self.lb1_3 -= self.alpha
                    if self.lb1_3 < 0:
                        self.lb1_3 = 0

                if yhat_yz[(1.0, (0, 1, 1))] > avg_1:
                    self.lb1_4 += self.alpha
                    if self.lb1_4 > 1:
                        self.lb1_4 = 1
                else:
                    self.lb1_4 -= self.alpha
                    if self.lb1_4 < 0:
                        self.lb1_4 = 0

                if yhat_yz[(1.0, (1, 0, 0))] > avg_1:
                    self.lb1_5 += self.alpha
                    if self.lb1_5 > 1:
                        self.lb1_5 = 1
                else:
                    self.lb1_5 -= self.alpha
                    if self.lb1_5 < 0:
                        self.lb1_5 = 0

                if yhat_yz[(1.0, (1, 0, 1))] > avg_1:
                    self.lb1_6 += self.alpha
                    if self.lb1_6 > 1:
                        self.lb1_6 = 1
                else:
                    self.lb1_6 -= self.alpha
                    if self.lb1_6 < 0:
                        self.lb1_6 = 0

                if yhat_yz[(1.0, (1, 1, 0))] > avg_1:
                    self.lb1_7 += self.alpha
                    if self.lb1_7 > 1:
                        self.lb1_7 = 1
                else:
                    self.lb1_7 -= self.alpha
                    if self.lb1_7 < 0:
                        self.lb1_7 = 0
                    # if yhat_yz[(1, 1)] > yhat_yz[(1, 0)]:
                    #     self.lb1 += self.alpha
                    # else:
                    #     self.lb1 -= self.alpha
            else:
                if yhat_yz[(-1.0, (0, 0, 0))] > avg_2:
                    self.lb2_1 += self.alpha
                    if self.lb2_1 > 1:
                        self.lb2_1 = 1
                else:
                    self.lb2_1 -= self.alpha
                    if self.lb2_1 < 0:
                        self.lb2_1 = 0

                if yhat_yz[(-1.0, (0, 0, 1))] > avg_2:
                    self.lb2_2 += self.alpha
                    if self.lb2_2 > 1:
                        self.lb2_2 = 1
                else:
                    self.lb2_2 -= self.alpha
                    if self.lb2_2 < 0:
                        self.lb2_2 = 0

                if yhat_yz[(-1.0, (0, 1, 0))] > avg_2:
                    self.lb2_3 += self.alpha
                    if self.lb2_3 > 1:
                        self.lb2_3 = 1
                else:
                    self.lb2_3 -= self.alpha
                    if self.lb2_3 < 0:
                        self.lb2_3 = 0

                if yhat_yz[(-1.0, (0, 1, 1))] > avg_2:
                    self.lb2_4 += self.alpha
                    if self.lb2_4 > 1:
                        self.lb2_4 = 1
                else:
                    self.lb2_4 -= self.alpha
                    if self.lb2_4 < 0:
                        self.lb2_4 = 0

                if yhat_yz[(-1.0, (1, 0, 0))] > avg_2:
                    self.lb2_5 += self.alpha
                    if self.lb2_5 > 1:
                        self.lb2_5 = 1
                else:
                    self.lb2_5 -= self.alpha
                    if self.lb2_5 < 0:
                        self.lb2_5 = 0

                if yhat_yz[(-1.0, (1, 0, 1))] > avg_2:
                    self.lb2_6 += self.alpha
                    if self.lb2_6 > 1:
                        self.lb2_6 = 1
                else:
                    self.lb2_6 -= self.alpha
                    if self.lb2_6 < 0:
                        self.lb2_6 = 0

                if yhat_yz[(-1.0, (1, 1, 0))] > avg_2:
                    self.lb2_7 += self.alpha
                    if self.lb2_7 > 1:
                        self.lb2_7 = 1
                else:
                    self.lb2_7 -= self.alpha
                    if self.lb2_7 < 0:
                        
                        self.lb2_7 = 0
                    
                
            if self.lb1_1 + self.lb1_2 + self.lb1_3 + self.lb1_4 + self.lb1_5 + self.lb1_6 + self.lb1_7 > 1:
                
                minus = (self.lb1_1 + self.lb1_2 + self.lb1_3 + self.lb1_4 + self.lb1_5 + self.lb1_6 + self.lb1_7 - 1)/7
                self.lb1_1 -= minus
                self.lb1_2 -= minus
                self.lb1_3 -= minus
                self.lb1_4 -= minus
                self.lb1_5 -= minus
                self.lb1_6 -= minus
                self.lb1_7 -= minus
            
            elif self.lb1_1 + self.lb1_2 + self.lb1_3 + self.lb1_4 + self.lb1_5 + self.lb1_6 + self.lb1_7 < 0:
                plus = (self.lb1_1 + self.lb1_2 + self.lb1_3 + self.lb1_4 + self.lb1_5 + self.lb1_6 + self.lb1_7 - 0)/7
                self.lb1_1 -= plus
                self.lb1_2 -= plus
                self.lb1_3 -= plus
                self.lb1_4 -= plus
                self.lb1_5 -= plus
                self.lb1_6 -= plus
                self.lb1_7 -= plus
            
            if self.lb2_1 + self.lb2_2 + self.lb2_3 + self.lb2_4 + self.lb2_5 + self.lb2_6 + self.lb2_7 > 1:
                minus = (self.lb2_1 + self.lb2_2 + self.lb2_3 + self.lb2_4 + self.lb2_5 + self.lb2_6 + self.lb2_7 - 1)/7
                self.lb2_1 -= minus
                self.lb2_2 -= minus
                self.lb2_3 -= minus
                self.lb2_4 -= minus
                self.lb2_5 -= minus
                self.lb2_6 -= minus
                self.lb2_7 -= minus
            
            elif self.lb2_1 + self.lb2_2 + self.lb2_3 + self.lb2_4 + self.lb2_5 + self.lb2_6 + self.lb2_7 < 0:
                plus = (self.lb2_1 + self.lb2_2 + self.lb2_3 + self.lb2_4 + self.lb2_5 + self.lb2_6 + self.lb2_7 - 0)/7
                self.lb2_1 -= plus
                self.lb2_2 -= plus
                self.lb2_3 -= plus
                self.lb2_4 -= plus
                self.lb2_5 -= plus
                self.lb2_6 -= plus
                self.lb2_7 -= plus
                
        elif self.fairness_type == 'dp':
            yhat_yz = {}
            yhat_y = {}
            
            ones_array = np.ones(len(self.y_data))
            ones_tensor = torch.FloatTensor(ones_array).cuda()
            dp_loss = criterion((F.tanh(logit.squeeze())+1)/2, ones_tensor.squeeze()) # Note that ones tensor puts as the true label
            
            for tmp_yz in self.yz_tuple:
                
                yhat_yz[tmp_yz] = float(torch.sum(dp_loss[self.clean_yz_index[tmp_yz]])) / self.clean_z_len[tmp_yz[1]]
                    
            
            summ_1 = 0
            for tmp_yz in self.yz_tuple:
                if tmp_yz[0] == 1:
                    summ_1 += yhat_yz[tmp_yz]
            avg_1 = summ_1/8
            summ_2 = 0
            for tmp_yz in self.yz_tuple:
                if tmp_yz[0] == -1:
                    summ_2 += yhat_yz[tmp_yz]
            avg_2 = summ_2/8
            std_1 = 0
            for tmp_yz in self.yz_tuple:
                if tmp_yz[0] == 1:
                    std_1 += abs(yhat_yz[tmp_yz]-avg_1)
            std_1 /= 8
            std_2 = 0
            for tmp_yz in self.yz_tuple:
                if tmp_yz[0] == -1:
                    std_2 += abs(yhat_yz[tmp_yz]-avg_2)
            std_2 /= 8
#             y1_diff = abs(yhat_yz[(1, 1)] - yhat_yz[(1, 0)])
#             y0_diff = abs(yhat_yz[(-1, 1)] - yhat_yz[(-1, 0)])
            
            # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
            # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0
            
            if std_1 > std_2:
                if yhat_yz[(1.0, (0, 0, 0))] > avg_1:
                    self.lb1_1 += self.alpha
                    if self.lb1_1 > 1:
                        self.lb1_1 = 1
                else:
                    self.lb1_1 -= self.alpha
                    if self.lb1_1 < 0:
                        self.lb1_1 = 0

                if yhat_yz[(1.0, (0, 0, 1))] > avg_1:
                    self.lb1_2 += self.alpha
                    if self.lb1_2 > 1:
                        self.lb1_2 = 1
                else:
                    self.lb1_2 -= self.alpha
                    if self.lb1_2 < 0:
                        self.lb1_2 = 0

                if yhat_yz[(1.0, (0, 1, 0))] > avg_1:
                    self.lb1_3 += self.alpha
                    if self.lb1_3 > 1:
                        self.lb1_3 = 1
                else:
                    self.lb1_3 -= self.alpha
                    if self.lb1_3 < 0:
                        self.lb1_3 = 0

                if yhat_yz[(1.0, (0, 1, 1))] > avg_1:
                    self.lb1_4 += self.alpha
                    if self.lb1_4 > 1:
                        self.lb1_4 = 1
                else:
                    self.lb1_4 -= self.alpha
                    if self.lb1_4 < 0:
                        self.lb1_4 = 0

                if yhat_yz[(1.0, (1, 0, 0))] > avg_1:
                    self.lb1_5 += self.alpha
                    if self.lb1_5 > 1:
                        self.lb1_5 = 1
                else:
                    self.lb1_5 -= self.alpha
                    if self.lb1_5 < 0:
                        self.lb1_5 = 0

                if yhat_yz[(1.0, (1, 0, 1))] > avg_1:
                    self.lb1_6 += self.alpha
                    if self.lb1_6 > 1:
                        self.lb1_6 = 1
                else:
                    self.lb1_6 -= self.alpha
                    if self.lb1_6 < 0:
                        self.lb1_6 = 0

                if yhat_yz[(1.0, (1, 1, 0))] > avg_1:
                    self.lb1_7 += self.alpha
                    if self.lb1_7 > 1:
                        self.lb1_7 = 1
                else:
                    self.lb1_7 -= self.alpha
                    if self.lb1_7 < 0:
                        self.lb1_7 = 0
                    # if yhat_yz[(1, 1)] > yhat_yz[(1, 0)]:
                    #     self.lb1 += self.alpha
                    # else:
                    #     self.lb1 -= self.alpha
            else:
                if yhat_yz[(-1.0, (0, 0, 0))] > avg_2:
                    self.lb2_1 += self.alpha
                    if self.lb2_1 > 1:
                        self.lb2_1 = 1
                else:
                    self.lb2_1 -= self.alpha
                    if self.lb2_1 < 0:
                        self.lb2_1 = 0

                if yhat_yz[(-1.0, (0, 0, 1))] > avg_2:
                    self.lb2_2 += self.alpha
                    if self.lb2_2 > 1:
                        self.lb2_2 = 1
                else:
                    self.lb2_2 -= self.alpha
                    if self.lb2_2 < 0:
                        self.lb2_2 = 0

                if yhat_yz[(-1.0, (0, 1, 0))] > avg_2:
                    self.lb2_3 += self.alpha
                    if self.lb2_3 > 1:
                        self.lb2_3 = 1
                else:
                    self.lb2_3 -= self.alpha
                    if self.lb2_3 < 0:
                        self.lb2_3 = 0

                if yhat_yz[(-1.0, (0, 1, 1))] > avg_2:
                    self.lb2_4 += self.alpha
                    if self.lb2_4 > 1:
                        self.lb2_4 = 1
                else:
                    self.lb2_4 -= self.alpha
                    if self.lb2_4 < 0:
                        self.lb2_4 = 0

                if yhat_yz[(-1.0, (1, 0, 0))] > avg_2:
                    self.lb2_5 += self.alpha
                    if self.lb2_5 > 1:
                        self.lb2_5 = 1
                else:
                    self.lb2_5 -= self.alpha
                    if self.lb2_5 < 0:
                        self.lb2_5 = 0

                if yhat_yz[(-1.0, (1, 0, 1))] > avg_2:
                    self.lb2_6 += self.alpha
                    if self.lb2_6 > 1:
                        self.lb2_6 = 1
                else:
                    self.lb2_6 -= self.alpha
                    if self.lb2_6 < 0:
                        self.lb2_6 = 0

                if yhat_yz[(-1.0, (1, 1, 0))] > avg_2:
                    self.lb2_7 += self.alpha
                    if self.lb2_7 > 1:
                        self.lb2_7 = 1
                else:
                    self.lb2_7 -= self.alpha
                    if self.lb2_7 < 0:
                        self.lb2_7 = 0
#                     if yhat_yz[(1, 1)] > yhat_yz[(1, 0)]:
#                         self.lb1 += self.alpha
#                     else:
#                         self.lb1 -= self.alpha
                    
                
            if self.lb1_1 + self.lb1_2 + self.lb1_3 + self.lb1_4 + self.lb1_5 + self.lb1_6 + self.lb1_7 > 1:
                minus = (self.lb1_1 + self.lb1_2 + self.lb1_3 + self.lb1_4 + self.lb1_5 + self.lb1_6 + self.lb1_7 - 1)/7
                self.lb1_1 -= minus
                self.lb1_2 -= minus
                self.lb1_3 -= minus
                self.lb1_4 -= minus
                self.lb1_5 -= minus
                self.lb1_6 -= minus
                self.lb1_7 -= minus
            
            elif self.lb1_1 + self.lb1_2 + self.lb1_3 + self.lb1_4 + self.lb1_5 + self.lb1_6 + self.lb1_7 < 0:
                plus = (self.lb1_1 + self.lb1_2 + self.lb1_3 + self.lb1_4 + self.lb1_5 + self.lb1_6 + self.lb1_7 - 0)/7
                self.lb1_1 -= plus
                self.lb1_2 -= plus
                self.lb1_3 -= plus
                self.lb1_4 -= plus
                self.lb1_5 -= plus
                self.lb1_6 -= plus
                self.lb1_7 -= plus
            
            if self.lb2_1 + self.lb2_2 + self.lb2_3 + self.lb2_4 + self.lb2_5 + self.lb2_6 + self.lb2_7 > 1:
                
                minus = (self.lb2_1 + self.lb2_2 + self.lb2_3 + self.lb2_4 + self.lb2_5 + self.lb2_6 + self.lb2_7 - 1)/7
                self.lb2_1 -= minus
                self.lb2_2 -= minus
                self.lb2_3 -= minus
                self.lb2_4 -= minus
                self.lb2_5 -= minus
                self.lb2_6 -= minus
                self.lb2_7 -= minus
            
            elif self.lb2_1 + self.lb2_2 + self.lb2_3 + self.lb2_4 + self.lb2_5 + self.lb2_6 + self.lb2_7 < 0:
                plus = (self.lb2_1 + self.lb2_2 + self.lb2_3 + self.lb2_4 + self.lb2_5 + self.lb2_6 + self.lb2_7 - 0)/7
                self.lb2_1 -= plus
                self.lb2_2 -= plus
                self.lb2_3 -= plus
                self.lb2_4 -= plus
                self.lb2_5 -= plus
                self.lb2_6 -= plus
                self.lb2_7 -= plus

    def select_fair_robust_sample(self):
        """Selects fair and robust samples and adjusts the lambda values for fairness. 
        See our paper for algorithm details.
        
        Returns:
            Indexes that indicate the data.
            
        """
        
        logit = self.get_logit()
        
        self.adjust_lambda(logit)

        criterion = torch.nn.BCELoss(reduction = 'none')
        
        loss = criterion ((F.tanh(logit.squeeze())+1)/2, (self.y_data.squeeze()+1)/2)
        profit = torch.max(loss)-loss
        
        current_weight_sum = {}
        
        lb_ratio = {}
        for tmp_yz in self.yz_tuple:
            if tmp_yz == (1.0, (0, 0, 0)):
                
                lb_ratio[tmp_yz] = self.lb1_1                
            elif tmp_yz == (1.0, (0, 0, 1)): 
                            
                lb_ratio[tmp_yz] = self.lb1_2
            elif tmp_yz == (1.0, (0, 1, 0)): 
                lb_ratio[tmp_yz] = self.lb1_3
            elif tmp_yz == (1.0, (0, 1, 1)): 
                lb_ratio[tmp_yz] = self.lb1_4
            elif tmp_yz == (1.0, (1, 0, 0)):
                lb_ratio[tmp_yz] = self.lb1_5
            elif tmp_yz == (1.0, (1, 0, 1)):
                            
                lb_ratio[tmp_yz] = self.lb1_6
            elif tmp_yz == (1.0, (1, 1, 0)):
                            
                lb_ratio[tmp_yz] = self.lb1_7
            elif tmp_yz == (1.0, (1, 1, 1)):
                lb_ratio[tmp_yz] = 1 - (self.lb1_1 + self.lb1_2 + self.lb1_3 + self.lb1_4 + self.lb1_5 + self.lb1_6 + self.lb1_7)
            elif tmp_yz == (-1.0, (0, 0, 0)):
                            
                lb_ratio[tmp_yz] = self.lb2_1
            elif tmp_yz == (-1.0, (0, 0, 1)): 
                lb_ratio[tmp_yz] = self.lb2_2
            elif tmp_yz == (-1.0, (0, 1, 0)): 
                lb_ratio[tmp_yz] = self.lb2_3
            elif tmp_yz == (-1.0, (0, 1, 1)): 
                lb_ratio[tmp_yz] = self.lb2_4
            elif tmp_yz == (-1.0, (1, 0, 0)): 
                lb_ratio[tmp_yz] = self.lb2_5
            elif tmp_yz == (-1.0, (1, 0, 1)): 
                lb_ratio[tmp_yz] = self.lb2_6
            elif tmp_yz == (-1.0, (1, 1, 0)): 
                lb_ratio[tmp_yz] = self.lb2_7
            elif tmp_yz == (-1.0, (1, 1, 1)):
                            
                lb_ratio[tmp_yz] = 1 - (self.lb2_1 + self.lb2_2 + self.lb2_3 + self.lb2_4 + self.lb2_5 + self.lb2_6 + self.lb2_7)
                            
            current_weight_sum[tmp_yz] = 0
        (_, sorted_index) = torch.topk(profit, len(profit), largest=True, sorted=True)
        
        clean_index = []
        
        total_selected = 0
        
        desired_size = int(self.tau * len(self.y_data))
        
        for j in sorted_index:
            tmp_y = self.y_data[j].item()
            tmp_z = tuple((self.z1_data[j].item(), self.z2_data[j].item(), self.z3_data[j].item()))
            current_weight_list = list(current_weight_sum.values())
            
            if total_selected >= desired_size:
                break
            if all(i < desired_size for i in current_weight_list):
                clean_index.append(j)
                for tmp_yz in self.yz_tuple:
                    if tmp_yz[0] != tmp_y:
                        current_weight_sum[tmp_yz] += 1
                    elif tmp_yz[1] != tmp_z:
                        current_weight_sum[tmp_yz] += 1 - lb_ratio[tmp_yz]
                    else:
                        current_weight_sum[tmp_yz] += 2 - lb_ratio[tmp_yz]
                            
                            
                total_selected += 1        
        
        clean_index = torch.LongTensor(clean_index).cuda()
        
        self.batch_num = int(len(clean_index)/self.batch_size)
        
        # Update the variables
        self.clean_index = clean_index
        
        for tmp_z in self.z_item:
            combined = torch.cat((self.z_index[tmp_z], self.clean_index))
            uniques, counts = combined.unique(return_counts=True)
            intersection = uniques[counts > 1]
            
            self.clean_z_index[tmp_z] = intersection 
            
        for tmp_y in self.y_item:
            combined = torch.cat((self.y_index[tmp_y], self.clean_index))
            uniques, counts = combined.unique(return_counts=True)
            intersection = uniques[counts > 1]
            
            self.clean_y_index[tmp_y] = intersection
        
        for tmp_yz in self.yz_tuple:
            combined = torch.cat((self.yz_index[tmp_yz], self.clean_index))
            uniques, counts = combined.unique(return_counts=True)
            intersection = uniques[counts > 1]
            
            self.clean_yz_index[tmp_yz] = intersection
        
        
        for tmp_z in self.z_item:
            self.clean_z_len[tmp_z] = len(self.clean_z_index[tmp_z])
            
        for tmp_y in self.y_item:
            self.clean_y_len[tmp_y] = len(self.clean_y_index[tmp_y])
        
        for tmp_yz in self.yz_tuple:
            self.clean_yz_len[tmp_yz] = len(self.clean_yz_index[tmp_yz])
            
        
        return clean_index
        
    
    def select_batch_replacement(self, batch_size, full_index, batch_num, replacement = False, weight = None):
        """Selects a certain number of batches based on the given batch size.
        
        Args: 
            batch_size: An integer for the data size in a batch.
            full_index: An array containing the candidate data indices.
            batch_num: An integer indicating the number of batches.
            replacement: A boolean indicating whether a batch consists of data with or without replacement.
        
        Returns:
            Indexes that indicate the data.
            
        """
        
        select_index = []
        
        if replacement == True:
            for _ in range(batch_num):
                if weight == None:
                    weight_norm = weight/torch.sum(weight)
                    select_index.append(np.random.choice(full_index, batch_size, replace = False, p = weight_norm))
                else:
                    select_index.append(np.random.choice(full_index, batch_size, replace = False))
        else:
            tmp_index = full_index.detach().cpu().numpy().copy()
            random.shuffle(tmp_index)
            
            start_idx = 0
            for i in range(batch_num):
                if start_idx + batch_size > len(full_index):
                    select_index.append(np.concatenate((tmp_index[start_idx:], tmp_index[ : batch_size - (len(full_index)-start_idx)])))
                    
                    start_idx = len(full_index)-start_idx
                else:

                    select_index.append(tmp_index[start_idx:start_idx + batch_size])
                    start_idx += batch_size
            
        return select_index


    
    def decide_fair_batch_size(self):
        """Calculates each class size based on the lambda values (lb1 and lb2) for fairness.
        
        Returns:
            Each class size for fairness.
            
        """
        
        each_size = {}
        summ_1 = 0
        summ_2 = 0

        for tmp_yz in self.yz_tuple:
            self.S[tmp_yz] = self.batch_size * (self.clean_yz_len[tmp_yz])/len(self.clean_index)
            if tmp_yz[0] == 1:
                summ_1 += self.S[tmp_yz]
            else:
                summ_2 += self.S[tmp_yz]

        # Based on the updated lambdas, determine the size of each class in a batch
        if self.fairness_type == 'eqopp':
            each_size[(1.0, (0, 0, 0))] = round(self.lb1_1 * summ_1)
            each_size[(1.0, (0, 0, 1))] = round(self.lb1_2 * summ_1)
            each_size[(1.0, (0, 1, 0))] = round(self.lb1_3 * summ_1)
            each_size[(1.0, (0, 1, 1))] = round(self.lb1_4 * summ_1)
            each_size[(1.0, (1, 0, 0))] = round(self.lb1_5 * summ_1)
            each_size[(1.0, (1, 0, 1))] = round(self.lb1_6 * summ_1)
            each_size[(1.0, (1, 1, 0))] = round(self.lb1_7 * summ_1)
            each_size[(1.0, (1, 1, 1))] = round((1 - (self.lb1_1 + self.lb1_2 + self.lb1_3 + self.lb1_4 + self.lb1_5 + self.lb1_6 + self.lb1_7))*summ_1)
            each_size[(-1.0, (0, 0, 0))] = round(self.S[(-1.0, (0, 0, 0))])
            each_size[(-1.0, (0, 0, 1))] = round(self.S[(-1.0, (0, 0, 1))])
            each_size[(-1.0, (0, 1, 0))] = round(self.S[(-1.0, (0, 1, 0))])
            each_size[(-1.0, (0, 1, 1))] = round(self.S[(-1.0, (0, 1, 1))])
            each_size[(-1.0, (1, 0, 0))] = round(self.S[(-1.0, (1, 0, 0))])
            each_size[(-1.0, (1, 0, 1))] = round(self.S[(-1.0, (1, 0, 1))])
            each_size[(-1.0, (1, 1, 0))] = round(self.S[(-1.0, (1, 1, 0))])
            each_size[(-1.0, (1, 1, 1))] = round(self.S[(-1.0, (1, 1, 1))])
#             each_size[(-1,1)] = round(self.S[(-1,1)])
#             each_size[(-1,0)] = round(self.S[(-1,0)])
                            
            

        elif self.fairness_type == 'eqodds':

            each_size[(1.0, (0, 0, 0))] = round(self.lb1_1 * summ_1)
            each_size[(1.0, (0, 0, 1))] = round(self.lb1_2 * summ_1)
            each_size[(1.0, (0, 1, 0))] = round(self.lb1_3 * summ_1)
            each_size[(1.0, (0, 1, 1))] = round(self.lb1_4 * summ_1)
            each_size[(1.0, (1, 0, 0))] = round(self.lb1_5 * summ_1)
            each_size[(1.0, (1, 0, 1))] = round(self.lb1_6 * summ_1)
            each_size[(1.0, (1, 1, 0))] = round(self.lb1_7 * summ_1)
            each_size[(1.0, (1, 1, 1))] = round((1 - (self.lb1_1 + self.lb1_2 + self.lb1_3 + self.lb1_4 + self.lb1_5 + self.lb1_6 + self.lb1_7))*summ_1)
            each_size[(-1.0, (0, 0, 0))] = round(self.lb2_1 * summ_2)
            each_size[(-1.0, (0, 0, 1))] = round(self.lb2_2 * summ_2)
            each_size[(-1.0, (0, 1, 0))] = round(self.lb2_3 * summ_2)
            each_size[(-1.0, (0, 1, 1))] = round(self.lb2_4 * summ_2)
            each_size[(-1.0, (1, 0, 0))] = round(self.lb2_5 * summ_2)
            each_size[(-1.0, (1, 0, 1))] = round(self.lb2_6 * summ_2)
            each_size[(-1.0, (1, 1, 0))] = round(self.lb2_7 * summ_2)
            each_size[(-1.0, (1, 1, 1))] = round((1 - (self.lb2_1 + self.lb2_2 + self.lb2_3 + self.lb2_4 + self.lb1_5 + self.lb2_6 + self.lb1_7))*summ_2)
        elif self.fairness_type == 'dp':

            
            # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
            # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0
            each_size[(1.0, (0, 0, 0))] = round(self.lb1_1 * summ_1)
            each_size[(1.0, (0, 0, 1))] = round(self.lb1_2 * summ_1)
            each_size[(1.0, (0, 1, 0))] = round(self.lb1_3 * summ_1)
            each_size[(1.0, (0, 1, 1))] = round(self.lb1_4 * summ_1)
            each_size[(1.0, (1, 0, 0))] = round(self.lb1_5 * summ_1)
            each_size[(1.0, (1, 0, 1))] = round(self.lb1_6 * summ_1)
            each_size[(1.0, (1, 1, 0))] = round(self.lb1_7 * summ_1)
            each_size[(1.0, (1, 1, 1))] = round((1 - (self.lb1_1 + self.lb1_2 + self.lb1_3 + self.lb1_4 + self.lb1_5 + self.lb1_6 + self.lb1_7))*summ_1)
            each_size[(-1.0, (0, 0, 0))] = round(self.lb2_1 * summ_2)
            each_size[(-1.0, (0, 0, 1))] = round(self.lb2_2 * summ_2)
            each_size[(-1.0, (0, 1, 0))] = round(self.lb2_3 * summ_2)
            each_size[(-1.0, (0, 1, 1))] = round(self.lb2_4 * summ_2)
            each_size[(-1.0, (1, 0, 0))] = round(self.lb2_5 * summ_2)
            each_size[(-1.0, (1, 0, 1))] = round(self.lb2_6 * summ_2)
            each_size[(-1.0, (1, 1, 0))] = round(self.lb2_7 * summ_2)
            each_size[(-1.0, (1, 1, 1))] = round((1 - (self.lb2_1 + self.lb2_2 + self.lb2_3 + self.lb2_4 + self.lb1_5 + self.lb2_6 + self.lb1_7))*summ_2)        
        return each_size
        
        
    
    def __iter__(self):
        """Iters the full process of fair and robust sample selection for serving the batches to training.
        
        Returns:
            Indexes that indicate the data in each batch.
            
        """
        self.count_epoch += 1
        
        if self.count_epoch > self.warm_start:

            _ = self.select_fair_robust_sample()


            each_size = self.decide_fair_batch_size()
            sort_index_y_1_z_000 = self.select_batch_replacement(each_size[(1.0, (0, 0, 0))], self.clean_yz_index[(1.0, (0, 0, 0))], self.batch_num, self.replacement)
            sort_index_y_1_z_001 = self.select_batch_replacement(each_size[(1.0, (0, 0, 1))], self.clean_yz_index[(1.0, (0, 0, 1))], self.batch_num, self.replacement)
            sort_index_y_1_z_010 = self.select_batch_replacement(each_size[(1.0, (0, 1, 0))], self.clean_yz_index[(1.0, (0, 1, 0))], self.batch_num, self.replacement)
            sort_index_y_1_z_011 = self.select_batch_replacement(each_size[(1.0, (0, 1, 1))], self.clean_yz_index[(1.0, (0, 1, 1))], self.batch_num, self.replacement)
            sort_index_y_1_z_100 = self.select_batch_replacement(each_size[(1.0, (1, 0, 0))], self.clean_yz_index[(1.0, (1, 0, 0))], self.batch_num, self.replacement)
            sort_index_y_1_z_101 = self.select_batch_replacement(each_size[(1.0, (1, 0, 1))], self.clean_yz_index[(1.0, (1, 0, 1))], self.batch_num, self.replacement)
            sort_index_y_1_z_110 = self.select_batch_replacement(each_size[(1.0, (1, 1, 0))], self.clean_yz_index[(1.0, (1, 1, 0))], self.batch_num, self.replacement)
            sort_index_y_1_z_111 = self.select_batch_replacement(each_size[(1.0, (1, 1, 1))], self.clean_yz_index[(1.0, (1, 1, 1))], self.batch_num, self.replacement)
                            
            sort_index_y_0_z_000 = self.select_batch_replacement(each_size[(-1.0, (0, 0, 0))], self.clean_yz_index[(-1.0, (0, 0, 0))], self.batch_num, self.replacement)
            sort_index_y_0_z_001 = self.select_batch_replacement(each_size[(-1.0, (0, 0, 1))], self.clean_yz_index[(-1.0, (0, 0, 1))], self.batch_num, self.replacement)
            sort_index_y_0_z_010 = self.select_batch_replacement(each_size[(-1.0, (0, 1, 0))], self.clean_yz_index[(-1.0, (0, 1, 0))], self.batch_num, self.replacement)
            sort_index_y_0_z_011 = self.select_batch_replacement(each_size[(-1.0, (0, 1, 1))], self.clean_yz_index[(-1.0, (0, 1, 1))], self.batch_num, self.replacement)
            sort_index_y_0_z_100 = self.select_batch_replacement(each_size[(-1.0, (1, 0, 0))], self.clean_yz_index[(-1.0, (1, 0, 0))], self.batch_num, self.replacement)
            sort_index_y_0_z_101 = self.select_batch_replacement(each_size[(-1.0, (1, 0, 1))], self.clean_yz_index[(-1.0, (1, 0, 1))], self.batch_num, self.replacement)
            sort_index_y_0_z_110 = self.select_batch_replacement(each_size[(-1.0, (1, 1, 0))], self.clean_yz_index[(-1.0, (1, 1, 0))], self.batch_num, self.replacement)
            sort_index_y_0_z_111 = self.select_batch_replacement(each_size[(-1.0, (1, 1, 1))], self.clean_yz_index[(-1.0, (1, 1, 1))], self.batch_num, self.replacement)
                            
#             sort_index_y_0_z_1 = self.select_batch_replacement(each_size[(-1, 1)], self.clean_yz_index[(-1,1)], self.batch_num, self.replacement)
#             sort_index_y_1_z_0 = self.select_batch_replacement(each_size[(1, 0)], self.clean_yz_index[(1,0)], self.batch_num, self.replacement)
#             sort_index_y_0_z_0 = self.select_batch_replacement(each_size[(-1, 0)], self.clean_yz_index[(-1,0)], self.batch_num, self.replacement)

            for i in range(self.batch_num):
                key_in_fairbatch = sort_index_y_0_z_000[i].copy()
                key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_1_z_000[i].copy()))
                key_in_fairbatch = sort_index_y_0_z_001[i].copy()
                key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_1_z_001[i].copy()))
                key_in_fairbatch = sort_index_y_0_z_010[i].copy()
                key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_1_z_010[i].copy()))
                key_in_fairbatch = sort_index_y_0_z_011[i].copy()
                key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_1_z_011[i].copy()))
                key_in_fairbatch = sort_index_y_0_z_100[i].copy()
                key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_1_z_100[i].copy()))
                key_in_fairbatch = sort_index_y_0_z_101[i].copy()
                key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_1_z_101[i].copy()))
                key_in_fairbatch = sort_index_y_0_z_110[i].copy()
                key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_1_z_110[i].copy()))
                key_in_fairbatch = sort_index_y_0_z_111[i].copy()
                key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_1_z_111[i].copy()))
                            
#                 key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_0_z_1[i].copy()))
#                 key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_1_z_1[i].copy()))

                random.shuffle(key_in_fairbatch)

                yield key_in_fairbatch

        else:
            entire_index = torch.FloatTensor([i for i in range(len(self.y_data))])

            sort_index = self.select_batch_replacement(self.batch_size, entire_index, self.batch_num, self.replacement)

            for i in range(self.batch_num):
                yield sort_index[i]
        
                               

    def __len__(self):
        """Returns the length of data."""
        
        return len(self.y_data)