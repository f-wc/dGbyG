import time
import copy
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm, trange
from random import shuffle, seed
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from .datasets import TrainDataset


class Trainer(object):
    def __init__(self) -> None:
        self.device = self.__get_device()
        self.criterion = nn.MSELoss
        self.optimizer = torch.optim.Adam


    def __get_device(self):
        # which device running on
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        return device

    @property
    def network(self):
        return self.__network__
    
    @network.setter
    def network(self, network:nn.Module):
        self.__network__ = network.to(self.device)

    
    def train(self, dataset:TrainDataset, epochs:int, lr:float, weight_decay=0) -> tuple:
        # train
        print('train on:', self.device)

        network = self.network.to(self.device)
        print(datetime.now().strftime(r'%Y-%m-%d %H:%M:%S'), 'start preparing data')
        S = dataset.S.to(self.device)
        dGs = dataset.dGs.to(self.device)
        weight = dataset.weight.to(self.device) if dataset.weight!=None else None

        for data in DataLoader(dataset, batch_size=len(dataset)):
            data.to(self.device)
            
        Result_df = pd.DataFrame([])
        Result_df['r'] = dataset.dGs.cpu().numpy().tolist()

        print(datetime.now().strftime(r'%Y-%m-%d %H:%M:%S'), 'start training')
        loss_history, p, n = self.__train(network=network, data=data, S=S, dgs=dGs, weight=weight,
            epochs=epochs, lr=lr, weight_decay=weight_decay)
        print(datetime.now().strftime(r'%Y-%m-%d %H:%M:%S'), 'training have done')

        loss_history = loss_history.cpu().numpy()
        Result_df['p'] = p.detach().cpu().numpy().tolist()

        return loss_history, Result_df, n

    def __train(self, network:nn.Module, data, S, dgs, weight, epochs, lr, weight_decay=0):
        #
        device = self.device
        criterion = self.criterion().to(device)
        optimizer = self.optimizer(network.parameters(), lr = lr, weight_decay=weight_decay)

        loss_history = torch.empty(size=(0,)).to(device)

        network.train()
        network_params = network.state_dict()
        
        min_loss = torch.inf
        print('Training |', end='')
        for epoch_counter in range(epochs):
            optimizer.zero_grad()

            cids_energy = network(data).view(1,-1)
            p_dgs = torch.mm(cids_energy, S).view(-1)
            
            if weight == None:
                loss = criterion(p_dgs, dgs) 
            else:
                loss = criterion(p_dgs*weight, dgs*weight)
            loss.backward()

            optimizer.step()
            
            loss_history = torch.concat((loss_history, loss.detach().view(-1)), axis=0)
            if loss < min_loss:
                min_loss = loss.detach()
                n = epoch_counter
                network_params = copy.deepcopy(network.state_dict())
                p_dGs = p_dgs
            
            if epoch_counter % (epochs//100+1) == 0:
                print('=', end='')
        print('| Done!')

        assert loss_history.shape == (epochs,)
        network.load_state_dict(network_params)
        
        return loss_history, p_dGs, n
    


    def cross_validation(self, dataset:TrainDataset, mode, epochs:int, lr:float, weight_decay=0, fold_num:int=None, 
                         n_start:int=None, n_end:int=None, train_idx:list=None, val_idx:list=None, random_seed:int=None):
        # This funtion is used for cross_validation
        total_idx = list(range(dataset.S.shape[1]))
        Result_df = pd.DataFrame(dataset.dGs.cpu().numpy(), index=total_idx, columns=['r'])
        Loss = torch.empty(size=(0,epochs)).to(self.device)
        
        print('Cross validation. Start at:{0}'.format(datetime.now().strftime(r'%Y-%m-%d %H:%M:%S'))) 
        print('train on:', self.device)

        S = dataset.S.to(self.device) # S.shape = (n_compounds, n_reactions)
        dGs = dataset.dGs.to(self.device) # dGs.shape = (n_reactions,)
        weight = dataset.weight.to(self.device) if dataset.weight!=None else None # weight.shape = (n_reactions,)
        for data in DataLoader(dataset, batch_size=len(dataset)):
            data.to(self.device)
        
        # Step 1. 
        if mode == 'leave-one-out':
            fold_num = len(total_idx)
            print('Mode: leave-one-out cross validation.')
            if n_start==None:
                n_start = 0
            if n_end==None:
                n_end = fold_num
        
        elif mode=='K-fold' and type(fold_num)==int and fold_num>=2:
            print('Mode: K-fold validation. K =', fold_num)
            if n_start==None:
                n_start = 0
            if n_end==None:
                n_end = fold_num

        elif mode=='reverse K-fold' and type(fold_num)==int and fold_num>=2:
            print('Mode: {0} validation. K = {1}'.format(mode, fold_num))
            if n_start==None:
                n_start = 0
            if n_end==None:
                n_end = fold_num

        elif mode=='manual' and train_idx is not None and val_idx is not None:
            print('Mode: manual. train size = {0}, val size = {1}'.format(len(train_idx), len(val_idx)))
            n_start, n_end = 0, 1

        elif type(mode)==int and mode>=2:
            fold_num = mode
            mode = 'K-fold'
            print('Mode: K-fold validation. K =', fold_num)
            if n_start==None:
                n_start = 0
            if n_end==None:
                n_end = fold_num

        else:
            print('no that mode!')
            return False
        
        seed(random_seed)
        shuffle(total_idx)

        # Step 2. 
        for n in range(n_start, n_end):#trange(fold_num, desc="cross validation", position=0):#
            # Step 2.1. Preliminaries
            #tqdm.write(f'fold: {n}.', end=' ')
            #tqdm.write('Start at:{0}'.format(datetime.now().strftime(r'%Y-%m-%d %H:%M:%S'))) 
            network_copy = copy.deepcopy(self.network)
            network_copy.to(self.device)
            
            if mode in ['leave-one-out', 'K-fold']:
                val_idx = total_idx[n::fold_num]
                train_idx = list(set(total_idx)-set(val_idx))
                assert len(total_idx) == len(val_idx) + len(train_idx)
            elif mode=='reverse K-fold':
                train_idx, val_idx = total_idx[n::fold_num], total_idx[(n+1)%fold_num::fold_num]
            elif mode=='manual':
                pass
            
            train_S = S[:, train_idx]
            val_S = S[:, val_idx]
            train_dGs = dGs[train_idx]
            train_weight = weight[train_idx] if weight!=None else None

            # Step 2.2. 
            loss_history, val_history = self.__cross_validation(network=network_copy, data=data, train_S=train_S, train_dGs=train_dGs, val_S=val_S, 
                train_weight=train_weight, epochs=epochs, lr=lr, weight_decay=weight_decay, desc=f'fold {n}')

            Loss = torch.concat((Loss, loss_history.view(1,epochs)), axis=0)

            Result_df.loc[val_idx, list(range(epochs))] = val_history.cpu().numpy()
            #print('End at:', datetime.now().strftime(r'%Y-%m-%d %H:%M:%S'))
        
        Loss = Loss.cpu().numpy()

        print('Cross validation. End at:', datetime.now().strftime(r'%Y-%m-%d %H:%M:%S'))
        return Loss, Result_df.astype(np.float32)



    def __cross_validation(self, network, 
                           data: torch.Tensor, 
                           train_S: torch.Tensor, 
                           train_dGs: torch.Tensor, 
                           val_S: torch.Tensor, 
                           epochs: int, 
                           lr: float, 
                           train_weight: torch.Tensor, 
                           weight_decay: float=0, 
                           desc: str='') -> Tuple[torch.Tensor, torch.Tensor]:
        # 
        # 
        criterion = self.criterion().to(self.device)
        optimizer = self.optimizer(network.parameters(), lr = lr, weight_decay=weight_decay)

        loss_history = torch.empty(size=(0,)).to(self.device)
        val_history = torch.empty(size=(val_S.shape[1], 0)).to(self.device)
        
        
        for epoch_counter in trange(epochs, desc=desc, leave=True):
            # train
            network.train()
            optimizer.zero_grad()
            
            cids_energy = network(data) # cids_energy.shape = [n_compounds (batch_size), 1]
            p_dGs = torch.mm(train_S.T, cids_energy).squeeze() # p_dGs.shape = [n_reactions, ]
            
            if train_weight == None:
                loss = criterion(p_dGs, train_dGs)
            else:
                loss = criterion(p_dGs*train_weight, train_dGs*train_weight)
            
            # val
            network.eval()
            with torch.no_grad():
                val_dGs = torch.mm(val_S.T, cids_energy) # val_dGs.shape = [n_reactions, 1]

            loss.backward()
            optimizer.step()
                
            loss_history = torch.concat((loss_history, loss.detach().view(-1)), axis=0)
            val_history = torch.concat((val_history, val_dGs), axis=1)

        assert loss_history.shape == (epochs,)
        assert val_history.shape == (val_S.shape[1], epochs)
        
        return loss_history, val_history
