import os, re
import torch
import torch.nn as nn

from .architecture import MPNN_model
from .datasets import Data
from ..utils._custom_error import InputValueError



class Inference_Model(nn.Module):
    def __init__(self, folder_path, device=None) -> None:
        #         atom_dim, bond_dim, emb_dim, num_layer, num_models
        super().__init__()
        # 
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.folder_path = folder_path

        # 
        pattern = r'A(\d+)_B(\d+)_E(\d+)_L(\d+)'
        match = re.search(pattern, folder_path)
        if match:
            self.atom_dim = int(match.group(1)) #
            self.bond_dim = int(match.group(2)) #
            self.emb_dim = int(match.group(3)) #
            self.num_layer = int(match.group(4)) #
            self.num_models = len(os.listdir(folder_path)) #
        else:
            raise InputValueError('The folder path does not match the pattern. Please check the folder path and try again.')

        self.MPNN_models = nn.ModuleList([])
        for file in os.listdir(self.folder_path):
            net = MPNN_model(atom_dim=self.atom_dim, bond_dim=self.bond_dim, emb_dim=self.emb_dim, num_layer=self.num_layer)

            # Loading multiple pre-trained MPNN models from the specified directory.
            path = os.path.join(self.folder_path, file)
            net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            self.MPNN_models.append(net)
        self.eval()
        self.to(self.device)
    
    def forward(self, data:Data, mode:str='molecule mode'):
        # 
        data.to(self.device)
        if mode == 'molecule mode':
            outputs = torch.zeros(size=(self.num_models, 1), requires_grad=False) # shape=[number of net, 1]
        elif mode == 'atom mode':
            outputs = torch.zeros(size=(self.num_models, data.x.shape[0], 1), requires_grad=False) # shape=[number of net, atom number, 1]
        
        with torch.no_grad():
            for i, net in enumerate(self.MPNN_models):
                outputs[i] = net(data, mode) # net.shape = [1,1] or [atom number, 1]

        return outputs
    