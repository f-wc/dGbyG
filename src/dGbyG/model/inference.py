import os, re
import torch
import torch.nn as nn

from . import MPNN_model
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

        # 
        if isinstance(folder_path, str):
            self.folder_path = [folder_path]
        elif isinstance(folder_path, (list, tuple)):
            self.folder_path = folder_path

        # 
        pattern = r'A(\d+)_B(\d+)_E(\d+)_L(\d+)'
        self.MPNN_models = nn.ModuleDict([])
        self.num_models = len(self.folder_path)
        for folder in self.folder_path:
            match = re.search(pattern, folder)
            if match:
                self.atom_dim = int(match.group(1)) #
                self.bond_dim = int(match.group(2)) #
                self.emb_dim = int(match.group(3)) #
                self.num_layer = int(match.group(4)) #
                self.num_head = len(os.listdir(folder)) #
            else:
                raise InputValueError('The folder path does not match the pattern. Please check the folder path and try again.')

            self.MPNN_head = nn.ModuleList([])
            for file in os.listdir(folder):
                net = MPNN_model(atom_dim=self.atom_dim, bond_dim=self.bond_dim, emb_dim=self.emb_dim, num_layer=self.num_layer)

                # Loading multiple pre-trained MPNN models from the specified directory.
                path = os.path.join(folder, file)
                net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
                self.MPNN_head.append(net)
            self.MPNN_models[folder] = self.MPNN_head
        self.eval()
        self.to(self.device)
    
    def forward(self, data:Data, mode:str='molecule mode'):
        # 
        data.to(self.device)
        if mode == 'molecule mode':
            outputs = torch.zeros(size=(self.num_models, self.num_head, 1), requires_grad=False) # shape=[number of model, number of head, 1]
        elif mode == 'atom mode':
            outputs = torch.zeros(size=(self.num_models, self.num_head, data.x.shape[0]), requires_grad=False) # shape=[number of model, number of head, atom number]
        
        with torch.no_grad():
            for i, head_list in enumerate(self.MPNN_models.values()):
                outputs[i] = torch.stack([head(data, mode) for head in head_list], dim=0)
        idx = torch.argmin(outputs.std(dim=1), dim=0)
        outputs = outputs[idx, :, torch.arange(len(idx), device=outputs.device)].T # outputs.shape = [number of head, 1] or [number of head, atom number]

        return outputs
    