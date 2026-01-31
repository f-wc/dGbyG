import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_add_pool

from .datasets import atom_func_dict


class MP_layer(MessagePassing):
    """
    Message Passing layer.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x_emb:torch.Tensor, edge_index:torch.Tensor, edge_emb:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the message passing layer.

        Parameters:
        ----------
        x_emb : torch.Tensor
            Node embeddings of shape [num_nodes, emb_dim].
        edge_index : torch.Tensor or SparseTensor
            Edge indices.
        edge_emb : torch.Tensor
            Edge embeddings of shape [num_edges, emb_dim].

        Returns:
        -------
        torch.Tensor
            Updated node embeddings of shape [num_nodes, emb_dim].
        """
        # Residual connection
        x_emb = x_emb + self.propagate(edge_index, x = x_emb, edge_attr = edge_emb)
        return x_emb

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # Hadamard product is better than plus
        return x_j * edge_attr



class MPNN_model(nn.Module):
    """
    Message Passing Neural Network for predicting Gibbs energy of molecules.
    """
    def __init__(self, atom_dim:int, bond_dim:int, emb_dim:int=300, num_layer:int=2):
        super().__init__()
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.emb_dim = emb_dim
        self.num_layer = num_layer

        # Number of atom features used for constructing base node embedding
        self.atom_num_feature = atom_func_dict['atomic number'][0]
        self.base_atom_lin = nn.Linear(self.atom_num_feature, self.atom_dim - self.atom_num_feature)
        
        # Node embedding layer: from atom features to node embedding
        self.atom_lin = nn.Linear(self.atom_dim, self.emb_dim)

        # Edge embedding: from bond features to edge embedding
        self.bond_lin = nn.Linear(self.bond_dim, self.emb_dim)

        # Message Passing layers: aggregate messages from neighbors
        self.MP_layers = nn.ModuleList([MP_layer() for _ in range(self.num_layer)])

        # Energy linear layer: from node embedding to energy value
        self.energy_lin = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim//2),
            nn.ReLU(),
            nn.Linear(self.emb_dim//2, 1, bias=False)
        )

        # Pooling function
        self.pool = global_add_pool

        # Weight initialization
        self.weight_init()


    def weight_init(self):
        """
        Initialize the weights of linear layers using Kaiming uniform initialization.
        """
        for layer in self.modules():
            # Since all trainable layers are nn.Linear, we only need to initialize nn.Linear layers with Kaiming uniform.
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight.data, nonlinearity='relu')


    def forward(self, data: Data, mode='molecule mode') -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters:
        ----------
        data : Data
            Input graph data containing node features, edge indices, and edge features.
            data.x.shape = [N, atom_num, atom_dim], data.edge_emb.shape = [N, bond_num, edge_dim]
        mode : str, optional
            Mode of operation, either 'molecule mode' for predicting Gibbs energy of molecule or 'atom mode' for predicting Gibbs energy of each atom. 
            Default is 'molecule mode'.

        Returns:
        -------
        torch.Tensor
            If mode is 'molecule mode', returns tensor of shape [N, 1, 1] representing Gibbs energy of each molecule.
            If mode is 'atom mode', returns tensor of shape [N, atom_num, 1] representing Gibbs energy of each atom.
        """
        # Step 1: embedding atoms and bonds
        base_node_x = torch.cat([data.x.T[:self.atom_num_feature].T, self.base_atom_lin(data.x.T[:self.atom_num_feature].T)], dim=-1) # base_node_x.shape = [N, atom_num, atom_dim]
        base_node_emb = self.atom_lin(base_node_x) # base_node_emb.shape = [N, atom_num, hidden_dim]
        node_emb = self.atom_lin(data.x) # node_emb.shape = [N, atom_num, hidden_dim]
        edge_emb = self.bond_lin(data.edge_attr) # edge_emb.shape = [N, bond_num, hidden_dim]
        
        data.x.T
        # Step 2: message passing
        for MP_layer in self.MP_layers:
            node_emb = MP_layer(node_emb, data.edge_index, edge_emb)

        # Step 3: transform node embedding of each node to a single value(energy value)
        node_energy = self.energy_lin(node_emb) - self.energy_lin(base_node_emb) # node_energy.shape = [N, atom_num, 1]

        # Step 4: add all the nodes' energy of a molecule to get the molecule's energy
        if mode=='molecule mode':
            dg = self.pool(node_energy, data.batch) # dg.shape = [N, 1, 1]
            return dg.squeeze(-1) # dg.shape = [N, 1]
        elif mode=='atom mode':
            return node_energy.squeeze(-1) # node_energy.shape = [N, atom_num]



class Multihead_MPNN_model(nn.Module):
    """
    Multihead Message Passing Neural Network for predicting Gibbs energy of molecules.
    """
    def __init__(self, parameters_list):
        super().__init__()
        # 
        self.sub_modules = nn.ModuleList()
        for param in parameters_list:
            sub_module = MPNN_model(atom_dim=param['atom_dim'], 
                                    bond_dim=param['bond_dim'], 
                                    emb_dim=param['emb_dim'], 
                                    num_layer=param['num_layer'])
            self.sub_modules.append(sub_module)
            

    def forward(self, data: Data, mode='molecule mode') -> torch.Tensor:
        # 1. 列表推导收集所有子模块输出（比for循环快，无原地赋值）
        # 2. 自动匹配数据/模型的设备，避免隐式拷贝
        outputs = [sub_module(data, mode) for sub_module in self.sub_modules]
        # 3. 批量堆叠为张量（形状：[num_heads, 1, 1]），替代逐个赋值
        output = torch.stack(outputs, dim=0)
        return output