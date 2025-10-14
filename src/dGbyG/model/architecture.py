import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_add_pool



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
        node_emb = self.atom_lin(data.x) # node_emb.shape = [N, atom_num, hidden_dim]
        edge_emb = self.bond_lin(data.edge_attr) # edge_emb.shape = [N, bond_num, hidden_dim]
        
        # Step 2: message passing
        for MP_layer in self.MP_layers:
            node_emb = MP_layer(node_emb, data.edge_index, edge_emb)

        # Step 3: transform node embedding of each node to a single value(energy value)
        node_energy = self.energy_lin(node_emb) # node_energy.shape = [N, atom_num, 1]

        # Step 4: add all the nodes' energy of a molecule to get the molecule's energy
        if mode=='molecule mode':
            dg = self.pool(node_energy, data.batch) # dg.shape = [N, 1, 1]
            return dg
        elif mode=='atom mode':
            return node_energy

