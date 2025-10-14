import numpy as np
import pandas as pd
from typing import List
from functools import reduce
from rdkit.Chem.rdchem import Mol, Atom, Bond, HybridizationType, ChiralType, BondType

import torch
from torch_geometric.data import Dataset, Data

from ..utils import to_mol
from ..utils.mol_utils import normalize_mol
from ..utils.reaction_utils import parse_equation


# Atom's features
Num_atomic_number = 119 # including the extra mask tokens
Num_atom_hybridization = len(HybridizationType.values)
Num_atom_aromaticity = 2 # Atom's aromaticity (not aromatic or aromactic)
Num_atom_chirality = len(ChiralType.values)
Num_atom_charge = 9
Num_atom_degree = 5

# Bond's features
Num_bond_type = len(BondType.values) # #including aromatic and self-loop edge(value = 22)


atom_func_dict = {'atomic number':(Num_atomic_number, Atom.GetAtomicNum), 
                  'hybridization': (Num_atom_hybridization, Atom.GetHybridization), 
                  'aromaticity': (Num_atom_aromaticity, Atom.GetIsAromatic), 
                  'charge': (Num_atom_charge, Atom.GetFormalCharge), 
                  'chirality': (Num_atom_chirality, Atom.GetChiralTag), 
                  'degree': (Num_atom_degree, Atom.GetDegree), 
                  }

bond_func_dict = {'bond type':(Num_bond_type+1, Bond.GetBondType), # remain "0" for self loop 
                  'begin atom num':(Num_atomic_number, lambda x:Bond.GetBeginAtom(x).GetAtomicNum()), 
                  'end atom num':(Num_atomic_number, lambda x:Bond.GetEndAtom(x).GetAtomicNum()), 
                  }



def one_hot(num:int, idx:int) -> List[int]:
    vector = [0]*num
    vector[idx] = 1
    return vector


def equations_to_S(equations) -> pd.DataFrame:
    comps = set()
    for cs in [parse_equation(x).keys() for x in equations]:
        for c in cs:
            comps.add(c)
    comps = list(comps)
    comps.sort()

    S = pd.DataFrame(index=comps, columns=range(len(equations)), dtype=float, data=0)
    for i in range(len(equations)):
        for comp, coeff in parse_equation(equations[i]).items():
            S.loc[comp,i] = coeff

    return S


def mol_to_graph_data(mol: Mol, 
                      atom_features=['atomic number', 'hybridization', 'aromaticity', 'charge'], 
                      bond_features=['bond type']) -> Data:
    # return (x, bond_index, bond_attr)
    # atoms features: including such below(ordered).
    atom_featurizers = [atom_func_dict[x] for x in atom_features]
    atoms_features = []

    for atom in mol.GetAtoms():
        feature = reduce(lambda x,y:x+y, [one_hot(num, fun(atom).real) for num, fun in atom_featurizers])
        atoms_features.append(feature)
    atoms_features = torch.tensor(atoms_features, dtype=torch.float32)

    
    # bonds index
    bonds_index = torch.empty(size=(2,0), dtype=torch.int64)
    
    # bonds attributes: including such below(ordered)
    # bond_type + bond_begin_atom_features + bond_end_atom_features
    bond_featurizers = [bond_func_dict[x] for x in bond_features]
    bond_dim = sum([num for num, _ in bond_featurizers])
    bonds_attrs = torch.empty(size=(0, bond_dim), dtype=torch.float32)
        
    for bond in mol.GetBonds():
        begin, end = Bond.GetBeginAtomIdx(bond), Bond.GetEndAtomIdx(bond)

        bonds_index = torch.cat((bonds_index, torch.tensor([begin,end]).unsqueeze(1)), dim=1)
        attr = reduce(lambda x,y:x+y, [one_hot(num, fun(bond).real) for num, fun in bond_featurizers])
        bonds_attrs = torch.cat((bonds_attrs, torch.tensor(attr).unsqueeze(0)), dim=0)

        bonds_index = torch.cat((bonds_index, torch.tensor([end,begin]).unsqueeze(1)), dim=1)
        attr = reduce(lambda x,y:x+y, [one_hot(num, fun(bond).real) for num, fun in bond_featurizers])
        bonds_attrs = torch.cat((bonds_attrs, torch.tensor(attr).unsqueeze(0)), dim=0)
    
    return Data(x=atoms_features, edge_index=bonds_index, edge_attr=bonds_attrs)



class TrainDataset(Dataset):
    #
    def __init__(self, 
                 equations:list,
                 dGs:list,
                 weights:list=None,
                 ):
        super(TrainDataset, self).__init__()
        self.equations = equations
        self.df = equations_to_S(self.equations)
        self.S = torch.tensor(self.df.to_numpy()).float()
        self.cids = self.df.index.to_list()
        self.dGs = torch.tensor(dGs, dtype=torch.float)
        self.weight = torch.tensor(weights).float() if weights is not None else None

        self.mols = np.asarray([to_mol(cid, cid_type='smiles') for cid in self.cids])
        self.normalized_mols = np.asarray([normalize_mol(mol) for mol in self.mols])
        

    def len(self) -> int:
        return self.__len__()
    
    def get(self, idx) -> Data:
        return self.__getitem__(idx)

    def __len__(self) -> int:
        return self.S.shape[0]
    
    def __getitem__(self, idx) -> Data:
        mol = self.normalized_mols[idx]
        graph_data = mol_to_graph_data(mol)
        return graph_data
