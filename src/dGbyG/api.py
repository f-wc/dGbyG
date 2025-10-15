import os
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdChemReactions import ChemicalReaction
import numpy as np
from typing import Dict, Tuple, Union, List
from functools import lru_cache
from torch_geometric.data import Data

from . import default_T, default_pMg, default_pH, default_I, default_e_potential, default_condition
from .utils import to_mol, transformed_ddGf, get_pKa
from .utils.mol_utils import normalize_mol, atom_bag
from .utils.reaction_utils import parse_equation, build_equation, atom_diff, is_balanced, read_rxn_file
from .utils._custom_error import NoPkaError, InputValueError
from .model.datasets import mol_to_graph_data
from .model.inference import Inference_Model

infer_model_path = os.path.join(__file__.split('src')[0], 'models', 'mpnn_A139_B23_E300_L2')
model_cache = {}



class Compound(object):
    def __init__(self, mol: Union[Mol, str, None], mol_type:str='mol') -> None:
        '''
        '''
        # 
        if isinstance(mol, Mol) or (mol is None):
            self.raw_mol = mol
        elif isinstance(mol, str):
            self.raw_mol = to_mol(mol, mol_type)
        else:
            raise InputValueError('The input of Compound() must be rdkit.Chem.rdchem.Mol, string, or None.')

        # Normalize mol
        if isinstance(self.raw_mol, Mol):
            self.mol = normalize_mol(self.raw_mol)
        elif self.raw_mol is None:
            self.mol = None
        else:
            raise InputValueError(f'Unknown error in Compound.__init__().')
        
        self.compartment = None
        self._condition = default_condition.copy()
        self._l_concentration = None
        self._u_concentration = None
        self._lz = None
        self._uz = None

    @property
    def Smiles(self) -> str:
        return Chem.MolToSmiles(Chem.RemoveHs(self.mol), canonical=True) if self.mol else None
    
    @property
    def InChI(self) -> str:
        return Chem.MolToInchi(self.mol) if self.mol else None
    
    @property
    def image(self):
        if None:
            for atom in self.mol.GetAtoms():
                atom.SetProp("atomNote", str(atom.GetIdx()))
                #atom.SetProp('molAtomMapNumber',str(atom.GetIdx()))
        return Chem.Draw.MolToImage(Chem.RemoveHs(self.mol)) if self.mol else None

    @property
    def atom_bag(self) -> Dict[str, int | float]:
        return atom_bag(self.mol) if self.mol else None
    
    @property
    def condition(self) -> Dict[str, float]:
        return self._condition
    @condition.setter
    def condition(self, condition: Dict[str, float | int]):
        if not isinstance(condition, dict):
            raise InputValueError('The input of condition must be a dict.')
        elif x:=set(condition.keys()) - set(self.condition.keys()):
            raise InputValueError(f'Condition includes {', '.join(self.condition.keys())}, but got {', '.join(x)}.')
        elif condition.get('T', 298.15) != 298.15:
            raise InputValueError('The temperature cannot be changed and must be 298.15 K.')
        else:
            for k,v in condition.items():
                if isinstance(v, (int, float)):
                    self._condition[k] = float(v)
                else:
                    raise InputValueError(f'The value of {k} must be a float or int, but got {type(condition[k])}.')

    @property
    def uz(self) -> float:
        return self._uz
    @uz.setter
    def uz(self, uz:float):
        self._uz = uz
        self._u_concentration = 10 ** uz
    
    @property
    def lz(self) -> float:
        return self._lz
    @lz.setter
    def lz(self, lz:float):
        self._lz = lz
        self._l_concentration = 10 ** lz

    @property
    def u_concentration(self) -> float:
        return self._u_concentration
    @u_concentration.setter
    def u_concentration(self, concentration:float):
        self._u_concentration = concentration
        self._uz = np.log10(concentration)

    @property
    def l_concentration(self) -> float:
        return self._l_concentration
    @l_concentration.setter
    def l_concentration(self, concentration:float):
        self._l_concentration = concentration
        self._lz = np.log10(concentration)

    @lru_cache(16)
    def pKa(self, temperature=default_T, source:Union[str, List[str]]='auto') -> Union[dict, None]:
        if self.Smiles == '[H+]':
            return {'acidicValuesByAtom': [{'atomIndex': 0, 'value': np.nan}], 
                    'basicValuesByAtom': [{'atomIndex': 0, 'value': np.nan}]}
        else:
            return get_pKa(self.Smiles, temperature, source) if self.mol else None
    
    @property
    def can_be_transformed(self) -> bool:
        return True if self.pKa(default_T) or (self.Smiles == '[H+]') else False

    @property
    def transformed_ddGf(self):
        if self.can_be_transformed == True:
            T = self.condition.get('T', default_T)            
            ddGf = transformed_ddGf(
                pKa=self.pKa(T), 
                pH=self.condition.get('pH', default_pH), 
                T=T, 
                pMg=self.condition.get('pMg', default_pMg), 
                I=self.condition.get('I', default_I), 
                e_potential=self.condition.get('e_potential', default_e_potential), 
                charge=self.atom_bag.get('charge', 0), 
                num_H=self.atom_bag.get('H', 0), 
                num_Mg=self.atom_bag.get('Mg', 0)
                )
            return ddGf
        elif self.can_be_transformed == False:
            raise NoPkaError('This compound has no available Pka value, so it cannot be transformed.')
        else:
            raise ValueError('Unknown value of self.can_be_transformed')
        
    @property
    @lru_cache(maxsize=None)
    def graph_data(self) -> Data:
        return mol_to_graph_data(self.mol) if self.mol else None

    @property
    @lru_cache(maxsize=None)
    def standard_dGf_prime_list(self) -> np.ndarray:
        # 
        if infer_model_path not in model_cache:
            model_cache[infer_model_path] = Inference_Model(infer_model_path)
        infer_model = model_cache[infer_model_path]

        # 
        if self.Smiles == '[H+]':
            return np.zeros(infer_model.num_models)
        elif self.mol:
            return infer_model(self.graph_data).squeeze().numpy()
        elif self.mol is None:
            return np.full(infer_model.num_models, np.nan)
        else:
            raise ValueError('Unknown value of self.mol')
    
    @property
    @lru_cache(maxsize=None)
    def standard_dGf_prime(self) -> Tuple[np.float32, np.float32]:
        return np.mean(self.standard_dGf_prime_list), np.std(self.standard_dGf_prime_list)
    
    @property
    def transformed_standard_dGf_prime(self) -> Tuple[np.float32, np.float32]:
        if self.condition == default_condition:
            return self.standard_dGf_prime
        elif self.can_be_transformed:
            transformed_standard_dg = (self.standard_dGf_prime[0] + self.transformed_ddGf)
            return transformed_standard_dg, self.standard_dGf_prime[1]
        else:
            return self.standard_dGf_prime
        




class Reaction(object):
    def __init__(self, reaction:Union[str, ChemicalReaction, Dict[Compound|Mol|str, float|int]], mol_type:str='compound') -> None:
        """
        Parameters
        ----------
        reaction: dict or str of equation, or rxn file path
            reaction can be:
            (1) a dict of {rdkit.Chem.rdchem.Mol: coefficient}.
            (2) a dict of {Compound: coefficient}. 
            (3) a string of equation. e.g.
            (4) a file path of rxn file.

        mol_type: str
            'path', 'compound', 'mol', 'smiles', 'kegg', and so on.
        """
        if 'path' in mol_type:
            reaction = read_rxn_file(reaction)
            self.raw_reaction = dict([(Compound(mol), -1) for mol in reaction.GetReactants()] + [(Compound(mol), 1) for mol in reaction.GetProducts()])
        elif isinstance(reaction, str):
            self.raw_reaction = parse_equation(reaction)
        elif isinstance(reaction, dict):
            self.raw_reaction = reaction
        elif isinstance(reaction, ChemicalReaction):
            self.raw_reaction = dict([(Compound(mol), -1) for mol in reaction.GetReactants()] + [(Compound(mol), 1) for mol in reaction.GetProducts()])
        else:
            raise InputValueError(f'Cannot accept type{type(reaction)} as the input of reaction.')
        
        self.reaction = {}
        for comp, coeff in self.raw_reaction.items():
            if isinstance(comp, Compound):
                pass
            elif isinstance(comp, (Mol, str)):
                comp = Compound(comp, mol_type)
            else:
                raise InputValueError('Cannot accept type{0}'.format(type(comp)))
            
            if not isinstance(coeff, (float, int)):
                raise InputValueError(f"The value's type of input dict should be float or int, but got {type(coeff)}")
            
            self.reaction.update({comp:coeff})

        # 
        self.ignore_H2O = False
        self.ignore_H_ion = True
        self.ignore_charge = False
        self.ignore_H = True

        # 
        self.reaction = self.balance(self.reaction)


    @property
    def condition(self) -> Dict[str, float]:
        conditions = {}
        for comp in self.reaction.keys():
            conditions[comp] = comp.condition
        return conditions
    @condition.setter
    def condition(self, condition: Dict[str, float | int]):
        for comp in self.reaction.keys():
            comp.condition = condition
        
    @property
    def rxnSmiles(self) -> dict:
        rxn_dict_smiles = map(lambda item: (item[0].Smiles, item[1]), self.reaction.items())
        return dict(rxn_dict_smiles)
    
    @property
    def rxnInChI(self) -> dict:
        rxn_dict_inchi = map(lambda item: (item[0].InChI, item[1]), self.reaction.items())
        return dict(rxn_dict_inchi)
    
    def equationSmiles(self, remove_H_ion:bool = False) -> str:
        temp = self.rxnSmiles.copy()
        if remove_H_ion == True:
            temp.pop('[H+]', None)
        return build_equation(temp)
    
    def equiationInChI(self, remove_H_ion:bool = False) -> str:
        temp = self.rxnInChI.copy()
        if remove_H_ion == True:
            temp.pop('InChI=1S/p+1', None)
        return build_equation(temp)
    
    @property
    def substrates(self) -> Dict[Compound, float]:
        return dict([(c,v) for c,v in self.reaction.items() if v<0])
    
    @property
    def products(self) -> Dict[Compound, float]:
        return dict([(c,v) for c,v in self.reaction.items() if v>0])
    
    def pKa(self, temperature=default_T):
        pKa = []
        for compound in self.reaction:
            pKa.append(compound.pKa(temperature))
        return pKa
    
    @property
    def atom_diff(self) -> Dict[Compound, float]:
        mol_dict = dict([(comp.mol, coeff) for comp, coeff in self.reaction.items()])
        return atom_diff(mol_dict)

    @property
    def is_balanced(self) -> bool:
        """
        Return whether the reaction is balanced.
        """
        mol_dict = dict([(comp.mol, coeff) for comp, coeff in self.reaction.items()])
        output = is_balanced(mol_dict, ignore_H2O=self.ignore_H2O, 
                             ignore_H_ion=self.ignore_H_ion, 
                             ignore_charge=self.ignore_charge, 
                             ignore_H=self.ignore_H)
        return output
    
    def balance(self, reaction: Dict[Compound, float]) -> Dict[Compound, float]:
        # 
        if (self.is_balanced is None) or (self.is_balanced is True):
            return reaction
        elif self.is_balanced is False:
            original_reaction = reaction
            reaction = reaction.copy()
            diff_atom = self.atom_diff

            compounds_smiles = [comp.Smiles for comp in reaction.keys()]
            
            num_H2O = diff_atom.get('O')
            if (not self.ignore_H2O) and num_H2O:
                if '[H]O[H]' not in compounds_smiles:
                    reaction[Compound(to_mol('[H]O[H]', cid_type='smiles'))] = -num_H2O
                else:
                    for comp in reaction.keys():
                        if comp.Smiles == '[H]O[H]':
                            reaction[comp] = reaction[comp] - num_H2O
                            break
            
            if diff_atom.get('charge', 0) * diff_atom.get('H', 0) <= 0:
                num_H_ion = 0
            elif diff_atom['charge'] < 0:
                num_H_ion = -max(diff_atom['charge'], diff_atom['H'])
            elif diff_atom['charge'] > 0:
                num_H_ion = -min(diff_atom['charge'], diff_atom['H'])
            else:
                pass
            if (not self.ignore_H_ion) and num_H_ion:
                if '[H+]' not in compounds_smiles:
                    reaction[Compound(to_mol('[H+]', cid_type='smiles'))] = -num_H_ion
                else:
                    for comp in reaction.keys():
                        if comp.Smiles == '[H+]':
                            reaction[comp] = reaction[comp] - num_H_ion
                            break

            if self.is_balanced:
                return reaction
            else:
                return original_reaction
        else:
            raise ValueError(f'Unknown balanced_bool value: {self.is_balanced}')
    
    @property
    def can_be_transformed(self) -> bool:
        for x in self.reaction.keys():
            if not x.can_be_transformed:
                return False
        return True
        
    @property
    @lru_cache(maxsize=None)
    def standard_dGr_prime_list(self) -> Union[np.ndarray, None]:
        """
        Return the list of standard dG for the reaction.
        """
        standard_dGr_list = np.sum([comp.standard_dGf_prime_list * coeff for comp, coeff in self.reaction.items()], axis=0)
        if self.is_balanced:
            return standard_dGr_list
        else:
            return standard_dGr_list * np.nan

    @property
    @lru_cache(maxsize=None)
    def standard_dGr_prime(self) -> Tuple[np.float32, np.float32]:
        """
        Returns
        -------
            The tuple of the mean and SD of the standard dG for the reaction.
        """
        return np.mean(self.standard_dGr_prime_list), np.std(self.standard_dGr_prime_list)
        
    @property
    def transformed_standard_dGr_prime(self) -> Tuple[np.float32, np.float32]:
        """
        Returns
        -------
            The tuple of the mean and SD of transformed standard dG for the reaction.
        """
        if (np.array(list(self.condition.values())) == default_condition).all():
            return self.standard_dGr_prime
        if self.can_be_transformed:
            transformed_ddGr = np.sum([comp.transformed_ddGf * coeff for comp, coeff in self.reaction.items()], axis=0)
            transformed_standard_dGr_prime = self.standard_dGr_prime[0] + transformed_ddGr
            return transformed_standard_dGr_prime, self.standard_dGr_prime[1]
        else:
            return self.standard_dGr_prime