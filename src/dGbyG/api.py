import os
import cobra
import base64
from io import BytesIO
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdChemReactions import ChemicalReaction
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, List
from tqdm import tqdm
from functools import lru_cache
from torch_geometric.data import Data

from .utils import to_mol, transformed_ddGf, get_pKa, is_chemaxon_java_available
from .utils.mol_utils import normalize_mol, atom_bag, parse_cid
from .utils.reaction_utils import parse_equation, build_equation, atom_diff, is_balanced, read_rxn_file
from .utils._custom_error import NoPkaError, InputValueError
from .model.datasets import mol_to_graph_data
from .model.inference import Inference_Model
from .utils._to_mol_methods import name_to_smiles, to_mol_methods
from .utils._get_pKa_methods import batch_predict_and_save_pka
from .constants import default_T, default_pMg, default_pH, default_I, default_e_potential, default_condition

pKa_source = None

infer_model_path = (os.path.join(__file__.split('src')[0], 'models', 'mpnn_A139_B23_E300_L2_v2'), )
model_cache = {}



class Compound(object):
    
    recognizable_cids = list(to_mol_methods().keys())

    def __init__(self, mol: Union[Mol, str, None], cid_type:Union[str, None]=None) -> None:
        '''
        '''
        # 
        self.input = mol
        
        if isinstance(mol, Mol) or (mol is None):
            self.raw_mol = mol
        elif isinstance(mol, str) and isinstance(cid_type, str):
            if cid_type.lower() in ['fuzzy name', 'fuzzy_name']:
                Candidates = [comp for comp in [Compound(x, 'smiles') for x in name_to_smiles(mol)] if comp.mol is not None]
                raw_mol = sorted(Candidates, key=lambda comp: comp.standard_dGf_prime)[0].mol if Candidates else None
                self.raw_mol = raw_mol
            else:
                self.raw_mol = to_mol(mol, cid_type)
        elif isinstance(mol, str) and cid_type is None:
            raise InputValueError(f"Please specify the type of {mol} with 'cid_type='.")
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

        self.pKa_source = None

    @property
    def Smiles(self) -> str:
        return Chem.MolToSmiles(Chem.RemoveHs(self.mol), canonical=True) if self.mol else None
    
    @property
    def InChI(self) -> str:
        return Chem.MolToInchi(self.mol) if self.mol else None
    
    @property
    def InChIKey(self) -> str:
        return Chem.MolToInchiKey(self.mol) if self.mol else None

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
    def pKa(self, temperature=default_T, source:Union[str, List[str], None]=None) -> Union[dict, None]:
        if self.Smiles in ['[H+]', '[1H+]']:
            return {'acidicValuesByAtom': [{'atomIndex': 0, 'value': np.nan}], 
                    'basicValuesByAtom': [{'atomIndex': 0, 'value': np.nan}]}
        else:
            if source is not None:
                pass
            elif self.pKa_source is not None:
                source = self.pKa_source
            elif pKa_source is not None:
                source = pKa_source
            else:
                source = 'auto'
            return get_pKa(self.Smiles, temperature, source) if self.mol else None
    
    @property
    def can_be_transformed(self) -> bool:
        return True if self.pKa(default_T) or (self.Smiles in ['[H+]', '[1H+]']) else False

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
            return np.zeros(infer_model.num_head)
        elif self.mol:
            return infer_model(self.graph_data).squeeze().numpy()
        elif self.mol is None:
            return np.full(infer_model.num_head, np.nan)
        else:
            raise ValueError('Unknown value of self.mol')
    
    @property
    @lru_cache(maxsize=None)
    def standard_dGf_prime(self) -> Tuple[np.float32, np.float32]:
        return np.mean(self.standard_dGf_prime_list).item(), np.std(self.standard_dGf_prime_list).item()
    
    @property
    def transformed_standard_dGf_prime(self) -> Tuple[np.float32, np.float32]:
        if self.condition == default_condition:
            return self.standard_dGf_prime
        elif self.can_be_transformed:
            transformed_standard_dg = (self.standard_dGf_prime[0] + self.transformed_ddGf)
            return transformed_standard_dg, self.standard_dGf_prime[1]
        else:
            return self.standard_dGf_prime
        
    # ---------------------------------------------------------
    # 1. 字符串显示方法 __str__
    # ---------------------------------------------------------
    def __str__(self) -> str:
        return self.Smiles if self.Smiles else "None"

    # ---------------------------------------------------------
    # 2. 文本显示方法 __repr__
    # ---------------------------------------------------------
    def __repr__(self) -> str:
        line = "─" * 60
        info = f"\n{line}\n"
        info += f"{'化合物对象':^58}\n"
        info += f"{line}\n"
        info += f"SMILES:    {self.Smiles}\n"
        info += f"InChI:     {self.InChI}\n"
        info += f"InChIKey:  {self.InChIKey}\n"
        
        # 热力学数据 (使用 _f 表示下标)
        dg_mean, dg_std = self.standard_dGf_prime
        if dg_mean is not None and not np.isnan(dg_mean):
            info += f"Δ_fG'°:    {dg_mean:.1f} ± {dg_std:.1f} kJ/mol\n"
        
        # 状态信息
        trans_status = "Yes" if self.can_be_transformed else "No"
        info += f"Has pKa:    {trans_status}\n"
        
        info += f"{line}"
        return info

    # ---------------------------------------------------------
    # 3. HTML 显示方法 _repr_html_
    # ---------------------------------------------------------
    def _repr_html_(self) -> str:
        # 1. 处理图片：转为 Base64 编码
        img_html = "<div style='color:#999; font-size:12px; text-align:center;'>无结构图</div>"
        if self.mol:
            try:
                img = self.image
                if img:
                    buffered = BytesIO()
                    img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    img_html = f"<img src='data:image/png;base64,{img_str}' style='max-width:200px; border-radius:4px; border:1px solid #eee;'/>"
            except Exception:
                img_html = "<div style='color:red;'>图片生成错误</div>"

        # 2. 提取关键数据
        input_obj = self.input if isinstance(self.input, str) else '化合物对象'
        smiles = self.Smiles if self.Smiles else "N/A"
        inchi = self.InChI if self.InChI else "N/A"
        inchikey = self.InChIKey if self.InChIKey else "N/A"
        
        dg_mean, dg_std = self.standard_dGf_prime
        dg_text = f"{dg_mean:.1f} ± {dg_std:.1f} kJ/mol" if not np.isnan(dg_mean) else "N/A"
        
        pka_status = "✅ Yes" if self.can_be_transformed else "❌ No"
        pka_color = "#28a745" if self.can_be_transformed else "#dc3545"

        # 3. HTML 结构：左右布局
        html = f"""
        <div style="font-family: 'Segoe UI', 'Microsoft YaHei', Arial, sans-serif;
                    background-color: #ffffff; color: #333;
                    border: 1px solid #e1e4e8; border-radius: 8px; 
                    padding: 15px; margin: 8px 0;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.05); 
                    display: inline-block; min-width: 450px;">
            <h4 style="margin: 0 0 10px 0; color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 8px;">
                🧪 {input_obj}
            </h4>
            <div style="display: flex; flex-direction: row; gap: 20px; align-items: center;">
                <div style="flex-shrink: 0; background: #f8f9fa; padding: 5px; border-radius: 4px;">
                    {img_html}
                </div>
                <div style="flex-grow: 1; font-size: 13px; line-height: 1.6;">
                    <div style="margin-bottom: 4px;">
                        <span style="font-weight: bold; color: #555;">SMILES:</span> 
                        <code style="background:#f5f5f5; padding:2px 4px; border-radius:3px;">{smiles}</code>
                    </div>
                    <div style="margin-bottom: 4px;">
                        <span style="font-weight: bold; color: #555;">InChI:</span> 
                        <code style="background:#f5f5f5; padding:2px 4px; border-radius:3px; font-size: 11px;">{inchi}</code>
                    </div>
                    <div style="margin-bottom: 4px;">
                        <span style="font-weight: bold; color: #555;">InChIKey:</span> {inchikey}
                    </div>
                    <div style="margin-bottom: 4px;">
                        <span style="font-weight: bold; color: #555;">Δ<sub>f</sub>G'°:</span> 
                        <span style="color: #007bff; font-weight: 600;">{dg_text}</span>
                    </div>
                    <div>
                        <span style="font-weight: bold; color: #555;">Has pKa:</span> 
                        <span style="color: {pka_color}; font-weight: 600;">{pka_status}</span>
                    </div>
                </div>
            </div>
        </div>
        """
        return html




class Reaction(object):
    def __init__(self, reaction:Union[str, ChemicalReaction, Dict[Compound|Mol|str, float|int]], 
                 cids_type:Union[str, None]=None) -> None:
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
        if isinstance(cids_type, str) and ('path' in cids_type):
            reaction = read_rxn_file(reaction)
            self.raw_reaction = dict([(Compound(mol), -1) for mol in reaction.GetReactants()] + [(Compound(mol), 1) for mol in reaction.GetProducts()])
        elif isinstance(reaction, ChemicalReaction):
            self.raw_reaction = dict([(Compound(mol), -1) for mol in reaction.GetReactants()] + [(Compound(mol), 1) for mol in reaction.GetProducts()])
        elif isinstance(reaction, str):
            self.raw_reaction = parse_equation(reaction)
        elif isinstance(reaction, dict):
            self.raw_reaction = reaction
        else:
            raise InputValueError(f'Cannot accept type{type(reaction)} as the input of reaction.')
        
        self.reaction = {}
        for comp, coeff in self.raw_reaction.items():
            if isinstance(comp, Compound):
                pass
            elif isinstance(comp, Mol):
                comp = Compound(comp)
            elif isinstance(comp, str) and isinstance(cids_type, str):
                comp = Compound(comp, cids_type)
            elif isinstance(comp, str) and cids_type is None:
                _cid, _cid_type = parse_cid(comp)
                comp = Compound(_cid, _cid_type)
                
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
        return np.mean(self.standard_dGr_prime_list).item(), np.std(self.standard_dGr_prime_list).item()
        
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
    
    def _build_equation_string(self, eq_sign: str = '=') -> str:
        """
        内部方法：构建反应方程式字符串
        
        Parameters
        ----------
        eq_sign : str, optional
            反应符号，默认为 '='
            
        Returns
        -------
        str
            格式化的反应方程式字符串
        """
        return build_equation(self.rxnSmiles, eq_sign=eq_sign)

    # ═════════════════════════════════════════════════════════════════
    # 2. 文本显示方法 __repr__
    # ═════════════════════════════════════════════════════════════════
    def __repr__(self) -> str:
        """
        文本格式显示，用于列表、终端环境或后备显示
        """
        line = "─" * 50
        info = f"\n{line}\n"
        info += f"{'Chemical Reaction':^48}\n"
        info += f"{line}\n"
        
        # Equation
        info += f"Equation:       {self._build_equation_string('→')}\n"
        
        # Balance
        balance = "✅ Yes" if self.is_balanced else "❌ No"
        info += f"Is Balanced:         {balance}\n"
        
        # Thermodynamics
        dg_mean, dg_std = self.standard_dGr_prime
        if dg_mean is not None and not np.isnan(dg_mean):
            info += f"Δ<sub>r</sub>G'°:          {dg_mean:.1f} ± {dg_std:.1f} kJ/mol\n"
        else:
            info += f"Δ<sub>r</sub>G'°:          N/A\n"
        
        info += f"{line}"
        return info

    # ═════════════════════════════════════════════════════════════════
    # 3. HTML 显示方法 _repr_html_ (精简版：垂直排列，英文标签)
    # ═════════════════════════════════════════════════════════════════
    def _repr_html_(self) -> str:
        """
        HTML 格式显示，用于 Jupyter Notebook
        精简紧凑，仅显示方程式、平衡状态和吉布斯能
        """
        equation = self._build_equation_string('=')
        
        balance_status = "✅ Yes" if self.is_balanced else "❌ No"
        balance_color = "#28a745" if self.is_balanced else "#dc3545"
        
        dg_text = "N/A"
        dg_mean, dg_std = self.standard_dGr_prime
        if dg_mean is not None and not np.isnan(dg_mean):
            dg_text = f"{dg_mean:.1f} ± {dg_std:.1f} kJ/mol"

        html = f"""
        <div style="
            font-family: 'Segoe UI', Arial, sans-serif;
            background-color: #ffffff; color: #333;
            border: 1px solid #e1e4e8; border-left: 4px solid #4a90e2;
            border-radius: 6px; padding: 12px;
            margin: 6px 0; box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            display: inline-block; min-width: 300px; font-size: 13px;
        ">
            <div style="margin-bottom: 8px;">
                <span style="font-weight: bold; color: #555;">Equation:</span><br>
                <code style="
                    background: #f8f9fa; padding: 4px 8px; border-radius: 3px;
                    font-size: 14px; color: #d63384; display: inline-block; margin-top: 3px;
                ">{equation}</code>
            </div>
            
            <div style="margin-bottom: 8px;">
                <span style="font-weight: bold; color: #555;">Is Balanced:</span> 
                <span style="color: {balance_color}; font-weight: 600;">{balance_status}</span>
            </div>
            
            <div>
                <span style="font-weight: bold; color: #555;">Δ<sub>r</sub>G'°:</span> 
                <span style="color: #007bff; font-weight: 600;">{dg_text}</span>
            </div>
        </div>
        """
        return html

    # ═════════════════════════════════════════════════════════════════
    # 4. 简洁字符串方法 __str__
    # ═════════════════════════════════════════════════════════════════
    def __str__(self) -> str:
        """
        print() 时显示简洁版本，使用默认的 '=' 符号
        """
        return self._build_equation_string()



def predict_transformed_dG_prime_for_GEM(
    gem: cobra.Model, 
    compartment_conditions: Dict[str, Dict[str, float|int]] | None = None, 
    use_met_id_types = 'all', 
    ignore_met_id_types = []
):
    """
    Predict transformed standard Gibbs free energies for metabolites and reactions in a GEM.

    For each metabolite, this function extracts compound identifiers from annotations,
    creates Compound objects, retrieves or predicts pKa values using ChemAxon, and then
    computes transformed standard Gibbs free energy of formation. For reactions, it
    combines the transformed dGf_prime values of participating metabolites to calculate
    the transformed standard Gibbs free energy of reaction.

    The function also persists pKa predictions into a local SQLite database via
    `batch_predict_and_save_pka`.

    Parameters
    ----------
    gem : cobra.Model
        A COBRApy genome-scale metabolic model. Metabolites must have annotations
        (e.g., 'inchi', 'pubchem.metabolite', 'kegg.compound', etc.) that can be used
        to look up compound structures.
    compartment_conditions : dict of {str: dict of {str: float|int}}, optional
        Mapping from compartment identifier (abbreviation or full name) to a dictionary
        of physicochemical conditions (e.g., `{'pH': 7.5, 'pMg': 3.0, 'I': 0.2}`).
        If a compartment is not found in this dictionary, a fallback is attempted:
        first the literal key `'c'`, then the full name of the default cytosolic
        compartment (`gem.compartments['c']`), and finally the module‑level
        `default_condition`. If `compartment_conditions` is None, all compartments
        default to `default_condition`.
    use_met_id_types : str or list of str, default='all'
        Which annotation types to use when constructing Compound objects.
        If `'all'`, use all types present in `Compound.recognizable_cids` that are not
        excluded by `ignore_met_id_types`. If a string, only that type is used.
        If a list, the types in the list are used.
    ignore_met_id_types : list of str, default=[]
        Annotation types to ignore when `use_met_id_types='all'`. Matching is performed
        on the beginning of the type string (case-insensitive). E.g., `'kegg'` would
        ignore both `'kegg.compound'` and `'kegg.drug'`.

    Returns
    -------
    Met_df : pandas.DataFrame
        Transformed standard Gibbs free energies of formation per metabolite.
        Index: metabolite IDs from `gem.metabolites`.
        Columns: annotation types (e.g., 'hmdb', 'bigg.metabolite') that were used.
        Values are the computed `comp.transformed_standard_dGf_prime` for each compound.
    Rxn_df : pandas.DataFrame
        Transformed standard Gibbs free energies of reaction per reaction.
        Index: reaction IDs from `gem.reactions`.
        Columns: `'dGr_prime'` and `'SD of dGr_prime'` (standard deviation).

    Raises
    ------
    ValueError
        If `compartment_conditions` is provided but is not a dictionary.
    NameError
        If the global variable `default_condition` is not defined in the module.
    AttributeError
        If required attributes (e.g., `Compound.recognizable_cids`, `Compound.Smiles`,
        `Compound.transformed_standard_dGf_prime`) are missing.
    """
    # ------------------------------------------------------------------
    # 1. Build compartment‑specific condition dictionaries
    # ------------------------------------------------------------------
    conditions = {}
    if isinstance(compartment_conditions, dict):
        # Match each model compartment using abbreviation, full name, or default
        for abbr, full_name in gem.compartments.items():
            if abbr in compartment_conditions:
                conditions[abbr] = compartment_conditions[abbr]
            elif full_name in compartment_conditions:
                conditions[abbr] = compartment_conditions[full_name]
            elif 'c' in compartment_conditions:
                conditions[abbr] = compartment_conditions['c']
            elif gem.compartments['c'] in compartment_conditions:
                conditions[abbr] = compartment_conditions[gem.compartments['c']]
            else:
                conditions[abbr] = default_condition   # global variable
    elif compartment_conditions is not None:
        raise ValueError("compartment_conditions should be a dict with keys as compartment abbreviations or full names.")

    # ------------------------------------------------------------------
    # 2. Normalize ignore_met_id_types (always a list)
    # ------------------------------------------------------------------
    if isinstance(ignore_met_id_types, str):
        ignore_met_id_types = [ignore_met_id_types]

    # ------------------------------------------------------------------
    # 3. Determine which annotation types to use (use_met_id_types)
    # ------------------------------------------------------------------
    if use_met_id_types == 'all':
        use_types = []
        for x in Compound.recognizable_cids:   # expects global Compound class
            keep_x = not any([x.lower().startswith(y) for y in ignore_met_id_types])
            if keep_x:
                use_types.append(x)
    elif isinstance(use_met_id_types, str):
        use_types = [use_met_id_types]
    else:
        use_types = use_met_id_types

    # ------------------------------------------------------------------
    # 4. For each metabolite, create Compound objects for selected annotation types
    # ------------------------------------------------------------------
    for met in tqdm(gem.metabolites, desc="Processing metabolites"):
        met.compound = {}
        for cid_type, cid in met.annotation.items():
            if cid_type in use_types:
                # If cid is a list, take the first element (assume it's the primary ID)
                comp = Compound(cid, cid_type) if isinstance(cid, str) else Compound(cid[0], cid_type)
                comp.condition = conditions.get(met.compartment, default_condition)
                met.compound[cid_type] = comp

    # ------------------------------------------------------------------
    # 5. Collect all unique SMILES from metabolites and pre‑compute/persist pKa
    # ------------------------------------------------------------------
    Smiles = []
    for met in gem.metabolites:
        Smiles.extend([comp.Smiles for comp in met.compound.values() if comp.Smiles])
    batch_predict_and_save_pka(Smiles)   # saves pKa to DB for future lookups

    # ------------------------------------------------------------------
    # 6. Build DataFrame of transformed dGf_prime per metabolite
    # ------------------------------------------------------------------
    Data = {}
    for met in tqdm(gem.metabolites, desc="Predicting transformed standard Gibbs free energy for metabolites"):
        met.pKa_source = 'chemaxon_pKa_db'   # record source for debugging
        Data[met.id] = {}
        for cid_type, comp in met.compound.items():
            Data[met.id][cid_type] = comp.transformed_standard_dGf_prime
    Met_df = pd.DataFrame(Data).T

    # ------------------------------------------------------------------
    # 7. Build DataFrame of transformed dGr_prime per reaction
    # ------------------------------------------------------------------
    Data = {}
    for rxn in tqdm(gem.reactions, desc="Predicting transformed standard Gibbs free energy for reactions"):
        rxn_dict = {}
        for met, coeff in rxn.metabolites.items():
            # Find the first Compound object associated with this metabolite that has a valid mol
            comp = Compound(None, None)        # fallback empty compound
            for comp in met.compound.values():
                if comp.mol is not None:
                    break
            rxn_dict[comp] = coeff
        reaction = Reaction(rxn_dict, cids_type='compound')
        Data[rxn.id] = reaction.transformed_standard_dGr_prime   # expected to be (dGr, sd)
    Rxn_df = pd.DataFrame(Data, index=['dGr_prime', 'SD of dGr_prime']).T

    return Met_df, Rxn_df

