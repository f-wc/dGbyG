import re
from typing import Dict, Callable
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdChemReactions import ChemicalReaction
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from ._custom_error import InputValueError
from .mol_utils import atom_bag


def parse_equation(equation:str, eq_sign=None) -> dict:
    """
    Parse equation to get reactants and products.
    
    Parameters:
    ----------
    equation: str
        The equation string.
    eq_sign: str
        The equation sign, default is None.

    Returns:
    -------
    equation_dict: dict
        The equation dictionary, with reactants and products as keys and their coefficients as values.
    """
    # 
    eq_Signs = [' = ', ' <=> ', ' -> ']
    if eq_sign:
        equation = equation.split(eq_sign)
    else:
        for eq_sign in eq_Signs:
            if eq_sign in equation:
                equation = equation.split(eq_sign)
                break
            
        
        
    if not type(equation)==list:
        return {equation:1}
    
    equation_dict = {}
    equation = [x.split(' + ') for x in equation]

    for side, coefficient in zip(equation, (-1,1)):
        for t in side:
            if (reactant := re.match(r'^(\d+) (.+)$', t)) or (reactant := re.match(r'^(\d+\.\d+) (.+)$', t)):
                value = float(reactant.group(1))
                entry = reactant.group(2)
                equation_dict[entry] = equation_dict.get(entry,0) + value * coefficient
            else:
                equation_dict[t] = equation_dict.get(t,0) + coefficient
                
    if '' in equation_dict:
        equation_dict.pop('')

    return equation_dict


def build_equation(equation_dict:dict, eq_sign:str='=') -> str:
    """
    Build equation from equation_dict.
    """
    # 
    left, right = [], []
    for comp, coeff in equation_dict.items():
        if coeff < 0:
            x = comp if coeff==-1 else str(-coeff)+' '+comp
            left.append(x)
        elif coeff > 0:
            x = comp if coeff==1 else str(coeff)+' '+comp
            right.append(x)
        elif coeff == 0:
            left.append(comp)
            right.append(comp)

    equation = ' + '.join(left)+' '+eq_sign.strip()+' '+' + '.join(right)
    return equation


def atom_diff(reaction:Dict[Mol|None, float|int]) -> bool:
    """
    Get the unbalanced atom of the reaction.
    Parameters:
    ----------
    reaction: Dict[Mol|None, float|int]
        The reaction dictionary, with molecules as keys and their coefficients as values.

    Returns:
    -------
    unbalanced_atom: Dict[str, int|float]|None
        The unbalanced atom dictionary, with atom symbols as keys and their net counts as values. 
        If the reaction is balanced, returns an empty dictionary. 
        If any molecule is None, returns None.

    """
    #  检查反应字典中是否存在None值
    if None in reaction.keys():
        return None
    # 初始化一个空字典用于存储原子差异
    diff_atom = {}
    # 遍历反应中的每个分子及其系数
    for mol, coeff in reaction.items():
        # 遍历分子中的每个原子及其数量
        for atom, num in atom_bag(mol).items():
            # 计算并更新原子的净数量
            diff_atom[atom] = diff_atom.get(atom, 0) + coeff * num

    # 检查并移除标记原子（R和*）如果它们的总和为0
    if (diff_atom.get('R', 0) + diff_atom.get('*', 0)) == 0:
        diff_atom.pop('R', None)
        diff_atom.pop('*', None)
        
    # 初始化不平衡原子字典
    unbalanced_atom = {}
    # 遍历差异原子字典
    for atom, num in diff_atom.items():
        # 如果原子数量不为0，则添加到不平衡原子字典中
        if num!=0:
            unbalanced_atom[atom] = num

    # 返回不平衡原子字典
    return unbalanced_atom



def is_balanced(reaction:Dict[Mol|None, float|int], 
                ignore_H2O=False, ignore_H_ion=False, ignore_charge=False, ignore_H=False) -> bool:
    # 
    if None in reaction.keys():
        return None
    
    diff_atom = atom_diff(reaction)

    if ignore_H2O:
        H2O_num = -diff_atom.get('O', 0)
        diff_atom['O'] = 0 # diff_atom.get('O', 0) + H2O_num
        diff_atom['H'] = diff_atom.get('H', 0) + 2*H2O_num
    if ignore_H_ion:
        if diff_atom.get('charge', 0) * diff_atom.get('H', 0) <= 0:
            H_ion_num = 0
        elif diff_atom['charge'] < 0:
            H_ion_num = -max(diff_atom['charge'], diff_atom['H'])
        elif diff_atom['charge'] > 0:
            H_ion_num = -min(diff_atom['charge'], diff_atom['H'])
        else:
            pass
        diff_atom['charge'] = diff_atom.get('charge', 0) + H_ion_num
        diff_atom['H'] = diff_atom.get('H', 0) + H_ion_num
    if ignore_charge:
        diff_atom['charge'] = 0
    if ignore_H:
        diff_atom['H'] = 0

        
    unbalanced_atom = {}
    for atom, num in diff_atom.items():
        if num!=0:
            unbalanced_atom[atom] = num

    return False if unbalanced_atom else True



def read_rxn_file(file_path:str) -> ChemicalReaction:
    """
    """
    rxn = AllChem.ReactionFromRxnFile(file_path)
    return rxn
