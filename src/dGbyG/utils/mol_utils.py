import re
from typing import Dict, Callable
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from ._custom_error import InputValueError
from ._to_mol_methods import to_mol
from ._get_pKa_methods import get_pKa_from_chemaxon, get_pKa_from_json, check_pKa_json



def normalize_mol(mol:Mol) -> Mol:
    """
    Normalize and uncharge the mol.

    Parameters:
    ----------
    mol: rdkit.Chem.rdchem.Mol
        The input molecule.

    Returns:
    -------
    mol: rdkit.Chem.rdchem.Mol
        The normalized and uncharged molecule with explicit hydrogens.
    """
    ## mol sometimes changed when transform to smiles and then read smiles
    mol = rdMolStandardize.Normalize(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    smiles = Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True)
    mol = to_mol(smiles, 'smiles')
    mol = rdMolStandardize.Normalize(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)

    return Chem.AddHs(mol)


def atom_bag(mol:Mol) -> Dict[str, int|float]:
    """
    Get the atom bag of the mol.

    Parameters:
    ----------
    mol: rdkit.Chem.rdchem.Mol
        The input molecule.
    
    Returns:
    -------
    atom_bag: Dict[str, int|float]
        The atom bag of the molecule.
    """
    atom_bag = {}
    charge = 0
    for atom in mol.GetAtoms():
        atom_bag[atom.GetSymbol()] = 1 + atom_bag.get(atom.GetSymbol(), 0)
        charge += atom.GetFormalCharge()
    atom_bag['charge'] = charge
        
    return atom_bag

