import re
from typing import Dict, Callable
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from ._custom_error import InputValueError
from ._to_mol_methods import to_mol, to_mol_methods
from ._get_pKa_methods import get_pKa_from_chemaxon, get_pKa_from_json, check_pKa_json



def parse_cid(cid:str) -> str:
    """
    Parse the cid to cid and cid_type.

    Parameters:
    ----------
    cid: str
        The input cid. e.g. 'kegg:C00063', 'inchi:InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3' or 'smiles:CCO'.
    
    Returns:
    -------
    cid: str
        The parsed cid.
    cid_type: str
        The type of the cid.
    """
    for cid_type, to_mol_method in to_mol_methods().items():
        if cid.lower().startswith(f'{cid_type}:'):
            _, cid = [x.strip() for x in cid.split(':', 1)]
            return cid, cid_type
    raise InputValueError(f'Invalid cid: {cid}. Please use the format of "cid_type:cid". e.g. "kegg:C00063" or "smiles:CCO".\nThe supported cid_types are {", ".join(to_mol_methods().keys())}')


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

