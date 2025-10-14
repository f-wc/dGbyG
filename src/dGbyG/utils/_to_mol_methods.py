# This file contains methods to convert different identifiers to RDKit molecules.
import os
import requests
import zipfile
import pandas as pd
from typing import Dict, Callable, Union
import pubchempy as pcp
from Bio import Entrez
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from ._custom_error import InputValueError
from .config import config


# set database paths
package_path = __file__.split('src')[0]
recon3d_mol_dir_path = os.path.join(package_path, 'data', 'Recon3D', 'mol')


cache = {}


def smiles_to_mol(smiles:str, sanitize=True) -> Union[Mol, None]:
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    return mol


def inchi_to_mol(inchi:str, sanitize=True) -> Union[Mol, None]:
    mol = Chem.MolFromInchi(inchi, removeHs=False, sanitize=sanitize)
    return mol


def file_to_mol(path:str, sanitize=True) -> Union[Mol, None]:
    mol = Chem.MolFromMolFile(path, removeHs=False, sanitize=sanitize)
    return mol


def kegg_compound_to_mol(entry:str, sanitize=True) -> Union[Mol, None]:
    path = os.path.join(config.kegg_database_path, 'mol', entry+'.mol')
    if os.path.exists(path):
        mol = file_to_mol(path, sanitize=sanitize)
    else:
        kegg_additions_csv_path = os.path.join(config.kegg_database_path, 'kegg_additions.csv')
        kegg_additions_df = pd.read_csv(kegg_additions_csv_path, index_col=0)
        if entry in kegg_additions_df.index:
            inchi = kegg_additions_df.loc[entry, 'inchi']
            mol = inchi_to_mol(inchi, sanitize=sanitize) if pd.notna(inchi) else None
    return mol


def metanetx_id_to_mol(cid:str, sanitize=True) -> Union[Mol, None]:
    # load data to cache
    if 'metanetx' not in cache:
        path = os.path.join(config.metanetx_database_path, 'structures.csv.zip')
        cache['metanetx'] = pd.read_csv(path, compression='zip', comment='#', index_col=0, header=0)
    
    # 
    df = cache['metanetx']
    if cid in df.index:
        smiles = df.loc[cid, 'SMILES']
        mol = smiles_to_mol(smiles, sanitize=sanitize)
        return mol
    else:
        return None

def hmdb_id_to_mol(cid:str, sanitize=True) -> Union[Mol, None]:
    # load data to cache
    if 'hmdb' not in cache:
        path = os.path.join(config.hmdb_database_path, 'structures.csv.zip')
        cache['hmdb'] = pd.read_csv(path, compression='zip', comment='#', index_col=0, header=0)
    
    # 
    if len(cid.replace('HMDB', '')) < 7:
        cid = 'HMDB' + '0'*(7-len(cid.replace('HMDB', ''))) + cid.replace('HMDB', '')
        
    df = cache['hmdb']
    if cid in df.index:
        smiles = df.loc[cid, 'SMILES']
        mol = smiles_to_mol(smiles, sanitize=sanitize)
        return mol
    else:
        return None


def chebi_id_to_mol(cid:str, sanitize=True) -> Union[Mol, None]:
    # load data to cache
    if 'chebi' not in cache:
        path = os.path.join(config.chebi_database_path, 'structures.csv.zip')
        cache['chebi'] = pd.read_csv(path, compression='zip', comment='#', header=0, dtype=str).set_index('ChEBI_ID')
    
    # 
    cid = str(cid).lower().replace('chebi:', '')
    
    df = cache['chebi']
    if cid in df.index:
        smiles = df.loc[cid, 'SMILES']
        mol = smiles_to_mol(smiles, sanitize=sanitize)
        return mol
    else:
        return None


def lipidmaps_id_to_mol(cid:str, sanitize=True) -> Union[Mol, None]:
    # load data to cache
    if 'lipidmaps' not in cache:
        path = os.path.join(config.lipidmaps_database_path, 'lipidmaps_ids_cc0.tsv')
        cache['lipidmaps'] = pd.read_csv(path, sep='\t', index_col=0).rename(columns={'smiles':'SMILES'})
    
    # 
    df = cache['lipidmaps']
    if cid in df.index:
        smiles = df.loc[cid, 'SMILES']
        mol = smiles_to_mol(smiles, sanitize=sanitize)
        return mol
    else:
        return None


def pubchem_to_mol(cid:str, sanitize=True) -> Union[Mol, None]:
    cid = str(cid).lower().replace('pubchem:', '')
    try:
        comp = pcp.Compound.from_cid(cid)
        smiles = comp.connectivity_smiles
    except:
        smiles = None
    
    if smiles:
        mol = smiles_to_mol(smiles, sanitize=sanitize)
        return mol
    else:
        return None
    

def recon3d_id_to_mol(cid:str, sanitize=True) -> Union[Mol, None]:
    # load data to cache
    with zipfile.ZipFile(config.recon3d_database_path+'/mol.zip') as zf:
        namelist = [os.path.basename(f).removesuffix('.mol') for f in zf.namelist() if f.endswith('.mol')]
        filelist = [os.path.join('mol', f'{name}.mol') for name in namelist]
        cache['recon3d'] = pd.DataFrame({'cid': namelist, 'file': filelist}).set_index('cid')
    
    # 
    df = cache['recon3d']
    if cid in df.index:
        with zipfile.ZipFile(config.recon3d_database_path+'/mol.zip') as zf:
            with zf.open(os.path.join('mol', f'{cid}.mol')) as f:
                mol = Chem.MolFromMolBlock(f.read(), removeHs=False, sanitize=sanitize)
        return mol
    else:
        return None


def inchi_key_to_mol(inchi_key:str, sanitize=True) -> Union[Mol, None]:
    # set your email here
    Entrez.email = "your_email@example.com"

    # 1. search PubChem database for the InChI Key
    handle = Entrez.esearch(db="pccompound", term=inchi_key, sort="relevance", retmax=1)
    record = Entrez.read(handle)
    handle.close()

    # 2. get PubChem ID（CID）
    cid = record["IdList"][0]

    # 3. get smiles of the compound from PubChem
    handle = Entrez.esummary(db="pccompound", id=cid)
    summary = Entrez.read(handle)
    handle.close()
    smiles = summary[0]['IsomericSmiles']
    
    # 3. finally
    if smiles: 
        mol = smiles_to_mol(smiles, sanitize=sanitize)
    else:
        mol = None
    return mol


def bigg_id_to_mol(cid:str, sanitize=True) -> Union[Mol, None]:
    # load data to cache
    if 'bigg' not in cache:
        path = os.path.join(config.bigg_database_path, 'bigg_models_metabolites.txt')
        cache['bigg'] = pd.read_csv(path, sep='\t', usecols=['universal_bigg_id', 'database_links']).drop_duplicates('universal_bigg_id').set_index('universal_bigg_id')
    
    # 
    df = cache['bigg']
    if cid in df.index:
        mol = None
        for s in df.loc[cid, 'database_links'].split('; '):
            database, url = s.split(': ')
            cid = [seg for seg in url.split('/') if seg][-1]
            if (mol is None) and (database == 'MetaNetX (MNX) Chemical'):
                mol = metanetx_id_to_mol(cid, sanitize=sanitize)
            elif (mol is None) and (database == 'CHEBI'):
                mol = chebi_id_to_mol(cid, sanitize=sanitize)
            elif (mol is None) and (database == 'KEGG Compound'):
                mol = kegg_compound_to_mol(cid, sanitize=sanitize)
            elif (mol is None) and (database == 'Human Metabolome Database'):
                mol = hmdb_id_to_mol(cid, sanitize=sanitize)
            elif (mol is None) and (database == 'LipidMaps'):
                mol = lipidmaps_id_to_mol(cid, sanitize=sanitize)
        return mol
    else:
        return None


def name_to_mol(name:str, sanitize=True) -> Union[Mol, None]:
    # from pubchempy
    comps = pcp.get_compounds(name, 'name')
    if comps:
        comp = comps[0]
        smiles = comp.smiles
    else:
        smiles = None

    # from NCI Cactus API
    if not smiles:
        url = f"https://cactus.nci.nih.gov/chemical/structure/{name}/smiles"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                smiles = response.text.strip()  # 返回SMILES
            else:
                pass
        except Exception as e:
            pass

    # finally
    if smiles: 
        mol = smiles_to_mol(smiles, sanitize=sanitize)
    else:
        mol = None
    return mol



def to_mol_methods() -> Dict[str, Callable]:
    methods = {'inchi': inchi_to_mol,
               'smiles': smiles_to_mol,
               'file': file_to_mol,
               'kegg': kegg_compound_to_mol,
               'kegg.compound': kegg_compound_to_mol,
               'metanetx': metanetx_id_to_mol,
               'metanetx.chemical': metanetx_id_to_mol,
               'hmdb': hmdb_id_to_mol,
               'chebi': chebi_id_to_mol,
               'lipidmaps': lipidmaps_id_to_mol,
               'lipid maps': lipidmaps_id_to_mol,
               'pubchem': pubchem_to_mol,
               'pubchem.compound': pubchem_to_mol,
               'recon3d': recon3d_id_to_mol,
               'inchi-key': inchi_key_to_mol,
               'inchi_key': inchi_key_to_mol,
               'inchi key': inchi_key_to_mol,
               'bigg': bigg_id_to_mol,
               'bigg.metabolite': bigg_id_to_mol,
               'name': name_to_mol,
               }
    return methods

def to_mol(cid:str, cid_type:str, Hs:bool=True, sanitize:bool=True) -> Union[Mol, None]:
    """
    Convert cid to mol. 

    Parameters
    ----------
    cid : string
        cid to be converted
    cid_type : string
        cid type, can be one of the to_mol_methods().keys()
    Hs : bool, optional
        If True, add Hs to the mol (default).  
        If False, remove Hs from the mol.
    
    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol

    Raises
    ------
    InputValueError : Exception
        If cid or cid_type is not string, or cid_type is not in to_mol_methods().keys()

    """
    if not isinstance(cid, str):
        raise InputValueError('cid must be String type, but got {0}'.format(type(cid)))
    elif not isinstance(cid_type, str):
        raise InputValueError('cid_type must be String type, but got {0}'.format(type(cid_type)))

    # the main body
    methods = to_mol_methods()

    if cid_type.lower() not in methods.keys():
        raise InputValueError(f'cid_type must be one of {list(methods.keys())}, {cid_type} id cannot be recognized')
    
    cid_type = cid_type.lower()
    if cid_type=='auto':
        _to_mols = methods
    else:
        _to_mols = {cid_type: methods[cid_type]}

    output = {}
    for _cid_type, _to_mol in _to_mols.items():
        try:
            mol = _to_mol(cid)
        except:
            mol = None
        if mol:
            if Hs==True:
                mol = Chem.AddHs(mol)
            elif Hs==False:
                mol = Chem.RemoveHs(mol)
            output[_cid_type] = mol
    if len(output)>1:
        raise ValueError(f'Which {cid} is {tuple(output.keys())}?')
    return tuple(output.values())[0] if output else None


