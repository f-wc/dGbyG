"""
This module provides a function to get pKa values for a compound.
pKa data can be obtained from three sources:
(1) From a local cache file (.json file).
(2) From ChemAxon's Calculator plugins (license required).
(3) From ChemAxon's REST API (only available for small molecules).
"""
import os
import json
import jpype
import shutil
import requests
import portalocker
import multiprocessing
import numpy as np
from copy import deepcopy
from typing import List, Union, Iterable
from tqdm import tqdm

from ._custom_error import NoLicenseError, InputValueError


pka_cache = {}


# Path settings
# set chemaxon_pka_json_path
chemaxon_pka_json_path = os.path.join(__file__.split('src')[0], 'data', 'chemaxon_pKa.json')

# set chemaxon_jar_dir
if shutil.which('cxcalc'):
    cxcalc_path = shutil.which('cxcalc')
    cxcalc_path = os.path.realpath(cxcalc_path) if os.path.islink(cxcalc_path) else cxcalc_path
    chemaxon_jar_dir = os.path.join(os.path.dirname(os.path.dirname(cxcalc_path)), 'lib')
else:
    chemaxon_jar_dir = ''

# set chemaxon_license_file_path, refer to https://docs.chemaxon.com/display/lts-iodine/license-installation.md
if os.environ.get("CHEMAXON_LICENSE_URL"):
    chemaxon_license_file_path = os.environ.get("CHEMAXON_LICENSE_URL")
elif os.environ.get("CHEMAXON_HOME"): # <ChemAxon_home> can be set by the CHEMAXON_HOME environment variable
    chemaxon_home = os.environ.get("CHEMAXON_HOME")
    chemaxon_license_file_path = os.path.join(chemaxon_home, "license.cxl") # The default location is <ChemAxon_home>/license.cxl or <ChemAxon_home>/licenses/ folder. 
else:
    user_home = os.environ.get("HOME")
    if os.name == 'posix':
        chemaxon_home = os.path.join(user_home, ".chemaxon") # <User_home>/.chemaxon (on Unix-like systems)
        chemaxon_license_file_path = os.path.join(chemaxon_home, "license.cxl")
    elif os.name == 'nt':
        chemaxon_home = os.path.join(user_home, "chemaxon") # <User_home>/chemaxon (on Windows)
        chemaxon_license_file_path = os.path.join(chemaxon_home, "license.cxl")
    else:
        chemaxon_license_file_path = ''



def get_pka_from_chemaxon_rest(smiles:str, temperature:float) -> dict:
    """
    Get pKa values for a single SMILES from ChemAxon's REST API.

    Parameters
    ----------
    smiles : str
        SMILES string of the compound.
    temperature : float, optional
        Temperature in Kelvin.

    Returns
    -------
    pka : dict
    """

    chemaxon_pka_api = 'https://jchem-microservices.chemaxon.com/jws-calculations/rest-v1/calculator/calculate/pka'
    headers = {'accept': '*/*', 'Content-Type': 'application/json'}
    pka_req_body = json.dumps({
        "inputFormat": "smiles",
        "micro": False,
        "outputFormat": "smiles",
        "outputStructureIncluded": False,
        "pKaLowerLimit": -20,
        "pKaUpperLimit": 20,
        "prefix": "STATIC",
        "structure": smiles,
        "temperature": temperature,
        "types": "acidic, basic",
        })
    
    try:
        pka = requests.post(chemaxon_pka_api, data=pka_req_body, headers=headers).json()
        if not pka.get('error'):
            pka = {'acidicValuesByAtom': pka['acidicValuesByAtom'], 'basicValuesByAtom':pka['basicValuesByAtom']}
        else:
            pka = None
    except:
        pka = None

    return pka

def _batch_get_pKa_from_chemaxon(smiles_list:List[str], temperature:float) -> List[str|dict]:
    """
    Get pKa values for a list of SMILES from ChemAxon's Calculator plugins (license required).

    Parameters
    ----------
    smiles_list : list of str
        A list of SMILES string of the compound.
    temperature : float
        Temperature in Kelvin.

    Returns
    -------
    output : list of dict

    Raises
    ------
    FileNotFoundError :
        If ChemAxon jar files not found.
    NoLicenseError :
        If ChemAxon license not found.
    """

    if not os.path.isdir(chemaxon_jar_dir):
        raise FileNotFoundError("ChemAxon jar files not found")
    else:
        jar_dir = chemaxon_jar_dir
        fileList = [os.path.join(jar_dir,i)  for i in os.listdir(jar_dir)]

    if not os.path.isfile(chemaxon_license_file_path):
        raise NoLicenseError("ChemAxon license not found")
    else:
        jpype.startJVM(f'-Dchemaxon.license.url={chemaxon_license_file_path}')
        
    for p in fileList:
        jpype.addClassPath(p)

    MolImporter = jpype.JClass('chemaxon.formats.MolImporter')
    pKaPlugin = jpype.JClass('chemaxon.marvin.calculations.pKaPlugin')
    pKa = pKaPlugin()
    if not pKa.isLicensed():
        return ["ChemAxon license not found"]
    
    output = []
    pKa.setTemperature(temperature)
    pKa.setpKaPrefixType(2) # 'acidic,basic'
    pKa.setAcidicpKaUpperLimit(20)
    pKa.setBasicpKaLowerLimit(-20)

    if len(smiles_list) > 1:
        smiles_list = tqdm(smiles_list)
        
    for smiles in smiles_list:
        mol=MolImporter.importMol(smiles)
        pKa.setMolecule(mol)
        #print(smiles)
        if not pKa.run():
            output.append("pKa calculation failed")
            continue

        apka = []
        bpka = []
        for i in range(mol.getAtomCount()):
            apka.append({'atomIndex': i, 'value': float(pKa.getpKa(i, pKa.ACIDIC))})
            bpka.append({'atomIndex': i, 'value': float(pKa.getpKa(i, pKa.BASIC))})
        res = {'acidicValuesByAtom':apka, 'basicValuesByAtom':bpka}
        output.append(res)
        
    jpype.shutdownJVM()
    print('shut down JVM!')
    return output

def get_pKa_from_chemaxon(smiles:str, temperature:float) -> Union[dict, None]:
    """
    Get pKa values for a single SMILES from ChemAxon's Calculator plugins (license required).

    Parameters
    ----------
    smiles : string
        A SMILES string of the compound.
    temperature : float, optional
        Temperature in Kelvin.

    Returns
    -------
    output: dict
    
    Raises
    ------
    FileNotFoundError :
        If ChemAxon jar files not found.
    NoLicenseError :
        If ChemAxon license not found.

    """
    
    if not os.path.isdir(chemaxon_jar_dir):
        raise FileNotFoundError("ChemAxon jar files not found")

    if not os.path.isfile(chemaxon_license_file_path):
        raise NoLicenseError("ChemAxon license not found")
    
    if not isinstance(smiles, str):
        raise InputValueError("get_pKa_from_chemaxon(smiles:str, temperature:float=default_T), smiles must be a string")
    queue = multiprocessing.Queue()
    func = lambda queue, smiles, temperature: queue.put(_batch_get_pKa_from_chemaxon([smiles], temperature))
    p = multiprocessing.Process(target=func, args=(queue, smiles, temperature, ))
    p.start()
    p.join()

    res = queue.get()[0]
    if res == "pKa calculation failed":
        print(smiles, res)
        return None
    elif isinstance(res, dict):
        return res
    else:
        raise Exception(f"Unknown error, return value: {res}")

def batch_calculation_pKa_to_json(smiles_list:list, temperature:float) -> None:
    """
    Batch get pKa values from ChemAxon's Calculator plugins (license required) and save to local files (.json).

    Parameters
    ----------
    smiles_list: list of string
        A list of SMILES string of compounds.
    temperature: float, optional
        Temperature in Kelvin.

    Returns
    -------
    output: list of dict
    
    Raises
    ------
    FileNotFoundError :
        If ChemAxon jar files not found.
    NoLicenseError :
        If ChemAxon license not found.
    """ 
    
    if not os.path.isdir(chemaxon_jar_dir):
        raise FileNotFoundError("ChemAxon jar files not found")

    if not os.path.isfile(chemaxon_license_file_path):
        raise NoLicenseError("ChemAxon license not found")
    
    if not isinstance(smiles_list, list):
        raise InputValueError("batch_calculation_pKa_to_json(), smiles must be a list")
    queue = multiprocessing.Manager().Queue()
    func = lambda queue, smiles_list, temperature: queue.put(_batch_get_pKa_from_chemaxon(smiles_list, temperature))
    p = multiprocessing.Process(target=func, args=(queue, smiles_list, temperature, ))
    p.start()
    p.join()
    print('ChemAxon pKa calculation finished')

    pKa_list = queue.get()
    if pKa_list[0] == "ChemAxon license not found":
        raise NoLicenseError("ChemAxon license not found")
    
    print('Reading json file...')
    with open(chemaxon_pka_json_path, "r", encoding="utf-8") as f:
        file = f.read()
        if len(file) > 0:
            data = json.loads(file)
        else:
            data = {}
    
    assert len(smiles_list) == len(pKa_list)
    for smiles, pka in zip(smiles_list, pKa_list):
        if pka == "pKa calculation failed":
            pka = None
        smiles_data = data.get(smiles, {})
        smiles_data[str(temperature)] = pka
        data[smiles] = smiles_data
    print('Writing json file...')
    with open(chemaxon_pka_json_path, "w", encoding="utf-8") as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        json.dump(data, f, sort_keys=True, )
    print('Done')
    return None

def get_pka_from_json(smiles:str, temperature:float) -> Union[dict, None]:
    """
    Get pKa values for a single SMILES from local files (.json).

    Parameters
    ----------
    smiles : string
        A SMILES string of the compound.
    temperature : float, optional
        Temperature in Kelvin.

    Returns
    -------
    dict
    """
    if 'chemaxon_pKa_json' not in pka_cache.keys():
        try:
            with open(chemaxon_pka_json_path, "r", encoding="utf-8") as f:
                pka_cache['chemaxon_pKa_json'] = json.load(f)
        except:
            pka_cache['chemaxon_pKa_json'] = {}
    
    pKa_json = pka_cache['chemaxon_pKa_json']
    if str(temperature) not in pKa_json.get(smiles, {}).keys():
        print(f'{smiles}, {temperature} K pKa not in file')
        batch_calculation_pKa_to_json([smiles], temperature)
        with open(chemaxon_pka_json_path, "r", encoding="utf-8") as f:
            pka_cache['chemaxon_pKa_json'] = json.load(f)
            pKa_json = pka_cache['chemaxon_pKa_json']
        
    return deepcopy(pKa_json[smiles][str(temperature)])

def check_pKa_json(smiles_list:list, temperature:float, predict:bool=True) -> List[str]:
    """
    Check if pKa values for a list of SMILES are in local .json file.  
    If predict is True, calculate pKa for them and save to local .json file.

    Parameters
    ----------
    smiles_list : list of string
        A list of SMILES strings of the compounds.
    temperature : float, optional
        Temperature in Kelvin.
    predict : bool, optional
        If True, calculate pKa for the compounds not in the .json file.

    Returns
    -------
    list : list of SMILES strings of the compounds not in the .json file.
    """
    with open(chemaxon_pka_json_path, "r", encoding="utf-8") as f:
        pka_cache['chemaxon_pKa_json'] = json.load(f)
        
    not_in_list = []
    pKa_json = pka_cache['chemaxon_pKa_json']
    for smiles in smiles_list:
        if str(temperature) not in pKa_json.get(smiles, {}).keys():
            not_in_list.append(smiles)

    if len(not_in_list) == 0:
        pass
    elif predict is True:
        try:
            batch_calculation_pKa_to_json(not_in_list, temperature)
            with open(chemaxon_pka_json_path, "r", encoding="utf-8") as f:
                pka_cache['chemaxon_pKa_json'] = json.load(f)
        except:
            pass
    else:
        pass
    return not_in_list



def get_pKa_methods():
    methods = {}
    # if pka json file exists, add method to get pka from json file
    if os.path.isfile(chemaxon_pka_json_path):
        methods['chemaxon_pKa_json'] = get_pka_from_json

    # if chemaxon jar files and license file exist, add method to get pka from chemaxon
    if os.path.isdir(chemaxon_jar_dir) and os.path.isfile(chemaxon_license_file_path):
        methods['chemaxon'] = get_pKa_from_chemaxon

    # 
    methods['chemaxon_rest'] = get_pka_from_chemaxon_rest
    return methods




def get_pKa(smiles, temperature, source:Union[str, List[str]]='auto') -> dict:
    """
    Get pKa values for a single SMILES.

    Parameters:
    ----------
    smiles : string
        A SMILES string of the compound.
    temperature : float
        Temperature in Kelvin.
    source : string or list of string, optional
        The source of pKa values. If 'auto', use the first available source of all. If a list of strings, use the first available source in the list. If a string, use the specified source. Default is 'auto'.
    
    Returns:
    -------
    dict
        A dictionary of pKa values. The keys are atom indices and the values are dictionaries of pKa values. 
    """
    # Avialable methods
    methods = get_pKa_methods()
    # the main body of this function
    if source=='auto':
        source = methods.keys()
    elif isinstance(source, str):
        source = [source]
    elif isinstance(source, (list, tuple, np.ndarray)):
        pass
    else:
        raise InputValueError('source must be string or list of string')

    # 
    for src in source:
        if not src in methods.keys():
            raise InputValueError(f'source must be one of {list(methods.keys())}, but got {src}')
    
    # 
    for src in source:
        if pKa := methods[src](smiles, temperature=temperature):
            break
        
    # 
    pKa = deepcopy(pKa)
    if pKa is None:
        return None
    else:
        for xpKa in pKa.values():
            for atom_pKa in xpKa.copy():
                if np.isnan(atom_pKa['value']):
                    xpKa.remove(atom_pKa)
        return pKa
