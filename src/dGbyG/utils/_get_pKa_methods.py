"""
This module provides a function to get pKa values for a compound.
pKa data can be obtained from three sources:
(1) From a local cache file (.json file).
(2) From ChemAxon's Calculator plugins (license required).
(3) From ChemAxon's REST API (only available for small molecules).
"""
import os
import json
import gzip
import jpype
import shutil
import sqlite3
import requests
import portalocker
import multiprocessing
import numpy as np
from copy import deepcopy
from typing import List, Union, Iterable
from tqdm import tqdm

from ._custom_error import NoLicenseError, InputValueError
from ..__init__ import default_T


# Path settings
chemaxon_pka_db_path = os.path.join(__file__.split('src')[0], 'data', 'chemaxon_pka.db')

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


def get_pKa_from_chemaxon_rest(smiles:str, temperature:float) -> dict:
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


def _batch_get_pKa_using_chemaxon_java(smiles_list:List[str], temperature:float) -> List[str|dict]:
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
        
    for smiles in smiles_list:
        try:
            mol=MolImporter.importMol(smiles)
            pKa.setMolecule(mol)
            if pKa.run():
                apka, bpka = {}, {}
                for i in range(mol.getAtomCount()):
                    a, b = float(pKa.getpKa(i, pKa.ACIDIC)), float(pKa.getpKa(i, pKa.BASIC))
                    if not np.isnan(a):
                        apka[i] = a
                    if not np.isnan(b):
                        bpka[i] = b
                res = {'SMILES': smiles, 'acidicValuesByAtom':apka, 'basicValuesByAtom':bpka}
            else:
                res = {'SMILES': smiles, 'error': "pKa calculation failed"}
        except:
            res = {'SMILES': smiles, 'error': "pKa calculation failed"}
        output.append(res)
        
    jpype.shutdownJVM()
    # print('shut down JVM!')
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
    # check if ChemAxon jar files exist
    if not os.path.isdir(chemaxon_jar_dir):
        raise FileNotFoundError("ChemAxon jar files not found")
    # check if ChemAxon license file exist
    if not os.path.isfile(chemaxon_license_file_path):
        raise NoLicenseError("ChemAxon license not found")
    # check if smiles is a string
    if not isinstance(smiles, str):
        raise InputValueError("get_pKa_from_chemaxon(smiles:str, temperature:float=default_T), smiles must be a string")
    
    # get pKa values from chemaxon
    queue = multiprocessing.Queue()
    func = lambda queue, smiles, temperature: queue.put(_batch_get_pKa_using_chemaxon_java([smiles], temperature))
    p = multiprocessing.Process(target=func, args=(queue, smiles, temperature, ))
    p.start()
    p.join()
    _, pKa = queue.get()[0]
    
    if pKa == "pKa calculation failed":
        print(smiles, pKa)
        return None
    elif isinstance(pKa, dict):
        return pKa
    else:
        raise Exception(f"Unknown error, return value: {pKa}")


def batch_get_pKa_from_chemaxon(smiles:Union[str, List[str]], temperature:float, batch_size:int = 100) -> Union[dict, None]:
    """
    Get pKa values for a single SMILES from ChemAxon's Calculator plugins (license required).

    Parameters
    ----------
    smiles : string or list of strings
        A SMILES string of the compound or a list of that.
    temperature : float, optional
        Temperature in Kelvin.
    batch_size : int, optional
        Number of SMILES per batch (default=100).

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
    # check if ChemAxon jar files exist
    if not os.path.isdir(chemaxon_jar_dir):
        raise FileNotFoundError("ChemAxon jar files not found")
    # check if ChemAxon license file exist
    if not os.path.isfile(chemaxon_license_file_path):
        raise NoLicenseError("ChemAxon license not found")
    
    # check if smiles is a string or a list of strings
    if isinstance(smiles, str):
        smiles_list = [smiles]
    elif isinstance(smiles, list):
        if all(isinstance(i, str) for i in smiles):
            smiles_list = smiles
        else:
            raise InputValueError("get_pKa_from_chemaxon(smiles:str, temperature:float=default_T), smiles must be a string or a list of strings")
    else:
        raise InputValueError("get_pKa_from_chemaxon(smiles:str, temperature:float=default_T), smiles must be a string or a list of strings")
    
    # Split into batches
    batches = []
    for i in range(0, len(smiles_list), batch_size):
        batches.append(smiles_list[i:i + batch_size])
    
    total_smiles = len(smiles_list)
    all_results = []
    
    # Use tqdm to show progress by SMILES count
    with tqdm(total=total_smiles, desc='Calculating pKa values using ChemAxon Java API', unit="smiles") as pbar:
        for batch in batches:

            # get pKa values from chemaxon
            queue = multiprocessing.Queue()
            func = lambda queue, smiles_list, temperature: queue.put(_batch_get_pKa_using_chemaxon_java(smiles_list, temperature))
            p = multiprocessing.Process(target=func, args=(queue, batch, temperature, ))
            p.start()
            p.join(timeout=300)  # 5 minutes timeout
            
            if p.is_alive():
                p.terminate()
                p.join()
                print(f"⚠️ Batch timeout ({len(batch)} SMILES)")
                pbar.update(len(batch))
                continue
            try:
                batch_results = queue.get(timeout=10)
                all_results.extend(batch_results)
                pbar.update(len(batch))
            except Exception as e:
                print(f"❌ Batch failed: {e}")
                pbar.update(len(batch))
                
    return all_results
    

def save_pka_to_db(pka_list):
    with sqlite3.connect(chemaxon_pka_db_path) as conn:
        cur = conn.cursor()

        # 1️⃣ 建表（只关心结构）
        cur.execute("""
        CREATE TABLE IF NOT EXISTS pKa (
            SMILES TEXT PRIMARY KEY,
            acidicValuesByAtom TEXT,
            basicValuesByAtom TEXT
        )
        """)

        # 2️⃣ 准备批量数据
        records = [
            (
                d["SMILES"],
                json.dumps(d["acidicValuesByAtom"]),
                json.dumps(d["basicValuesByAtom"]),
            )
            for d in pka_list
        ]

        # 3️⃣ 批量写入（主流写法）
        cur.executemany(
            """
            INSERT OR REPLACE INTO pKa (SMILES, acidicValuesByAtom, basicValuesByAtom)
            VALUES (?, ?, ?)
            """,
            records
        )

        # 4️⃣ 显式提交（心里有底）
        conn.commit()


def load_pka_by_smiles(smiles: str, temperature: float = default_T) -> Union[dict, None]:
    if temperature != default_T:
        raise NotImplementedError("Only default temperature is supported")
    
    with sqlite3.connect(chemaxon_pka_db_path) as conn:
        cur = conn.cursor()

        cur.execute("""
            SELECT acidicValuesByAtom, basicValuesByAtom
            FROM pKa
            WHERE SMILES = ?
        """, (smiles,))

        row = cur.fetchone()
        if row is not None:
            acidic = [{'atomIndex': int(k), 'value': v} for k, v in json.loads(row[0]).items()]
            basic = [{'atomIndex': int(k), 'value': v} for k, v in json.loads(row[1]).items()]
            return {
                "acidicValuesByAtom": acidic,
                "basicValuesByAtom": basic,
            }
        else:
            return None


def batch_predict_and_save_pka(
    smiles: Union[str, List[str]], 
    temperature: float = default_T,
    recalc_existing: bool = False
) -> List[dict]:
    """
    批量预测 pKa 并保存到数据库
    
    Parameters
    ----------
    smiles : str or List[str]
        单个 SMILES 或 SMILES 列表
    temperature : float
        温度（Kelvin）
    recalc_existing : bool, default=False
        是否重新预测已存在于数据库中的分子
    
    Returns
    -------
    List[dict]
        成功预测的 pKa 数据列表
    """
    # 1️⃣ 处理输入 SMILES
    if isinstance(smiles, str):
        smiles_list = [smiles]
    elif isinstance(smiles, list):
        if all(isinstance(i, str) for i in smiles):
            smiles_list = smiles
        else:
            raise InputValueError("smiles must be a string or a list of strings")
    else:
        raise InputValueError("smiles must be a string or a list of strings")
    
    # 2️⃣ 检查数据库中已存在的分子
    existing_smiles = set()
    if os.path.isfile(chemaxon_pka_db_path):
        with sqlite3.connect(chemaxon_pka_db_path) as conn:
            cur = conn.cursor()
            placeholders = ','.join(['?'] * len(smiles_list))
            cur.execute(f"SELECT SMILES FROM pKa WHERE SMILES IN ({placeholders})", smiles_list)
            existing_smiles = {row[0] for row in cur.fetchall()}
    
    # 3️⃣ 决定要预测的 SMILES
    if recalc_existing:
        # 重新预测所有分子
        to_predict = smiles_list
        print(f"🔄 将重新预测所有 {len(to_predict)} 个分子（包括已存在的）")
    else:
        # 只预测不存在的分子
        to_predict = [smi for smi in smiles_list if smi not in existing_smiles]
        print(f"📊 数据库中已存在 {len(existing_smiles)} 个分子，将预测 {len(to_predict)} 个新分子")
    
    if not to_predict:
        print("✅ 所有分子都已存在于数据库中，无需预测")
        return []
    
    # 4️⃣ 批量预测 pKa
    try:
        pka_results = []
        for pka in batch_get_pKa_from_chemaxon(to_predict, temperature):
            if 'error' in pka:
                print(f"❌ 预测失败: {pka['error']}")
            else:
                pka_results.append(pka)
    except (FileNotFoundError, NoLicenseError, InputValueError) as e:
        print(f"❌ 预测失败: {e}")
        return []
    
    if not pka_results:
        print("⚠️ 没有获取到 pKa 数据")
        return []
    
    # 5️⃣ 保存到数据库
    try:
        save_pka_to_db(pka_results)
        print(f"✅ 成功保存 {len(pka_results)} 个分子的 pKa 数据")
        return pka_results
    except Exception as e:
        print(f"❌ 保存到数据库失败: {e}")
        return []


def get_pKa_methods():
    methods = {}
    # if pka json file exists, add method to get pka from json file
    # if os.path.isfile(chemaxon_pka_json_path) or os.path.isfile(chemaxon_pka_json_path.removesuffix(".gz")):
    #     methods['chemaxon_pKa_json'] = get_pKa_from_json

    # if pka db file exists, add method to get pka from db file
    if os.path.isfile(chemaxon_pka_db_path):
        methods['chemaxon_pKa_db'] = load_pka_by_smiles

    # if chemaxon jar files and license file exist, add method to get pka from chemaxon
    if os.path.isdir(chemaxon_jar_dir) and os.path.isfile(chemaxon_license_file_path):
        methods['chemaxon'] = get_pKa_from_chemaxon

    # 
    methods['chemaxon_rest'] = get_pKa_from_chemaxon_rest
    return methods


def get_pKa(smiles: str, temperature: float = default_T, source: Union[str, List[str]] = 'auto') -> dict:
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
    if not isinstance(smiles, str):
        print('Input smiles must be a string')
        return None
    
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
