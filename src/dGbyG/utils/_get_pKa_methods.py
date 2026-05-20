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
from pathlib import Path

from ._custom_error import NoLicenseError, InputValueError
from ..constants import default_T
from ..config import config

_CHEMAXON_AVAILABLE = None  # Cached flag indicating whether ChemAxon is available.
chemaxon_jar_dir: Path|str|None = None
chemaxon_license_file_path: Path|str|None = None



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



def check_license():
    jpype.startJVM(f'-Dchemaxon.license.url={chemaxon_license_file_path}')
    # 动态设置根 Logger 级别
    Logger = jpype.JClass('java.util.logging.Logger')
    root = Logger.getLogger('')
    root.setLevel(jpype.JClass('java.util.logging.Level').SEVERE)
    for p in chemaxon_jar_dir.iterdir():
        jpype.addClassPath(p)
    isLicensed = jpype.JClass('chemaxon.marvin.calculations.pKaPlugin')().isLicensed()
    jpype.shutdownJVM()
    return isLicensed


def is_chemaxon_java_available():
    """
    Check if ChemAxon Java plugins are available and licensed.

    On first call, this function locates the ChemAxon jar directory and license file
    by examining the environment (``cxcalc`` in PATH, ``CHEMAXON_LICENSE_URL``,
    ``CHEMAXON_HOME``, user home). It then starts the JVM, loads the pKa plugin
    and verifies the license. The result (``True`` or ``False``) is cached in a
    module-level variable ``_CHEMAXON_AVAILABLE`` and returned on subsequent calls
    without repeating the expensive checks.

    Note
    ----
    This function reads and **sets** the following global variables in the module:
        - ``_CHEMAXON_AVAILABLE``  (bool or None)    - cached availability flag
        - ``chemaxon_jar_dir``      (str or Path)    - directory containing jar files
        - ``chemaxon_license_file_path`` (str or Path)     - path to the ChemAxon license file

    Returns
    -------
    bool
        ``True`` if the ChemAxon Java environment is ready to use, ``False`` otherwise.

    Raises
    ------
    ValueError
        If the internal cache ``_CHEMAXON_AVAILABLE`` is in an unexpected state
        (not ``None``, ``True`` or ``False``).
    """
    global _CHEMAXON_AVAILABLE

    # --- 1. Return cached result if already determined ---
    if (_CHEMAXON_AVAILABLE is True) or (_CHEMAXON_AVAILABLE is False):
        return _CHEMAXON_AVAILABLE

    # --- 2. First call: locate ChemAxon installation ---
    elif _CHEMAXON_AVAILABLE is None:
        # --- 2a. Locate jar directory via cxcalc ---
        global chemaxon_jar_dir
        if shutil.which('cxcalc'):
            cxcalc_path = shutil.which('cxcalc')
            # chemaxon_jar_dir is set to the 'lib' directory next to the cxcalc parent
            chemaxon_jar_dir = Path(cxcalc_path).resolve(strict=False).parent.parent / 'lib'
        else:
            chemaxon_jar_dir = ''

        # --- 2b. Locate license file following ChemAxon conventions ---
        # See https://docs.chemaxon.com/display/lts-iodine/license-installation.md
        global chemaxon_license_file_path
        if os.environ.get("CHEMAXON_LICENSE_URL"):
            chemaxon_license_file_path = Path(os.environ.get("CHEMAXON_LICENSE_URL"))
        elif os.environ.get("CHEMAXON_HOME"):
            # <ChemAxon_home> can be set by the CHEMAXON_HOME environment variable
            chemaxon_home = Path(os.environ.get("CHEMAXON_HOME"))
            # The default location is <ChemAxon_home>/license.cxl or <ChemAxon_home>/licenses/
            chemaxon_license_file_path = chemaxon_home / "license.cxl"
        else:
            user_home = Path(os.environ.get("HOME"))
            if os.name == 'posix':
                # <User_home>/.chemaxon on Unix-like systems
                chemaxon_license_file_path = user_home / ".chemaxon" / "license.cxl"
            elif os.name == 'nt':
                # <User_home>/chemaxon on Windows
                chemaxon_license_file_path = user_home / "chemaxon" / "license.cxl"
            else:
                chemaxon_license_file_path = ''

        # --- 2c. Perform the actual JVM‑based license check ---
        if not (os.path.isdir(chemaxon_jar_dir) and os.path.isfile(chemaxon_license_file_path)):
            _CHEMAXON_AVAILABLE = False
        else:
            try:
                _CHEMAXON_AVAILABLE = multiprocessing.Pool(1).apply(check_license, ())
            except:
                _CHEMAXON_AVAILABLE = False
        return _CHEMAXON_AVAILABLE

    # --- 3. Unexpected cache state ---
    else:
        raise ValueError("Error in checking ChemAxon Java availability")


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
    elif not os.path.isfile(chemaxon_license_file_path):
        raise NoLicenseError("ChemAxon license not found")
    else:
        pass
    
    # 
    jpype.startJVM(f'-Dchemaxon.license.url={chemaxon_license_file_path}')
    # 动态设置根 Logger 级别
    Logger = jpype.JClass('java.util.logging.Logger')
    root = Logger.getLogger('')
    root.setLevel(jpype.JClass('java.util.logging.Level').SEVERE)
    for p in chemaxon_jar_dir.iterdir():
        jpype.addClassPath(p)

    MolImporter = jpype.JClass('chemaxon.formats.MolImporter')
    pKaPlugin = jpype.JClass('chemaxon.marvin.calculations.pKaPlugin')
    pKa = pKaPlugin()
    if not pKa.isLicensed():
        jpype.shutdownJVM()
        raise NoLicenseError("ChemAxon license not found")
    
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
                res = {'SMILES': smiles, 'acidicValuesByAtom':None, 'basicValuesByAtom':None}
        except:
            res = {'SMILES': smiles, 'acidicValuesByAtom':None, 'basicValuesByAtom':None}
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
    # check if ChemAxon available
    if not is_chemaxon_java_available():
        raise RuntimeError("ChemAxon Java not available")
    
    # check if smiles is a string
    if not isinstance(smiles, str):
        raise InputValueError("get_pKa_from_chemaxon(smiles:str, temperature:float=default_T), smiles must be a string")
    
    # get pKa values from chemaxon
    queue = multiprocessing.Queue()
    func = lambda queue, smiles, temperature: queue.put(_batch_get_pKa_using_chemaxon_java([smiles], temperature))
    p = multiprocessing.Process(target=func, args=(queue, smiles, temperature, ))
    p.start()
    p.join()
    pKa = queue.get()[0]
    acidic = pKa['acidicValuesByAtom']
    basic = pKa['basicValuesByAtom']
    
    if (acidic is None) and (basic is None):
        return None
    elif (acidic is not None) and (basic is not None):
        return {
            "acidicValuesByAtom": [{'atomIndex': int(k), 'value': v} for k, v in acidic.items()],
            "basicValuesByAtom": [{'atomIndex': int(k), 'value': v} for k, v in basic.items()],
        }
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
    # check if ChemAxon available
    if not is_chemaxon_java_available():
        raise RuntimeError("ChemAxon Java not available")
    
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
    """
    Persist a list of pKa records into the SQLite database.
    
    Parameters
    ----------
    pka_list : list of dict
        Each dict must contain the keys "SMILES", "acidicValuesByAtom",
        and "basicValuesByAtom". The acidic/basic values are expected to
        be dictionaries (or None) and will be serialized to JSON strings.
    """
    with sqlite3.connect(config.chemaxon_pka_db_path) as conn:
        cur = conn.cursor()

        # 1. Ensure the table exists (creates it if missing)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS pKa (
            SMILES TEXT PRIMARY KEY,
            acidicValuesByAtom TEXT,
            basicValuesByAtom TEXT
        )
        """)

        # 2. Prepare batch data: serialize atom-value dicts to JSON
        records = [
            (
                d["SMILES"],
                json.dumps(d["acidicValuesByAtom"]),
                json.dumps(d["basicValuesByAtom"]),
            )
            for d in pka_list
        ]

        # 3. Bulk upsert – INSERT OR REPLACE handles existing SMILES
        cur.executemany(
            """
            INSERT OR REPLACE INTO pKa (SMILES, acidicValuesByAtom, basicValuesByAtom)
            VALUES (?, ?, ?)
            """,
            records
        )

    # 4. Close the database connection to release resources
    conn.close()
    return


def load_pka_by_smiles(smiles: str, temperature: float = default_T) -> Union[dict, None]:
    if temperature != default_T:
        raise NotImplementedError("Only default temperature is supported")
    
    with sqlite3.connect(config.chemaxon_pka_db_path) as conn:
        cur = conn.cursor()

        cur.execute("""
            SELECT acidicValuesByAtom, basicValuesByAtom
            FROM pKa
            WHERE SMILES = ?
        """, (smiles,))

        row = cur.fetchone()
        if row is not None:
            acidic = json.loads(row[0])
            basic = json.loads(row[1])
            if (acidic is None) and (basic is None):
                return None
            elif (acidic is not None) and (basic is not None):
                return {
                    "acidicValuesByAtom": [{'atomIndex': int(k), 'value': v} for k, v in acidic.items()],
                    "basicValuesByAtom": [{'atomIndex': int(k), 'value': v} for k, v in basic.items()],
                }
            else:
                raise ValueError("Inconsistent data in database: acidicValuesByAtom and basicValuesByAtom should both be None or both be not None")
        else:
            return None


def batch_predict_and_save_pka(
    smiles: Union[str, List[str]],
    temperature: float = default_T,
    recalc_existing: bool = False
) -> List[dict]:
    """
    Batch predict pKa values for one or more molecules and persist them to the database.

    This function normalizes the input SMILES strings, checks which molecules already exist
    in the database, and predicts pKa values only for those that are missing (unless
    ``recalc_existing=True`` forces re‑prediction for all). Predictions are performed via
    ChemAxon, and the results are saved into a local SQLite database.

    Parameters
    ----------
    smiles : str or list of str
        A single SMILES string or a list of SMILES strings representing the molecules.
    temperature : float, optional
        Temperature in Kelvin at which the pKa values are calculated.
        Defaults to ``default_T`` (module‑level constant).
    recalc_existing : bool, optional
        If ``False`` (default), only molecules not already present in the database are
        processed. If ``True``, all given molecules are re‑predicted, regardless of whether
        they already exist in the database. Note that re‑prediction may lead to duplicate
        entries or overwrites depending on the implementation of ``save_pka_to_db``.

    Returns
    -------
    list of dict
        A list of dictionaries containing the successfully predicted and saved pKa data.
        Each dictionary typically includes the SMILES string, acidic dissociation values
        per atom, and basic dissociation values per atom. Returns an empty list if
        prediction fails or if no new predictions are needed.

    Raises
    ------
    RuntimeError
        If the ChemAxon Java environment is not available (raised by
        ``is_chemaxon_java_available()`` or later when prediction is attempted).
    InputValueError
        If the ``smiles`` argument is neither a string nor a list of strings, or if any
        element of the list is not a string.
    """
    # ------------------------------------------------------------------
    # 1. Normalize input: convert single string to a list of strings
    # ------------------------------------------------------------------
    if isinstance(smiles, str):
        smiles_list = [smiles]
    elif isinstance(smiles, list):
        if all(isinstance(i, str) for i in smiles):
            smiles_list = smiles
        else:
            raise InputValueError("smiles must be a string or a list of strings")
    else:
        raise InputValueError("smiles must be a string or a list of strings")

    # ------------------------------------------------------------------
    # 2. Query the database to find which SMILES already have pKa data
    # ------------------------------------------------------------------
    existing_smiles = set()
    if os.path.isfile(config.chemaxon_pka_db_path):
        with sqlite3.connect(config.chemaxon_pka_db_path) as conn:
            cur = conn.cursor()
            placeholders = ','.join(['?'] * len(smiles_list))
            cur.execute(f"SELECT SMILES FROM pKa WHERE SMILES IN ({placeholders})", smiles_list)
            existing_smiles = {row[0] for row in cur.fetchall()}

    # ------------------------------------------------------------------
    # 3. Decide which molecules need to be predicted
    # ------------------------------------------------------------------
    if recalc_existing:
        # Force prediction for all given molecules, even if they already exist in the DB
        to_predict = smiles_list
        print(f"🔄 Re-predicting pKa for all {len(to_predict)} molecules (including those already in the database)")
    else:
        # Predict only those that are not yet stored
        to_predict = [smi for smi in smiles_list if smi not in existing_smiles]

    # ------------------------------------------------------------------
    # 4. Run batch prediction via ChemAxon (if needed and available)
    # ------------------------------------------------------------------
    if to_predict and is_chemaxon_java_available():
        print("---------- ChemAxon pKa plugin is available ----------")
        # NOTE: The following print statement can be misleading when recalc_existing=True
        # because it says "new ones" but actually predicts all molecules.
        print(f"📊 {len(existing_smiles)} molecules already exist in the pKa database. Predicting {len(to_predict)} new ones.")
        try:
            pka_results = list(batch_get_pKa_from_chemaxon(to_predict, temperature))
        except (FileNotFoundError, NoLicenseError, InputValueError) as e:
            print(f"❌ Prediction failed: {e}")
            return []
    elif not to_predict:
        print("✅ All molecules exist in the pKa database — skipping prediction.")
        return []
    elif not is_chemaxon_java_available():
        print("❌ ChemAxon Java environment is not available — skipping prediction.")
        return []
    else:
        return []

    # ------------------------------------------------------------------
    # 5. Save the successfully predicted results into the database
    # ------------------------------------------------------------------
    if not pka_results:
        print("-------------- No pKa data was obtained --------------")
        return []
    else:
        save_pka_to_db(pka_results)
        print("-------- pKa data has been saved successfully --------")
        return pka_results


def get_pKa_methods(source = 'auto'):
    """Return a dict of available pKa prediction methods based on current environment.

    Checks for local ChemAxon database files and Java plugin availability to
    dynamically build a mapping from method name to callable.

    Returns:
        dict: Mapping of method key (str) to the corresponding function.
    """
    methods = {}

    # Use the local ChemAxon pKa database.
    methods['chemaxon_pKa_db'] = load_pka_by_smiles

    # Use the ChemAxon Java plugin if available.
    if source in ['auto', 'chemaxon'] and is_chemaxon_java_available():
        methods['chemaxon'] = get_pKa_from_chemaxon

    # ChemAxon REST API is always available as a fallback/primary option.
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
        return pKa
