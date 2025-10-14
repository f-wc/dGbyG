import os, time
from datetime import datetime
from urllib.error import HTTPError
from typing import Dict, List
from Bio.KEGG import REST
#import Bio.KEGG.Compound

from ..utils.config import config



def download_compound_mol(entry:str, force:bool=False) -> bool:
    """
    The function download mol file from KEGG database.

    Parameters:
    ----------
    entry: str
        The entry of the compound in KEGG compound database.
    force: bool
        If True, the function will download the mol file even if it already exists.

    Returns:
    -------
    bool
        If the mol file is downloaded successfully, return True. If the mol file download failed, return False. If the mol file already exists, return None.
    """
    file_name = entry + '.mol'
    file_path = os.path.join(config.kegg_database_path, 'mol', file_name)
    
    # judge if the direaction kegg_compound exists
    if not os.path.isdir(config.kegg_database_path):
        os.mkdir(config.kegg_database_path)
    if not os.path.isdir(os.path.join(config.kegg_database_path, 'mol')):
        os.mkdir(os.path.join(config.kegg_database_path, 'mol'))
        
    # judge if the mol file exists, if not then download the mol file
    if not os.path.isfile(file_path) or force:
        try:
            mol = REST.kegg_get(entry+'/mol')
            with open(file_path, 'w') as f:
                f.write(mol.read())
                print(entry, 'download successful!')
            return True
        except:
            return False
    else:
        print(entry, 'already exists')
        return None
        
"""
def download_compound(entry:str, force:bool=False) -> None:
    file_path = os.path.join(kegg_database_path, 'compound', f'{entry}.txt')
    if force or not os.path.exists(file_path):
        t = REST.kegg_get(entry).read()
        with open(file_path, 'w') as f:
            f.write(t)
        print(entry, 'successfully downloaded')
        return True
    else:
        print(entry, 'already exists')
        return True
"""


def batch_download_compound(entry_list:list, force:bool=False):
    # 
    for entry in entry_list:
        try:
            download_compound_mol(entry, force)
        except HTTPError as e:
            if e.code == 404:
                print('no', entry)
            else:
                print(entry, 'error', e.code)
        except BaseException as e:
            print(entry, 'error', e)
        time.sleep(0.01)



def download_all_compound(force=False):
    """
    Download all compounds from KEGG compound database.

    Parameters:
    ----------
    force: bool
        If True, the function will download the mol file even if it already exists.
    """
    entry_list = REST.kegg_list('compound').readlines()
    entry_list = [entry.split('\t')[0] for entry in entry_list]
    batch_download_compound(entry_list, force=force)

    date = datetime.now().strftime(r"%Y.%m.%d %H:%M:%S")
    print(f'{date} All compounds downloaded!!!')

    with open(os.path.join(config.kegg_database_path, 'mol', 'download.log'), 'w') as f:
        f.write(f"{date} {entry_list}")