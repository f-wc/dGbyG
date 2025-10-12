import os
import zipfile
import pandas as pd
from io import StringIO

from ..utils.config import config


def raw_file_to_csv(file='structures.tsv.gz'):
    """
    # raw data was downloaded from https://www.ebi.ac.uk/chebi/downloads
    """
    dir_path = config.chebi_database_path
    file_path = os.path.join(dir_path, file)
    print(f"Extracting {file_path}")
    df =df = pd.read_csv(file_path, sep='\t', dtype=str)
    df = df.loc[:, ['compound_id', 'smiles']].rename(columns={'compound_id': 'ChEBI_ID', 'smiles':'SMILES'})

    # 
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, mode='a', index=False, header=True)
    
    # write to file
    csv_path = os.path.join(dir_path, 'structures.csv.zip')
    print(f"Writing {csv_path}")
    with zipfile.ZipFile(csv_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr(csv_path, csv_buffer.getvalue())
    print(f"Done.")