import os
import zipfile
import pandas as pd
from io import StringIO

from ..utils.config import config


def raw_file_to_csv(file='chem_prop.tsv'):
    """
    # raw data was downloaded from https://www.metanetx.org/mnxdoc/mnxref.html
    """
    file_path = os.path.join(config.metanetx_database_path, file)
    print(f"Extracting {file_path}")
    with open(file_path, 'r') as f:
        lines_bytes = f.readlines()
    
    csv_buffer = StringIO()
    for line in lines_bytes:
        if line.startswith('#'):
            comment_line = line
            csv_buffer.write(comment_line+'\n')
        else:
            break
    # 
    df = pd.read_csv(file_path, sep='\t', comment='#', header=None)
    df.columns = comment_line.replace('#', '').strip().split('\t')
    df = df.loc[:, ['ID', 'SMILES']].rename(columns={'ID': 'MetaNetX_ID'})
    df.to_csv(csv_buffer, mode='a', index=False, header=True)
    
    # write to file
    csv_path = os.path.join(config.metanetx_database_path, 'structures.csv.zip')
    print(f"Writing {csv_path}")
    with zipfile.ZipFile(csv_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr(csv_path, csv_buffer.getvalue())
    print(f"Done.")