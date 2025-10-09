import os
import zipfile
import pandas as pd
from io import BytesIO, StringIO
from rdkit import Chem

from ..utils.config import config

def raw_file_to_csv(zip_file='structures.zip'):
    """
    # raw data was downloaded from https://hmdb.ca/downloads
    """
    zip_path = os.path.join(config.hmdb_database_path, zip_file)
    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        sdf_files = [f for f in zf.namelist() if f.lower().endswith('.sdf')]
        for file in sdf_files:
            with zf.open(file) as f:
                f = BytesIO(f.read())
                supplier = Chem.ForwardSDMolSupplier(f)
    
    csv_buffer = StringIO()
    csv_buffer.write(f"# {zip_file}\n")

    data = []
    for mol in supplier:
        if mol is not None:
            data.append([mol.GetProp('HMDB_ID'), mol.GetProp('SMILES')])
    df = pd.DataFrame(data, columns=['HMDB_ID', 'SMILES'])
    df.to_csv(csv_buffer, mode='a', index=False, header=True)
    
    # write to file
    csv_path = os.path.join(config.hmdb_database_path, 'structures.csv.zip')
    print(f"Writing {csv_path}")
    with zipfile.ZipFile(csv_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr(csv_path, csv_buffer.getvalue())
    print(f"Done.")