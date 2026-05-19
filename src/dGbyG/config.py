import os
from pathlib import Path

class Config:
    def __init__(self):
        # 
        datasets_dir = Path(__file__.split('src')[0]) / 'data'
        self.kegg_database_path = datasets_dir / 'KEGG'
        self.hmdb_database_path = datasets_dir / 'HMDB'
        self.bigg_database_path = datasets_dir / 'BiGG'
        self.chebi_database_path = datasets_dir / 'ChEBI'
        self.lipidmaps_database_path = datasets_dir / 'LIPID_MAPS'
        self.pubchem_database_path = datasets_dir / 'PubChem'
        self.metanetx_database_path = datasets_dir / 'MetaNetX'
        self.recon3d_database_path = datasets_dir / 'Recon3D'
        self.chemaxon_pka_db_path = datasets_dir / 'chemaxon_pka.db'

        # 


config = Config()