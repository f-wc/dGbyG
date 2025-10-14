import os


class Config:
    def __init__(self):
        # 
        self.kegg_database_path = os.path.join(__file__.split('src')[0], 'data', 'KEGG')
        self.hmdb_database_path = os.path.join(__file__.split('src')[0], 'data', 'HMDB')
        self.bigg_database_path = os.path.join(__file__.split('src')[0], 'data', 'BiGG')
        self.chebi_database_path = os.path.join(__file__.split('src')[0], 'data', 'libChEBI')
        self.lipidmaps_database_path = os.path.join(__file__.split('src')[0], 'data', 'LIPID_MAPS')
        self.pubchem_database_path = os.path.join(__file__.split('src')[0], 'data', 'PubChem')
        self.metanetx_database_path = os.path.join(__file__.split('src')[0], 'data', 'MetaNetX')
        self.recon3d_database_path = os.path.join(__file__.split('src')[0], 'data', 'Recon3D')
    

config = Config()