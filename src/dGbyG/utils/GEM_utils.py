import os
import functools
import cobra
import pandas as pd

from .. import default_T, default_I, default_pMg, default_pH, default_e_potential



class GEM_library(object):
    def __init__(self, ):
        # get the path of the current script
        self.package_path = os.path.abspath(__file__).split('src')[0]
        self.GEM_dir = os.path.join(self.package_path, 'data', 'GEMs')
        
    @functools.cached_property
    def Recon3D(self, path):
        dir_path = os.path.join(self.script_dir, 'Recon3D')
        gem = cobra.io.load_matlab_model(os.path.join(dir_path, 'Recon3D_301.mat'))
        gem.S = cobra.util.array.create_stoichiometric_matrix(gem)
        gem.transformed_standard_dGr = pd.read_csv(os.path.join(dir_path, 'Recon3D_standard_dGr_dGbyG.csv'), index_col=0)
        gem.compartment_conditions = {
            'c':{'pH':7.20, 'e_potential':0, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
            'e':{'pH':7.40, 'e_potential':30 * 1e-3, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
            'n':{'pH':7.20, 'e_potential':0, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
            'r':{'pH':7.20, 'e_potential':0, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
            'g':{'pH':6.35, 'e_potential':0, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
            'l':{'pH':5.50, 'e_potential':19 * 1e-3, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
            'm':{'pH':8.00, 'e_potential':-155 * 1e-3, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
            'i':{'pH':8.00, 'e_potential':-155 * 1e-3, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
            'x':{'pH':7.00, 'e_potential':12 * 1e-3, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
            }
        return gem

    @functools.cached_property
    def Human1(self):
        dir_path = os.path.join(self.script_dir, 'Human1')
        gem = cobra.io.read_sbml_model(os.path.join(dir_path, 'Human-GEM/model/Human-GEM.xml'))
        gem.S = cobra.util.array.create_stoichiometric_matrix(gem)
        gem.transformed_standard_dGr = pd.read_csv(os.path.join(dir_path, 'Human1_standard_dGr_dGbyG.csv'), index_col=0)
        gem.compartment_conditions = {
            'c':{'pH':7.20, 'e_potential':0, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
            'e':{'pH':7.40, 'e_potential':30 * 1e-3, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
            'n':{'pH':7.20, 'e_potential':0, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
            'r':{'pH':7.20, 'e_potential':0, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
            'g':{'pH':6.35, 'e_potential':0, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
            'l':{'pH':5.50, 'e_potential':19 * 1e-3, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
            'm':{'pH':8.00, 'e_potential':-155 * 1e-3, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
            'i':{'pH':8.00, 'e_potential':-155 * 1e-3, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
            'x':{'pH':7.00, 'e_potential':12 * 1e-3, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
            }
        return gem
    
    @functools.cached_property
    def iCHOv1(self):
        dir_path = os.path.join(self.script_dir, 'iCHOv1')
        gem = cobra.io.read_sbml_model(os.path.join(self.script_dir, 'iCHOv1/iCHOv1.xml'))
        gem.S = cobra.util.array.create_stoichiometric_matrix(gem)
        gem.transformed_standard_dGr = pd.read_csv(os.path.join(dir_path, 'iCHOv1_standard_dGr_dGbyG.csv'), index_col=0)
        gem.compartment_conditions = {
            'c':{'pH':7.20, 'e_potential':0, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
            'e':{'pH':7.40, 'e_potential':30 * 1e-3, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
            'n':{'pH':7.20, 'e_potential':0, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
            'r':{'pH':7.20, 'e_potential':0, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
            'g':{'pH':6.35, 'e_potential':0, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
            'l':{'pH':5.50, 'e_potential':19 * 1e-3, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
            'm':{'pH':8.00, 'e_potential':-155 * 1e-3, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
            'im':{'pH':8.00, 'e_potential':-155 * 1e-3, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
            'x':{'pH':7.00, 'e_potential':12 * 1e-3, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
            }
        return gem
    
    @functools.cached_property
    def Yeast9(self):
        dir_path = os.path.join(self.script_dir, 'Yeast9')
        gem = cobra.io.read_sbml_model(os.path.join(self.script_dir, 'Yeast9/yeast-GEM.xml'))
        gem.S = cobra.util.array.create_stoichiometric_matrix(gem)
        gem.transformed_standard_dGr = pd.read_csv(os.path.join(dir_path, 'Yeast9_standard_dGr_dGbyG.csv'), index_col=0)
        return gem

    @functools.cached_property
    def iML1515(self):
        dir_path = os.path.join(self.script_dir, 'iML1515')
        gem = cobra.io.read_sbml_model(os.path.join(self.script_dir, 'iML1515/iML1515.xml'))
        gem.S = cobra.util.array.create_stoichiometric_matrix(gem)
        gem.transformed_standard_dGr = pd.read_csv(os.path.join(dir_path, 'iML1515_standard_dGr_dGbyG.csv'), index_col=0)
        gem.compartment_conditions = {
            'c':{'pH': 7.5, 'e_potential': -125 * 1e-3, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
            'e':{'pH': 7.0, 'e_potential': 0.0, 'T':default_T, 'I':default_I, 'pMg':default_pMg}, 
            'p':{'pH': 7.0, 'e_potential': 0.0, 'T':default_T, 'I':default_I, 'pMg':default_pMg}
            }
        return gem
    