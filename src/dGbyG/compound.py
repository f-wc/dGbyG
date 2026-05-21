"""
Compound class – represents a chemical compound and its thermodynamic properties.
"""
import math
import base64
from io import BytesIO

from rdkit import Chem
from rdkit.Chem.rdchem import Mol
import numpy as np
from typing import Dict, Tuple, Union, List
from functools import lru_cache
from torch_geometric.data import Data

from .utils import to_mol, transformed_ddGf, get_pKa, is_chemaxon_java_available
from .utils.mol_utils import normalize_mol, atom_bag, parse_cid
from .utils._custom_error import NoPkaError, InputValueError
from .model.datasets import mol_to_graph_data
from .model.inference import Inference_Model
from .utils._to_mol_methods import name_to_smiles, to_mol_methods
from .constants import default_T, default_pMg, default_pH, default_I, default_e_potential, default_condition, R
from ._globals import pKa_source, model_cache
from .config import config


class Compound(object):

    recognizable_cids = list(to_mol_methods().keys())

    def __init__(self, mol: Union[Mol, str, None], cid_type: Union[str, None] = None) -> None:
        """
        Parameters
        ----------
        mol : Mol, str, or None
            The molecule input. Can be an rdkit Mol object, a string identifier,
            or None (for placeholder compounds).
        cid_type : str or None
            The type of the string identifier (e.g. 'smiles', 'kegg', 'inchi key').
            Required when *mol* is a string.
        """
        #
        self.input = mol

        if isinstance(mol, Mol) or (mol is None):
            self.raw_mol = mol
        elif isinstance(mol, str) and isinstance(cid_type, str):
            if cid_type.lower() in ['fuzzy name', 'fuzzy_name']:
                Candidates = [comp for comp in [Compound(x, 'smiles') for x in name_to_smiles(mol)] if comp.mol is not None]
                raw_mol = sorted(Candidates, key=lambda comp: comp.standard_dGf_prime)[0].mol if Candidates else None
                self.raw_mol = raw_mol
            else:
                self.raw_mol = to_mol(mol, cid_type)
        elif isinstance(mol, str) and cid_type is None:
            raise InputValueError(f"Please specify the type of {mol} with 'cid_type='.")
        else:
            raise InputValueError('The input of Compound() must be rdkit.Chem.rdchem.Mol, string, or None.')

        # Normalize mol
        if isinstance(self.raw_mol, Mol):
            self.mol = normalize_mol(self.raw_mol)
        elif self.raw_mol is None:
            self.mol = None
        else:
            raise InputValueError(f'Unknown error in Compound.__init__().')

        self.compartment = None
        self._concentration = 1
        self._l_concentration = None
        self._u_concentration = None
        self._z = 0
        self._lz = None
        self._uz = None
        self._condition = default_condition.copy()

        self.pKa_source = None

    # ------------------------------------------------------------------
    # Molecular identifier properties
    # ------------------------------------------------------------------
    @property
    def Smiles(self) -> str:
        return Chem.MolToSmiles(Chem.RemoveHs(self.mol), canonical=True) if self.mol else None

    @property
    def InChI(self) -> str:
        return Chem.MolToInchi(self.mol) if self.mol else None

    @property
    def InChIKey(self) -> str:
        return Chem.MolToInchiKey(self.mol) if self.mol else None

    @property
    def image(self):
        if None:
            for atom in self.mol.GetAtoms():
                atom.SetProp("atomNote", str(atom.GetIdx()))
                # atom.SetProp('molAtomMapNumber',str(atom.GetIdx()))
        return Chem.Draw.MolToImage(Chem.RemoveHs(self.mol)) if self.mol else None

    @property
    def atom_bag(self) -> Dict[str, int | float]:
        return atom_bag(self.mol) if self.mol else None

    # ------------------------------------------------------------------
    # Condition (pH, pMg, I, e_potential, T)
    # ------------------------------------------------------------------
    @property
    def condition(self) -> Dict[str, float]:
        return self._condition

    @condition.setter
    def condition(self, condition: Dict[str, float | int]):
        if not isinstance(condition, dict):
            raise InputValueError('The input of condition must be a dict.')
        elif x := set(condition.keys()) - set(self.condition.keys()):
            raise InputValueError(f'Condition includes {", ".join(self.condition.keys())}, but got {", ".join(x)}.')
        elif condition.get('T', 298.15) != 298.15:
            raise InputValueError('The temperature cannot be changed and must be 298.15 K.')
        else:
            for k, v in condition.items():
                if isinstance(v, (int, float)):
                    self._condition[k] = float(v)
                else:
                    raise InputValueError(f'The value of {k} must be a float or int, but got {type(condition[k])}.')

    # ------------------------------------------------------------------
    # Concentration & charge (z) properties
    # ------------------------------------------------------------------
    @property
    def z(self) -> float:
        return self._z

    @z.setter
    def z(self, z: float):
        self._z = z
        self._concentration = 10 ** self._z

    @property
    def uz(self) -> float:
        return self._uz

    @uz.setter
    def uz(self, uz: float):
        if self.Smiles in ['[H+]', '[1H+]']:
            self._uz = -self.condition['pH']
        else:
            self._uz = uz
            self._u_concentration = 10 ** uz

    @property
    def lz(self) -> float:
        return self._lz

    @lz.setter
    def lz(self, lz: float):
        self._lz = lz
        self._l_concentration = 10 ** lz

    @property
    def concentration(self) -> float:
        return self._concentration

    @concentration.setter
    def concentration(self, concentration: float):
        self._concentration = concentration
        self._z = np.log10(self._concentration)

    @property
    def u_concentration(self) -> float:
        return self._u_concentration

    @u_concentration.setter
    def u_concentration(self, concentration: float):
        if self.Smiles in ['[H+]', '[1H+]']:
            pass
        else:
            self._u_concentration = concentration
            self._uz = np.log10(concentration)

    @property
    def l_concentration(self) -> float:
        return self._l_concentration

    @l_concentration.setter
    def l_concentration(self, concentration: float):
        self._l_concentration = concentration
        self._lz = np.log10(concentration)

    # ------------------------------------------------------------------
    # pKa
    # ------------------------------------------------------------------
    @lru_cache(16)
    def pKa(self, temperature=default_T, source: Union[str, List[str], None] = None) -> Union[dict, None]:
        if self.Smiles in ['[H+]', '[1H+]']:
            return {'acidicValuesByAtom': [{'atomIndex': 0, 'value': np.nan}],
                    'basicValuesByAtom': [{'atomIndex': 0, 'value': np.nan}]}
        else:
            if source is not None:
                pass
            elif self.pKa_source is not None:
                source = self.pKa_source
            elif pKa_source is not None:
                source = pKa_source
            else:
                source = 'auto'
            return get_pKa(self.Smiles, temperature, source) if self.mol else None

    @property
    def can_be_transformed(self) -> bool:
        return True if self.pKa(default_T) or (self.Smiles in ['[H+]', '[1H+]']) else False

    # ------------------------------------------------------------------
    # Thermodynamic properties
    # ------------------------------------------------------------------
    @property
    def transformed_ddGf(self):
        if self.can_be_transformed == True:
            T = self.condition.get('T', default_T)
            ddGf = transformed_ddGf(
                pKa=self.pKa(T),
                pH=self.condition.get('pH', default_pH),
                T=T,
                pMg=self.condition.get('pMg', default_pMg),
                I=self.condition.get('I', default_I),
                e_potential=self.condition.get('e_potential', default_e_potential),
                charge=self.atom_bag.get('charge', 0),
                num_H=self.atom_bag.get('H', 0),
                num_Mg=self.atom_bag.get('Mg', 0)
            )
            return ddGf
        elif self.can_be_transformed == False:
            raise NoPkaError('This compound has no available Pka value, so it cannot be transformed.')
        else:
            raise ValueError('Unknown value of self.can_be_transformed')

    @property
    @lru_cache(maxsize=None)
    def graph_data(self) -> Data:
        return mol_to_graph_data(self.mol) if self.mol else None

    @property
    @lru_cache(maxsize=None)
    def standard_dGf_prime_list(self) -> np.ndarray:
        #
        if config.infer_model_path not in model_cache:
            model_cache[config.infer_model_path] = Inference_Model(config.infer_model_path)
        infer_model = model_cache[config.infer_model_path]

        #
        if self.Smiles in ['[H+]', '[1H+]']:
            return np.zeros(infer_model.num_head)
        elif self.mol:
            return infer_model(self.graph_data).squeeze().numpy()
        elif self.mol is None:
            return np.full(infer_model.num_head, np.nan)
        else:
            raise ValueError('Unknown value of self.mol')

    @property
    @lru_cache(maxsize=None)
    def standard_dGf_prime(self) -> Tuple[np.float32, np.float32]:
        return np.mean(self.standard_dGf_prime_list).item(), np.std(self.standard_dGf_prime_list).item()

    @property
    def transformed_standard_dGf_prime(self) -> Tuple[np.float32, np.float32]:
        if self.condition == default_condition:
            return self.standard_dGf_prime
        elif self.can_be_transformed:
            transformed_standard_dg = (self.standard_dGf_prime[0] + self.transformed_ddGf)
            return transformed_standard_dg, self.standard_dGf_prime[1]
        else:
            return self.standard_dGf_prime

    @property
    def transformed_dGf_prime(self) -> Tuple[np.float32, np.float32]:
        if self.Smiles in ['[H+]', '[1H+]']:
            real_dGf = self.transformed_standard_dGf_prime[0]
        else:
            real_dGf = self.transformed_standard_dGf_prime[0] + default_T * R * math.log(self.concentration)
        return real_dGf, self.transformed_standard_dGf_prime[1]

    # ------------------------------------------------------------------
    # Display methods
    # ------------------------------------------------------------------
    def __str__(self) -> str:
        return self.Smiles if self.Smiles else "None"

    def __repr__(self) -> str:
        line = "─" * 60
        info = f"\n{line}\n"
        info += f"{'化合物对象':^58}\n"
        info += f"{line}\n"
        info += f"SMILES:    {self.Smiles}\n"
        info += f"InChI:     {self.InChI}\n"
        info += f"InChIKey:  {self.InChIKey}\n"

        # 热力学数据 (使用 _f 表示下标)
        dg_mean, dg_std = self.standard_dGf_prime
        if dg_mean is not None and not np.isnan(dg_mean):
            info += f"Δ_fG'°:    {dg_mean:.1f} ± {dg_std:.1f} kJ/mol\n"

        # 状态信息
        trans_status = "Yes" if self.can_be_transformed else "No"
        info += f"Has pKa:    {trans_status}\n"

        info += f"{line}"
        return info

    def _repr_html_(self) -> str:
        # 1. 处理图片：转为 Base64 编码
        img_html = "<div style='color:#999; font-size:12px; text-align:center;'>无结构图</div>"
        if self.mol:
            try:
                img = self.image
                if img:
                    buffered = BytesIO()
                    img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    img_html = f"<img src='data:image/png;base64,{img_str}' style='max-width:200px; border-radius:4px; border:1px solid #eee;'/>"
            except Exception:
                img_html = "<div style='color:red;'>图片生成错误</div>"

        # 2. 提取关键数据
        input_obj = self.input if isinstance(self.input, str) else '化合物对象'
        smiles = self.Smiles if self.Smiles else "N/A"
        inchi = self.InChI if self.InChI else "N/A"
        inchikey = self.InChIKey if self.InChIKey else "N/A"

        dg_mean, dg_std = self.standard_dGf_prime
        dg_text = f"{dg_mean:.1f} ± {dg_std:.1f} kJ/mol" if not np.isnan(dg_mean) else "N/A"

        pka_status = "✅ Yes" if self.can_be_transformed else "❌ No"
        pka_color = "#28a745" if self.can_be_transformed else "#dc3545"

        # 3. HTML 结构：左右布局
        html = f"""
        <div style="font-family: 'Segoe UI', 'Microsoft YaHei', Arial, sans-serif;
                    background-color: #ffffff; color: #333;
                    border: 1px solid #e1e4e8; border-radius: 8px;
                    padding: 15px; margin: 8px 0;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
                    display: inline-block; min-width: 450px;">
            <h4 style="margin: 0 0 10px 0; color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 8px;">
                🧪 {input_obj}
            </h4>
            <div style="display: flex; flex-direction: row; gap: 20px; align-items: center;">
                <div style="flex-shrink: 0; background: #f8f9fa; padding: 5px; border-radius: 4px;">
                    {img_html}
                </div>
                <div style="flex-grow: 1; font-size: 13px; line-height: 1.6;">
                    <div style="margin-bottom: 4px;">
                        <span style="font-weight: bold; color: #555;">SMILES:</span>
                        <code style="background:#f5f5f5; padding:2px 4px; border-radius:3px;">{smiles}</code>
                    </div>
                    <div style="margin-bottom: 4px;">
                        <span style="font-weight: bold; color: #555;">InChI:</span>
                        <code style="background:#f5f5f5; padding:2px 4px; border-radius:3px; font-size: 11px;">{inchi}</code>
                    </div>
                    <div style="margin-bottom: 4px;">
                        <span style="font-weight: bold; color: #555;">InChIKey:</span> {inchikey}
                    </div>
                    <div style="margin-bottom: 4px;">
                        <span style="font-weight: bold; color: #555;">Δ<sub>f</sub>G'°:</span>
                        <span style="color: #007bff; font-weight: 600;">{dg_text}</span>
                    </div>
                    <div>
                        <span style="font-weight: bold; color: #555;">Has pKa:</span>
                        <span style="color: {pka_color}; font-weight: 600;">{pka_status}</span>
                    </div>
                </div>
            </div>
        </div>
        """
        return html
