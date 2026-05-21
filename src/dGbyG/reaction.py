"""
Reaction class – represents a chemical reaction and its thermodynamic properties.
"""
import numpy as np
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdChemReactions import ChemicalReaction
from typing import Dict, Tuple, Union
from functools import lru_cache

from .utils import to_mol
from .utils.mol_utils import parse_cid
from .utils.reaction_utils import parse_equation, build_equation, atom_diff, is_balanced, read_rxn_file
from .utils._custom_error import InputValueError
from .constants import default_T, default_condition, R
from .compound import Compound


class Reaction(object):
    def __init__(self, reaction: Union[str, ChemicalReaction, Dict[Compound | Mol | str, float | int]],
                 cids_type: Union[str, None] = None) -> None:
        """
        Parameters
        ----------
        reaction: dict or str of equation, or rxn file path
            reaction can be:
            (1) a dict of {rdkit.Chem.rdchem.Mol: coefficient}.
            (2) a dict of {Compound: coefficient}.
            (3) a string of equation. e.g.
            (4) a file path of rxn file.

        cids_type: str
            'path', 'compound', 'mol', 'smiles', 'kegg', and so on.
        """
        if isinstance(cids_type, str) and ('path' in cids_type):
            reaction = read_rxn_file(reaction)
            self.raw_reaction = dict([(Compound(mol), -1) for mol in reaction.GetReactants()] + [(Compound(mol), 1) for mol in reaction.GetProducts()])
        elif isinstance(reaction, ChemicalReaction):
            self.raw_reaction = dict([(Compound(mol), -1) for mol in reaction.GetReactants()] + [(Compound(mol), 1) for mol in reaction.GetProducts()])
        elif isinstance(reaction, str):
            self.raw_reaction = parse_equation(reaction)
        elif isinstance(reaction, dict):
            self.raw_reaction = reaction
        else:
            raise InputValueError(f'Cannot accept type{type(reaction)} as the input of reaction.')

        self.reaction = {}
        for comp, coeff in self.raw_reaction.items():
            if isinstance(comp, Compound):
                pass
            elif isinstance(comp, Mol):
                comp = Compound(comp)
            elif isinstance(comp, str) and isinstance(cids_type, str):
                comp = Compound(comp, cids_type)
            elif isinstance(comp, str) and cids_type is None:
                _cid, _cid_type = parse_cid(comp)
                comp = Compound(_cid, _cid_type)

            else:
                raise InputValueError('Cannot accept type{0}'.format(type(comp)))

            if not isinstance(coeff, (float, int)):
                raise InputValueError(f"The value's type of input dict should be float or int, but got {type(coeff)}")

            self.reaction.update({comp: coeff})

        #
        self.ignore_H2O = False
        self.ignore_H_ion = True
        self.ignore_charge = False
        self.ignore_H = True

        #
        self.reaction = self.balance(self.reaction)

    # ------------------------------------------------------------------
    # Condition
    # ------------------------------------------------------------------
    @property
    def condition(self) -> Dict[str, float]:
        conditions = {}
        for comp in self.reaction.keys():
            conditions[comp] = comp.condition
        return conditions

    @condition.setter
    def condition(self, condition: Dict[str, float | int]):
        for comp in self.reaction.keys():
            comp.condition = condition

    # ------------------------------------------------------------------
    # Reaction identifiers
    # ------------------------------------------------------------------
    @property
    def rxnSmiles(self) -> dict:
        rxn_dict_smiles = map(lambda item: (item[0].Smiles, item[1]), self.reaction.items())
        return dict(rxn_dict_smiles)

    @property
    def rxnInChI(self) -> dict:
        rxn_dict_inchi = map(lambda item: (item[0].InChI, item[1]), self.reaction.items())
        return dict(rxn_dict_inchi)

    def equationSmiles(self, remove_H_ion: bool = False) -> str:
        temp = self.rxnSmiles.copy()
        if remove_H_ion == True:
            temp.pop('[H+]', None)
        return build_equation(temp)

    def equiationInChI(self, remove_H_ion: bool = False) -> str:
        temp = self.rxnInChI.copy()
        if remove_H_ion == True:
            temp.pop('InChI=1S/p+1', None)
        return build_equation(temp)

    # ------------------------------------------------------------------
    # Substrates & products
    # ------------------------------------------------------------------
    @property
    def substrates(self) -> Dict[Compound, float]:
        return dict([(c, v) for c, v in self.reaction.items() if v < 0])

    @property
    def products(self) -> Dict[Compound, float]:
        return dict([(c, v) for c, v in self.reaction.items() if v > 0])

    # ------------------------------------------------------------------
    # pKa
    # ------------------------------------------------------------------
    def pKa(self, temperature=default_T):
        pKa = []
        for compound in self.reaction:
            pKa.append(compound.pKa(temperature))
        return pKa

    # ------------------------------------------------------------------
    # Balance
    # ------------------------------------------------------------------
    @property
    def atom_diff(self) -> Dict[Compound, float]:
        mol_dict = dict([(comp.mol, coeff) for comp, coeff in self.reaction.items()])
        return atom_diff(mol_dict)

    @property
    def is_balanced(self) -> bool:
        """
        Return whether the reaction is balanced.
        """
        mol_dict = dict([(comp.mol, coeff) for comp, coeff in self.reaction.items()])
        output = is_balanced(mol_dict, ignore_H2O=self.ignore_H2O,
                             ignore_H_ion=self.ignore_H_ion,
                             ignore_charge=self.ignore_charge,
                             ignore_H=self.ignore_H)
        return output

    def balance(self, reaction: Dict[Compound, float]) -> Dict[Compound, float]:
        #
        if (self.is_balanced is None) or (self.is_balanced is True):
            return reaction
        elif self.is_balanced is False:
            original_reaction = reaction
            reaction = reaction.copy()
            diff_atom = self.atom_diff

            compounds_smiles = [comp.Smiles for comp in reaction.keys()]

            num_H2O = diff_atom.get('O')
            if (not self.ignore_H2O) and num_H2O:
                if '[H]O[H]' not in compounds_smiles:
                    reaction[Compound(to_mol('[H]O[H]', cid_type='smiles'))] = -num_H2O
                else:
                    for comp in reaction.keys():
                        if comp.Smiles == '[H]O[H]':
                            reaction[comp] = reaction[comp] - num_H2O
                            break

            if diff_atom.get('charge', 0) * diff_atom.get('H', 0) <= 0:
                num_H_ion = 0
            elif diff_atom['charge'] < 0:
                num_H_ion = -max(diff_atom['charge'], diff_atom['H'])
            elif diff_atom['charge'] > 0:
                num_H_ion = -min(diff_atom['charge'], diff_atom['H'])
            else:
                pass
            if (not self.ignore_H_ion) and num_H_ion:
                if '[H+]' not in compounds_smiles:
                    reaction[Compound(to_mol('[H+]', cid_type='smiles'))] = -num_H_ion
                else:
                    for comp in reaction.keys():
                        if comp.Smiles == '[H+]':
                            reaction[comp] = reaction[comp] - num_H_ion
                            break

            if self.is_balanced:
                return reaction
            else:
                return original_reaction
        else:
            raise ValueError(f'Unknown balanced_bool value: {self.is_balanced}')

    # ------------------------------------------------------------------
    # Thermodynamic properties
    # ------------------------------------------------------------------
    @property
    def can_be_transformed(self) -> bool:
        for x in self.reaction.keys():
            if not x.can_be_transformed:
                return False
        return True

    @property
    @lru_cache(maxsize=None)
    def standard_dGr_prime_list(self) -> Union[np.ndarray, None]:
        """
        Return the list of standard dG for the reaction.
        """
        standard_dGr_list = np.sum([comp.standard_dGf_prime_list * coeff for comp, coeff in self.reaction.items()], axis=0)
        if self.is_balanced:
            return standard_dGr_list
        else:
            return standard_dGr_list * np.nan

    @property
    @lru_cache(maxsize=None)
    def standard_dGr_prime(self) -> Tuple[np.float32, np.float32]:
        """
        Returns
        -------
            The tuple of the mean and SD of the standard dG for the reaction.
        """
        return np.mean(self.standard_dGr_prime_list).item(), np.std(self.standard_dGr_prime_list).item()

    @property
    def transformed_standard_dGr_prime(self) -> Tuple[np.float32, np.float32]:
        """
        Returns
        -------
            The tuple of the mean and SD of transformed standard dG for the reaction.
        """
        if (np.array(list(self.condition.values())) == default_condition).all():
            return self.standard_dGr_prime
        if self.can_be_transformed:
            transformed_ddGr = np.sum([comp.transformed_ddGf * coeff for comp, coeff in self.reaction.items()], axis=0)
            transformed_standard_dGr_prime = self.standard_dGr_prime[0] + transformed_ddGr
            return transformed_standard_dGr_prime, self.standard_dGr_prime[1]
        else:
            return self.standard_dGr_prime

    @property
    def transformed_dGr_prime(self) -> Tuple[np.float32, np.float32]:
        """
        Returns
        -------
            The tuple of the mean and SD of transformed dG for the reaction.
        """
        dGr_prime = sum([comp.transformed_dGf_prime[0] * coeff for comp, coeff in self.reaction.items()])
        return dGr_prime, self.transformed_standard_dGr_prime[1]

    # ------------------------------------------------------------------
    # Display methods
    # ------------------------------------------------------------------
    def _build_equation_string(self, eq_sign: str = '=') -> str:
        """
        内部方法：构建反应方程式字符串

        Parameters
        ----------
        eq_sign : str, optional
            反应符号，默认为 '='

        Returns
        -------
        str
            格式化的反应方程式字符串
        """
        return build_equation(self.rxnSmiles, eq_sign=eq_sign)

    def __repr__(self) -> str:
        """
        文本格式显示，用于列表、终端环境或后备显示
        """
        line = "─" * 50
        info = f"\n{line}\n"
        info += f"{'Chemical Reaction':^48}\n"
        info += f"{line}\n"

        # Equation
        info += f"Equation:       {self._build_equation_string('→')}\n"

        # Balance
        balance = "✅ Yes" if self.is_balanced else "❌ No"
        info += f"Is Balanced:         {balance}\n"

        # Thermodynamics
        dg_mean, dg_std = self.standard_dGr_prime
        if dg_mean is not None and not np.isnan(dg_mean):
            info += f"Δ<sub>r</sub>G'°:          {dg_mean:.1f} ± {dg_std:.1f} kJ/mol\n"
        else:
            info += f"Δ<sub>r</sub>G'°:          N/A\n"

        info += f"{line}"
        return info

    def _repr_html_(self) -> str:
        """
        HTML 格式显示，用于 Jupyter Notebook
        精简紧凑，仅显示方程式、平衡状态和吉布斯能
        """
        equation = self._build_equation_string('=')

        balance_status = "✅ Yes" if self.is_balanced else "❌ No"
        balance_color = "#28a745" if self.is_balanced else "#dc3545"

        dg_text = "N/A"
        dg_mean, dg_std = self.standard_dGr_prime
        if dg_mean is not None and not np.isnan(dg_mean):
            dg_text = f"{dg_mean:.1f} ± {dg_std:.1f} kJ/mol"

        html = f"""
        <div style="
            font-family: 'Segoe UI', Arial, sans-serif;
            background-color: #ffffff; color: #333;
            border: 1px solid #e1e4e8; border-left: 4px solid #4a90e2;
            border-radius: 6px; padding: 12px;
            margin: 6px 0; box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            display: inline-block; min-width: 300px; font-size: 13px;
        ">
            <div style="margin-bottom: 8px;">
                <span style="font-weight: bold; color: #555;">Equation:</span><br>
                <code style="
                    background: #f8f9fa; padding: 4px 8px; border-radius: 3px;
                    font-size: 14px; color: #d63384; display: inline-block; margin-top: 3px;
                ">{equation}</code>
            </div>

            <div style="margin-bottom: 8px;">
                <span style="font-weight: bold; color: #555;">Is Balanced:</span>
                <span style="color: {balance_color}; font-weight: 600;">{balance_status}</span>
            </div>

            <div>
                <span style="font-weight: bold; color: #555;">Δ<sub>r</sub>G'°:</span>
                <span style="color: #007bff; font-weight: 600;">{dg_text}</span>
            </div>
        </div>
        """
        return html

    def __str__(self) -> str:
        """
        print() 时显示简洁版本，使用默认的 '=' 符号
        """
        return self._build_equation_string()
