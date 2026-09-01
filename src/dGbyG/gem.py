"""
GEM (Genome-scale Metabolic Model) batch processing utilities.
"""
import cobra
import numpy as np
import pandas as pd
from typing import Dict
from tqdm import tqdm

from .compound import Compound
from .reaction import Reaction
from .utils._to_mol_methods import to_mol_methods
from .utils._get_pKa_methods import batch_predict_and_save_pka
from .constants import default_condition


def predict_transformed_dG_prime_for_GEM(
    gem: cobra.Model,
    compartment_conditions: Dict[str, Dict[str, float | int]] | None = None,
    use_met_id_types='all',
    ignore_met_id_types=[]
):
    """
    Predict transformed standard Gibbs free energies for metabolites and reactions in a GEM.

    For each metabolite, this function extracts compound identifiers from annotations,
    creates Compound objects, retrieves or predicts pKa values using ChemAxon, and then
    computes transformed standard Gibbs free energy of formation. For reactions, it
    combines the transformed dGf_prime values of participating metabolites to calculate
    the transformed standard Gibbs free energy of reaction.

    The function also persists pKa predictions into a local SQLite database via
    `batch_predict_and_save_pka`.

    Parameters
    ----------
    gem : cobra.Model
        A COBRApy genome-scale metabolic model. Metabolites must have annotations
        (e.g., 'inchi', 'pubchem.metabolite', 'kegg.compound', etc.) that can be used
        to look up compound structures.
    compartment_conditions : dict of {str: dict of {str: float|int}}, optional
        Mapping from compartment identifier (abbreviation or full name) to a dictionary
        of physicochemical conditions (e.g., `{'pH': 7.5, 'pMg': 3.0, 'I': 0.2}`).
        If a compartment is not found in this dictionary, a fallback is attempted:
        first the literal key `'c'`, then the full name of the default cytosolic
        compartment (`gem.compartments['c']`), and finally the module‑level
        `default_condition`. If `compartment_conditions` is None, all compartments
        default to `default_condition`.
    use_met_id_types : str or list of str, default='all'
        Which annotation types to use when constructing Compound objects.
        If `'all'`, use all types present in `Compound.recognizable_cids` that are not
        excluded by `ignore_met_id_types`. If a string, only that type is used.
        If a list, the types in the list are used.
    ignore_met_id_types : list of str, default=[]
        Annotation types to ignore when `use_met_id_types='all'`. Matching is performed
        on the beginning of the type string (case-insensitive). E.g., `'kegg'` would
        ignore both `'kegg.compound'` and `'kegg.drug'`.

    Returns
    -------
    Met_df : pandas.DataFrame
        Transformed standard Gibbs free energies of formation per metabolite.
        Index: metabolite IDs from `gem.metabolites`.
        Columns: annotation types (e.g., 'hmdb', 'bigg.metabolite') that were used.
        Values are the computed `comp.transformed_standard_dGf_prime` for each compound.
    Rxn_df : pandas.DataFrame
        Transformed standard Gibbs free energies of reaction per reaction.
        Index: reaction IDs from `gem.reactions`.
        Columns: `'dGr_prime'` and `'SD of dGr_prime'` (standard deviation).

    Raises
    ------
    ValueError
        If `compartment_conditions` is provided but is not a dictionary
        If the global variable `default_condition` is not defined in the module.
    AttributeError
        If required attributes (e.g., `Compound.recognizable_cids`, `Compound.Smiles`,
        `Compound.transformed_standard_dGf_prime`) are missing.
    """
    # ------------------------------------------------------------------
    # 1. Build compartment‑specific condition dictionaries
    # ------------------------------------------------------------------
    conditions = {}
    if isinstance(compartment_conditions, dict):
        # Match each model compartment using abbreviation, full name, or default
        for abbr, full_name in gem.compartments.items():
            if abbr in compartment_conditions:
                conditions[abbr] = compartment_conditions[abbr]
            elif full_name in compartment_conditions:
                conditions[abbr] = compartment_conditions[full_name]
            elif 'c' in compartment_conditions:
                conditions[abbr] = compartment_conditions['c']
            elif gem.compartments['c'] in compartment_conditions:
                conditions[abbr] = compartment_conditions[gem.compartments['c']]
            else:
                conditions[abbr] = default_condition   # global variable
    elif compartment_conditions is not None:
        raise ValueError("compartment_conditions should be a dict with keys as compartment abbreviations or full names.")

    # ------------------------------------------------------------------
    # 2. Normalize ignore_met_id_types (always a list)
    # ------------------------------------------------------------------
    if isinstance(ignore_met_id_types, str):
        ignore_met_id_types = [ignore_met_id_types]

    # ------------------------------------------------------------------
    # 3. Determine which annotation types to use (use_met_id_types)
    # ------------------------------------------------------------------
    if use_met_id_types == 'all':
        use_types = []
        for x in Compound.recognizable_cids:   # expects global Compound class
            keep_x = not any([x.lower().startswith(y) for y in ignore_met_id_types])
            if keep_x:
                use_types.append(x)
    elif isinstance(use_met_id_types, str):
        use_types = [use_met_id_types]
    else:
        use_types = use_met_id_types

    # ------------------------------------------------------------------
    # 4. For each metabolite, create Compound objects for selected annotation types
    # ------------------------------------------------------------------
    for met in tqdm(gem.metabolites, desc="Processing metabolites"):
        met.compound = {}
        for cid_type, cid in met.annotation.items():
            if cid_type in use_types:
                # If cid is a list, take the first element (assume it's the primary ID)
                if isinstance(cid, str):
                    comp = Compound(cid, cid_type)
                else:
                    for sub_cid in cid:
                        comp = Compound(sub_cid, cid_type)
                        if comp.mol:
                            break
                comp.condition = conditions.get(met.compartment, default_condition)
                met.compound[cid_type] = comp

    # ------------------------------------------------------------------
    # 5. Collect all unique SMILES from metabolites and pre‑compute/persist pKa
    # ------------------------------------------------------------------
    Smiles = []
    for met in gem.metabolites:
        Smiles.extend([comp.Smiles for comp in met.compound.values() if comp.Smiles])
    batch_predict_and_save_pka(Smiles)   # saves pKa to DB for future lookups

    # ------------------------------------------------------------------
    # 6. Build DataFrame of transformed dGf_prime per metabolite
    # ------------------------------------------------------------------
    Data = {}
    for met in tqdm(gem.metabolites, desc="Predicting transformed standard Gibbs free energy for metabolites"):
        Data[met.id] = {}
        for cid_type, comp in met.compound.items():
            comp.pKa_source = 'chemaxon_pKa_db'   # record source for debugging
            Data[met.id][cid_type] = comp.transformed_standard_dGf_prime
    Met_df = pd.DataFrame(Data).T

    # ------------------------------------------------------------------
    # 7. Build DataFrame of transformed dGr_prime per reaction
    # ------------------------------------------------------------------
    Data = {}
    for rxn in tqdm(gem.reactions, desc="Predicting transformed standard Gibbs free energy for reactions"):
        rxn_dict = {}
        for met, coeff in rxn.metabolites.items():
            # Find the first Compound object associated with this metabolite that has a valid mol
            comp = Compound(None, None)        # fallback empty compound
            for comp in met.compound.values():
                if comp.mol is not None:
                    break
            rxn_dict[comp] = coeff
        reaction = Reaction(rxn_dict, cids_type='compound')
        Data[rxn.id] = reaction.transformed_standard_dGr_prime   # expected to be (dGr, sd)
    Rxn_df = pd.DataFrame(Data, index=['dGr_prime', 'SD of dGr_prime']).T

    return Met_df, Rxn_df
