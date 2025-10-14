import math
import numpy as np

from .. import R, FARADAY, default_T, default_pH, default_I, default_pMg, default_e_potential

standard_formation_dgf_H = 0
standard_formation_dg_Mg = -455.3
standard_formation_dh_Mg = -467.0


def H_term(num_H:float|int, pH:float, T:float)-> float:
    RT = R * T
    return num_H * (RT * np.log(10) * pH - standard_formation_dgf_H)


def Mg_term(num_Mg:float|int, pMg:float, T:float)-> float:
    RT = R * T
    _dg_mg = (T / default_T) * standard_formation_dg_Mg + (1.0 - T / default_T) * standard_formation_dh_Mg
    return num_Mg * (RT * np.log(10) * pMg - _dg_mg)


def debye_hueckel(ionic_strength:float, charge:float, num_H:int, T:float) -> float:
    """Compute the ionic-strength-dependent transformation coefficient.

    For the Legendre transform to convert between chemical and biochemical
    Gibbs energies, we use the extended Debye-Hueckel theory to calculate the
    dependence on ionic strength and temperature.

    Parameters
    ----------
    ionic_strength : float
        The ionic-strength in M
    temperature : float
        The temperature in Kelvin


    Returns
    -------
    Quantity
        The DH factor associated with the ionic strength at this
        temperature in kJ/mol

    """
    _a1 = 9.20483e-3  # kJ / mol / M^0.5 / K
    _a2 = 1.284668e-5  # kJ / mol / M^0.5 / K^2
    _a3 = 4.95199e-8  # kJ / mol / M^0.5 / K^3
    B = 1.6  # 1 / M^0.5
    alpha = _a1 * T - _a2 * T**2 + _a3 * T**3  # kJ / mol
    sqrt_ionic_strength = math.sqrt(ionic_strength)
    return -alpha * (charge**2 - num_H) * (sqrt_ionic_strength / (1.0 + B * sqrt_ionic_strength)) # kJ / mol


def term(pH:float, pMg:float, I:float, T:float, charge:float, num_H:float, num_Mg:float) -> float:
    # 
    _H_term = H_term(num_H=num_H, pH=pH, T=T)
    _Mg_term = Mg_term(num_Mg=num_Mg, pMg=pMg, T=T)
    _ionic_strength_term = debye_hueckel(ionic_strength=I, charge=charge, num_H=num_H, T=T)

    return  _H_term + _Mg_term + _ionic_strength_term


def pseudoisomers_ddGf(pKa:dict, pH:float, T:float):
    # 
    acidicV = dict([(x['atomIndex'],x['value']) for x in pKa['acidicValuesByAtom'] if not np.isnan(x['value'])])
    basicV = dict([(x['atomIndex'],x['value']) for x in pKa['basicValuesByAtom'] if not np.isnan(x['value'])])
    t = list(set(acidicV.keys())|set(basicV.keys()))
    t.sort()

    ddGf=np.array([0])
    d_num_H_ion=np.array([0])

    for atom_i in t:
        i_pka = acidicV.get(atom_i)
        i_pkb = basicV.get(atom_i)
        if i_pka is not None:
            ddGf_ = ddGf - R * T * np.log(10) * (pH - i_pka)
            ddGf = np.concatenate([ddGf, ddGf_])
            d_num_H_ion = np.concatenate([d_num_H_ion, d_num_H_ion-1])
        elif i_pkb is not None:
            ddGf_ = ddGf - R * T * np.log(10) * (i_pkb - pH)#((14 - pH) - i_pkb)
            ddGf = np.concatenate([ddGf, ddGf_])
            d_num_H_ion = np.concatenate([d_num_H_ion, d_num_H_ion+1])
        else:
            raise Exception("Wrong in 'iter_pseudoisomers()'")
    return ddGf, d_num_H_ion


def pseudoisomers_ratio(pKa:dict, pH:float, T:float, plot:bool=False):
    # 
    RT = R * T
    ddGf, d_num_H_ion = pseudoisomers_ddGf(pKa, pH, T)
    ratio = np.exp(-ddGf/RT)/np.sum(np.exp(-ddGf/RT))
    return ratio, d_num_H_ion


def transformed_pseudoisomers_ddGf(pKa:dict, pH:float, T:float, pMg:float, I:float, charge:float, num_H:float, num_Mg:float):
    _standard_pseudoisomers_ddGf, d_num_H_ion = pseudoisomers_ddGf(pKa=pKa, pH=default_pH, T=default_T)

    d_term = []
    for d_H_ion in d_num_H_ion:
        _charge = charge + d_H_ion
        _num_H = num_H + d_H_ion
        default_term = term(pH=default_pH, pMg=default_pMg, I=default_I, T=default_T, charge=_charge, num_H=_num_H, num_Mg=num_Mg)
        _term = term(pH=pH, pMg=pMg, I=I, T=T, charge=_charge, num_H=_num_H, num_Mg=num_Mg)
        d_term.append(_term-default_term)
    d_term = np.array(d_term)
    
    return _standard_pseudoisomers_ddGf + d_term, d_num_H_ion


def transformed_pseudoisomers_ratio(pKa:dict, pH:float, T:float, pMg:float, I:float, charge:float, num_H:float, num_Mg:float):
    # 
    RT = R * T
    ddGf, d_num_H_ion = transformed_pseudoisomers_ddGf(pKa=pKa, pH=pH, T=T, pMg=pMg, I=I, charge=charge, num_H=num_H, num_Mg=num_Mg)
    ratio = np.exp(-ddGf/RT)/np.sum(np.exp(-ddGf/RT))
    return ratio, d_num_H_ion


def electric_term(charge, e_potential):
    # 
    return FARADAY * charge * e_potential


def transformed_ddGf(pKa:dict, pH:float, T:float, pMg:float, I:float, e_potential:float, charge:float, num_H:float, num_Mg:float):
    # 
    RT = R * T
    _transformed_pseudoisomers_ddGf, _ = transformed_pseudoisomers_ddGf(pKa=pKa, pH=pH, T=T, pMg=pMg, I=I, charge=charge, num_H=num_H, num_Mg=num_Mg)
    _transformed_ddGf = -RT * np.log(np.sum(np.exp(-_transformed_pseudoisomers_ddGf/RT)))
    
    _pseudoisomers_ddGf, _ = pseudoisomers_ddGf(pKa=pKa, pH=default_pH, T=default_T)
    _default_ddGf = -RT * np.log(np.sum(np.exp(-_pseudoisomers_ddGf/RT)))

    _d_e_potential = e_potential - default_e_potential
    _electric_term = electric_term(charge=charge, e_potential=_d_e_potential)
    
    return _transformed_ddGf - _default_ddGf + _electric_term