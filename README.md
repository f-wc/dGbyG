# dGbyG
The python API of dGbyG.

## Description
dGbyG is a GNN-based tool for predicting standard Gibbs energy changes of metabolic reactions, which can accept a variety of molecular identification
types, including: 

- SMILES
- InChI
- KEGG compound entry
- MetaNetX chemical ID
- ChEBI ID
- HMDB ID
- BiGG universal metabolite ID
- PubChem compound CID
- MOL file
- RXN file
- InChI Key
- name

Read [the dGbyG paper](https://doi.org/10.1016/j.cels.2025.101393).


## Guidelines
### Prerequisites
- Git
- Conda (Anaconda or Miniconda)

### Step 1: Clone the Repository
Clone the repository to your local machine using the following command:

```bash
git clone https://github.com/f-wc/dGbyG.git
```

### Step 2: Create a New Conda Environment and Install Dependencies (Highly Recommended)
#### Install the required dependencies:
```bash
cd /path/to/dGbyG
conda env create -f environment.yaml -n dGbyG
```

#### Install `chemaxon.marvin.calculations.pKaPlugin`(optional)
The `chemaxon.marvin.calculations.pKaPlugin` is used for pKa calculation. It is not necessary for the basic usage of dGbyG, but it is recommended for users who want to calculate transformed standard Gibbs energy change between different pH. Note that this plugin is not free, and you can find more information from the ChemAxon website (https://docs.chemaxon.com/display/docs/calculators_index.md).


### Step 3: Install dGbyG
#### Install dGbyG
```bash
cd /path/to/dGbyG
pip install -e .
```


## Citation
If you use or extend our work, please cite the paper as follows:  
- Fan et al., Unraveling principles of thermodynamics for genome-scale metabolic networks using graph neural net
works, Cell Systems (2025), https://doi.org/10.1016/j.cels.2025.101393