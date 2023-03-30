from rdkit import Chem

def make_mol(s: str):
    """
    Builds an RDKit molecule from a SMILES string.
    
    :param s: SMILES string.
    :return: RDKit molecule.
    """
    mol = Chem.MolFromSmiles(s)
    return mol
