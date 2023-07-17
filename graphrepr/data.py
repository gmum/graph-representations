# code from gmum/MLinPL2019_cheminfo_workshops and gmum/geo-gcn
import os
import logging
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles
from typing import Optional, List, Union


def load_dataset(filepaths, smiles_index, y_index, skip_line: bool = False, delimiter: str = ',',
                 scale: Optional[str] = None, average: Optional[str] = None,
                 neighbours: bool = False, total_num_hs: bool = False,
                 formal_charge: bool = False, is_in_ring: bool = False, 
                 is_aromatic: bool = False, get_positions: bool = False,
                 dmpnn_representation: bool = False, cheminet_representation: bool = False, 
                 deepchemstable_representation: bool = False, duvenaud_representation: bool = False):

    """
    Load dataset from csv, calculate representation of each entry, return a dataset
    :param filepaths: list: paths to csv files with data
    :param smiles_index: int: index of the column with smiles in the csv
    :param y_index: int: index of the column with the label in the csv
    :param skip_line: boolean: True if the first line of the csv file contains column names, False otherwise
    :param delimiter: delimeter used in csv
    :param scale: should y be scaled? (useful with skewed distributions of y)
    :param average: if the same SMILES appears multiple times how should its label be averaged?
    :param neighbours: bool: use number of neighbours as a feature?
    :param total_num_hs: bool: use total number of Hs as a feature?
    :param formal_charge: bool: use formal charage as a feature?
    :param is_in_ring: bool: use ringness as a feature?
    :param is_aromatic: bool: use aromaticity as a feature?
    :return: (torch_geometric.data.Data, list ) - data_set, smiles
    """

    # strict type checking
    for param in [skip_line, neighbours, total_num_hs, formal_charge, is_in_ring,
                  is_aromatic, get_positions, dmpnn_representation, cheminet_representation,
                  deepchemstable_representation, duvenaud_representation]:
        assert isinstance(param, bool), f"Param should be bool, is {type(param)} with value {param}."
    
    
    smiles, labels = load_data_from_df(filepaths, smiles_index, y_index, skip_line, delimiter, scale, average)

    data_set, smiles = load_data_from_smiles(smiles, labels, neighbours=neighbours, total_num_hs=total_num_hs,
                                         formal_charge=formal_charge, is_in_ring=is_in_ring, is_aromatic=is_aromatic,
                                         get_positions=get_positions, dmpnn_representation=dmpnn_representation,
                                         cheminet_representation=cheminet_representation,
                                         deepchemstable_representation=deepchemstable_representation,
                                         duvenaud_representation=duvenaud_representation)

    return data_set, smiles


def load_data_from_df(dataset_paths, smiles_index: int, y_index: int,
                      skip_line: bool = False, delimiter: str = ',',
                      scale: Optional[str] = None, average: Optional[str] = None):

    """
    Load multiple files from csvs, concatenate and return smiles and ys
    :param dataset_paths: list: paths to csv files with data
    :param smiles_index: int: index of the column with smiles
    :param y_index: int: index of the column with the label
    :param skip_line: boolean: True if the first line of the file contains column names, False otherwise
    :param delimiter: delimeter used in csv
    :param scale: should y be scaled? (useful with skewed distributions of y)
    :param average: if the same SMILES appears multiple times how should its values be averaged?
    :return: (smiles, labels) - np.arrays
    """

    assert isinstance(skip_line, bool), f"skip_line should be bool, is {type(param)} with value {param}."

    # column names present in files?
    header = 0 if skip_line else None

    # reading all the files
    dfs = []
    for data_path in dataset_paths:
        dfs.append(pd.read_csv(data_path, delimiter=delimiter, header=header))

    # merging
    data_df = pd.concat(dfs)

    # scaling
    if scale is not None:
        if 'sqrt' == scale.lower().strip():
            data_df.iloc[:, y_index] = np.sqrt(data_df.iloc[:, y_index])
        elif 'log' == scale.lower().strip():
            data_df.iloc[:, y_index] = np.log(1 + data_df.iloc[:, y_index])
        else:
            raise NotImplementedError(f"Scale {scale} is not implemented.")

    # averaging if one smiles has multiple values
    if average is not None:
        smiles_col = data_df.iloc[:, smiles_index].name
        y_col = data_df.iloc[:, y_index].name

        data_df = data_df.loc[:, [smiles_col, y_col]]  # since now: smiles is 0, y_col is 1, dropping other columns
        smiles_index = 0
        y_index = 1
        if 'median' == average.lower().strip():
            data_df[y_col] = data_df[y_col].groupby(data_df[smiles_col]).transform('median')
        else:
            raise NotImplementedError(f"Averaging {average} is not implemented.")

    # breaking into x and y
    data_df = data_df.values
    data_x = data_df[:, smiles_index]
    data_y = data_df[:, y_index]

    if data_y.dtype == np.float64:
        data_y = data_y.astype(np.float32)

    return data_x, data_y


def load_data_from_smiles(x_smiles, labels, neighbours: bool = False,
                          total_num_hs: bool = False, formal_charge: bool = False,
                          is_in_ring: bool = False, is_aromatic: bool = False, 
                          get_positions: bool = False,
                          dmpnn_representation: bool = False, 
                          cheminet_representation: bool = False,
                          deepchemstable_representation: bool = False,
                          duvenaud_representation=False):
    # strict type checking
    for param in [neighbours, total_num_hs, formal_charge, is_in_ring, is_aromatic,
                  get_positions, dmpnn_representation, cheminet_representation,
                  deepchemstable_representation, duvenaud_representation]:
        assert isinstance(param, bool), f"Param should be bool, is {type(param)} with value {param}."
    
    x_all, y_all, smiles_all = [], [], []
    for smiles, label in zip(x_smiles, labels):
        try:
            if len(smiles) < 2:
                raise ValueError
                
            mol = MolFromSmiles(smiles)
            
            if get_positions:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol)
                AllChem.UFFOptimizeMolecule(mol)
                mol = Chem.RemoveHs(mol)
                
            afm, adj, mat_positions = featurize_mol(mol, neighbours=neighbours,
                                                    total_num_hs=total_num_hs,
                                                    formal_charge=formal_charge,
                                                    is_in_ring=is_in_ring,
                                                    is_aromatic=is_aromatic,
                                                    get_positions=get_positions,
                                                    dmpnn_representation=dmpnn_representation,
                                                    cheminet_representation=cheminet_representation,
                                                    deepchemstable_representation=deepchemstable_representation,
                                                    duvenaud_representation=duvenaud_representation)

            x_all.append([afm, adj, mat_positions])
            y_all.append([label])
            smiles_all.append([smiles])
        except ValueError as e:
            logging.warning('SMILES ({}) can not be converted to a graph.\nREASON: {}'.format(smiles, e))

    data_set = transform_dataset_pg([[*i, j, smi] for i, j, smi in zip(x_all, y_all, smiles_all)])
    return data_set, smiles_all


def featurize_mol(mol, neighbours: bool = False, total_num_hs: bool = False,
                  formal_charge: bool = False, is_in_ring: bool = False, is_aromatic=False,
                  get_positions: bool = False, dmpnn_representation: bool = False,
                  cheminet_representation: bool = False, deepchemstable_representation: bool = False,
                  duvenaud_representation: bool = False):
    
    # strict type checking
    for param in [neighbours, total_num_hs, formal_charge, is_in_ring, is_aromatic,
                  get_positions, dmpnn_representation, cheminet_representation,
                  deepchemstable_representation, duvenaud_representation]:
        assert isinstance(param, bool), f"Param should be bool, is {type(param)} with value {param}."
    
    if dmpnn_representation is True:
        node_features = np.array([get_atom_features_dmpnn(atom) for atom in mol.GetAtoms()])
    elif cheminet_representation is True:
        node_features = np.array([get_atom_features_chemi_net(atom) for atom in mol.GetAtoms()])
    elif deepchemstable_representation is True:
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        node_features = np.array([get_atom_features_deep_chem_stable(atom) for atom in mol.GetAtoms()])
    elif duvenaud_representation is True:
        node_features = np.array([get_atom_features_duvenaud(atom) for atom in mol.GetAtoms()])
    else:
        node_features = np.array(
            [get_atom_features(atom, neighbours=neighbours,
                               total_num_hs=total_num_hs,
                               formal_charge=formal_charge,
                               is_in_ring=is_in_ring, is_aromatic=is_aromatic)
             for atom in mol.GetAtoms()
             ])

    adj_matrix = np.eye(mol.GetNumAtoms())

    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom().GetIdx()
        end_atom = bond.GetEndAtom().GetIdx()
        adj_matrix[begin_atom, end_atom] = adj_matrix[end_atom, begin_atom] = 1

    if get_positions:
        conf = mol.GetConformer()
        pos_matrix = np.array([[conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z]
                               for k in range(mol.GetNumAtoms())])
    else:
        pos_matrix = None

    return node_features, adj_matrix, pos_matrix


def get_atom_features(atom, neighbours: bool = False,
                      total_num_hs: bool = False, formal_charge: bool = False,
                      is_in_ring=False, is_aromatic: bool = False):

    """
    Calculate feature vector for atom.
    :param atom: atom to featurise
    :param neighbours: bool: use number of neighbours as a feature?
    :param total_num_hs: bool: use total number of Hs as a feature?
    :param formal_charge: bool: use formal charage as a feature?
    :param is_in_ring: bool: use ringness as a feature?
    :param is_aromatic: bool: use aromaticity as a feature?
    :return: np.array of attributes - a vector representation of atom
    """

    # strict type checking
    for param in [neighbours, total_num_hs, formal_charge, is_in_ring, is_aromatic]:
        assert isinstance(param, bool), f"Param should be bool, is {type(param)} with value {param}."

    attributes = []
    attributes += one_hot_vector(atom.GetAtomicNum(), [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999])
    if neighbours:
        attributes += one_hot_vector(len(atom.GetNeighbors()), [0, 1, 2, 3, 4, 5])
    if total_num_hs:
        attributes += one_hot_vector(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    if formal_charge:
        attributes.append(atom.GetFormalCharge())
    if is_in_ring:
        attributes.append(atom.IsInRing())
    if is_aromatic:
        attributes.append(atom.GetIsAromatic())

    return np.array(attributes, dtype=np.float32)


def get_atom_features_dmpnn(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:

    """
    Builds a feature vector for an atom. (In the paper: Yang)
    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """

    # Atom feature sizes
    MAX_ATOMIC_NUM = 100
    ATOM_FEATURES = {
        'atomic_num': list(range(MAX_ATOMIC_NUM)),
        'degree': [0, 1, 2, 3, 4, 5],
        'formal_charge': [-1, -2, 1, 2, 0],
        'chiral_tag': [0, 1, 2, 3],
        'num_Hs': [0, 1, 2, 3, 4],
        'hybridization': [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2
        ],
    }
    
    def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:

        """
        Creates a one-hot encoding with an extra category for uncommon values.
        :param value: The value for which the encoding should be one.
        :param choices: A list of possible values.
        :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
                 If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
        """

        encoding = [0] * (len(choices) + 1)
        index = choices.index(value) if value in choices else -1
        encoding[index] = 1
        return encoding
    
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
           onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
           onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
           onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
           onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
           onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    if functional_groups is not None:
        features += functional_groups
    return np.array(features, dtype=np.float32)


def get_atom_features_chemi_net(atom):

    """
    Calculate feature vector for atom. (In the paper: Liu)
    :param atom: atom to featurise
    :return: np.array of attributes - a vector representation of atom
    """

    periodic_table = Chem.GetPeriodicTable()

    attributes = []
    attributes += one_hot_vector(atom.GetAtomicNum(), list(range(1,21)) + [35, 53, 999])
    
    attributes.append(periodic_table.GetRvdw(atom.GetSymbol()))
    attributes.append(periodic_table.GetRcovalent(atom.GetSymbol()))
    
    attributes.append(atom.IsInRingSize(3))
    attributes.append(atom.IsInRingSize(4))
    attributes.append(atom.IsInRingSize(5))
    attributes.append(atom.IsInRingSize(6))
    attributes.append(atom.IsInRingSize(7))
    attributes.append(atom.IsInRingSize(8))
    
    attributes.append(atom.GetIsAromatic())
    attributes.append(atom.GetFormalCharge())

    return np.array(attributes, dtype=np.float32)


def get_atom_features_deep_chem_stable(atom):

    """
    Calculate feature vector for atom. (In the paper: Li)
    :param atom: atom to featurise
    :return: np.array of attributes - a vector representation of atom
    """

    attributes = []
    attributes += one_hot_vector(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'B', 'H',  'Unknown'])
    attributes += one_hot_vector(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
    attributes += one_hot_vector(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    attributes += one_hot_vector(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
    
    attributes.append(atom.GetIsAromatic()) 
    attributes.append(atom.GetFormalCharge())
    attributes.append(atom.GetNumRadicalElectrons())
        
    attributes += one_hot_vector(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType. SP3D, Chem.rdchem.HybridizationType.SP3D2])
    
    add_Gasteiger = float(atom.GetProp('_GasteigerCharge'))
    if np.isnan(add_Gasteiger) or np.isinf(add_Gasteiger):
        add_Gasteiger = 0.0
    attributes.append(add_Gasteiger)

    return np.array(attributes, dtype=np.float32)


def get_atom_features_duvenaud(atom):

    """
    Calculate feature vector for atom.
    :param atom: atom to featurise
    :return: np.array of attributes - a vector representation of atom
    """

    attributes = []
        
    attributes += one_hot_vector(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                                       'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
                                       'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
                                       'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                                       'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'])
    
    attributes += one_hot_vector(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
    attributes += one_hot_vector(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    attributes += one_hot_vector(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
    
    attributes.append(atom.GetIsAromatic())

    return np.array(attributes, dtype=np.float32)


def one_hot_vector(val, lst):

    """Converts a value to a one-hot vector based on options in lst"""

    if val not in lst:
        val = lst[-1]
    return map(lambda x: x == val, lst)


def transform_dataset_pg(dataset):
    dataset_pg = [transform_molecule_pg(mol) for mol in dataset]
    return dataset_pg


def transform_molecule_pg(mol):
    afm, adj, positions, label, smi = mol

    x = torch.tensor(afm)
    y = torch.tensor(label)
    edge_index = torch.tensor(get_edge_indices(adj)).t().contiguous()
    pos = torch.tensor(positions) if positions else None

    return Data(x=x, y=y, edge_index=edge_index, pos=pos, smiles=smi)


def get_edge_indices(adj):
    edges_list = []
    for i in range(adj.shape[0]):
        for j in range(i, adj.shape[0]):
            if adj[i, j] == 1:
                edges_list.append((i, j))
    return edges_list
