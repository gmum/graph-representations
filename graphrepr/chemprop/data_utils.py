from collections import OrderedDict
from logging import Logger
from typing import List

from tqdm import tqdm

from .data_data import MoleculeDatapoint, MoleculeDataset, make_mols


def filter_invalid_smiles(data: MoleculeDataset) -> MoleculeDataset:
    """
    Filters out invalid SMILES.

    :param data: A :class:`~chemprop.data.MoleculeDataset`.
    :return: A :class:`~chemprop.data.MoleculeDataset` with only the valid molecules.
    """
    return MoleculeDataset([datapoint for datapoint in tqdm(data)
                            if all(s != '' for s in datapoint.smiles) and all(m is not None for m in datapoint.mol)
                            and all(m.GetNumHeavyAtoms() > 0 for m in datapoint.mol if not isinstance(m, tuple))
                            and all(m[0].GetNumHeavyAtoms() + m[1].GetNumHeavyAtoms() > 0 for m in datapoint.mol if isinstance(m, tuple))])


def get_invalid_smiles_from_list(smiles: List[List[str]]) -> List[List[str]]:
    """
    Returns the invalid SMILES from a list of lists of SMILES strings.

    :param smiles: A list of list of SMILES.
    :return: A list of lists of SMILES, for the invalid SMILES among the lists provided.
    """
    invalid_smiles = []

    for mol_smiles in smiles:
        mols = make_mols(smiles=mol_smiles)
        if any(s == '' for s in mol_smiles) or \
           any(m is None for m in mols) or \
           any(m.GetNumHeavyAtoms() == 0 for m in mols if not isinstance(m, tuple)) or \
           any(m[0].GetNumHeavyAtoms() + m[1].GetNumHeavyAtoms() == 0 for m in mols if isinstance(m, tuple)):

            invalid_smiles.append(mol_smiles)

    return invalid_smiles


def get_data_from_smiles(smiles: List[List[str]],
                         skip_invalid_smiles: bool = True,
                         logger: Logger = None,
                         features_generator: List[str] = None) -> MoleculeDataset:
    """
    Converts a list of SMILES to a :class:`~chemprop.data.MoleculeDataset`.

    :param smiles: A list of lists of SMILES with length depending on the number of molecules.
    :param skip_invalid_smiles: Whether to skip and filter out invalid smiles using :func:`filter_invalid_smiles`
    :param logger: A logger for recording output.
    :param features_generator: List of features generators.
    :return: A :class:`~chemprop.data.MoleculeDataset` with all of the provided SMILES.
    """
    debug = logger.debug if logger is not None else print

    data = MoleculeDataset([
        MoleculeDatapoint(
            smiles=smile,
            row=OrderedDict({'smiles': smile}),
            features_generator=features_generator
        ) for smile in smiles
    ])

    # Filter out invalid SMILES
    if skip_invalid_smiles:
        original_data_len = len(data)
        data = filter_invalid_smiles(data)

        if len(data) < original_data_len:
            debug(f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    return data


def validate_dataset_type(data: MoleculeDataset, dataset_type: str) -> None:
    """
    Validates the dataset type to ensure the data matches the provided type.

    :param data: A :class:`~chemprop.data.MoleculeDataset`.
    :param dataset_type: The dataset type to check.
    """
    target_set = {target for targets in data.targets() for target in targets} - {None}
    classification_target_set = {0, 1}

    if dataset_type == 'classification' and not (target_set <= classification_target_set):
        raise ValueError('Classification data targets must only be 0 or 1 (or None). '
                         'Please switch to regression.')
    elif dataset_type == 'regression' and target_set <= classification_target_set:
        raise ValueError('Regression data targets must be more than just 0 or 1 (or None). '
                         'Please switch to classification.')
