import torch
import numpy as np

from graphrepr.config import csv_section, optimizer_section
from graphrepr.data import load_data_from_df

from graphrepr.chemprop.data_utils import filter_invalid_smiles
from graphrepr.chemprop.data_data import MoleculeDatapoint, construct_molecule_batch, MoleculeDataLoader


def load_data_chemprop(paths, data_config, model_config, shuffle=False, num_workers=1):
    smiles, labels = load_data_from_df(paths, **data_config[csv_section])
    
    data = construct_molecule_batch([
            MoleculeDatapoint(smiles=[smis], targets=[targets]) for smis, targets in zip(smiles, labels)
        ])

    data = remove_invalid_mols(data)
    
    loader = MoleculeDataLoader(
        dataset=data,
        batch_size=model_config[optimizer_section]['batch_size'],
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return smiles, labels, loader


def remove_invalid_mols(data):
    original_data_len = len(data)
    data = filter_invalid_smiles(data)

    if len(data) < original_data_len:
        print(f'Warning: {original_data_len - len(data)} SMILES are invalid.')
        
    return data


def run_epoch(model, loss_function, optimizer, data_loader, device):
    """
    Run one epoch. Code is based on graphconv.run_epoch and
    https://github.com/chemprop/chemprop/blob/ac6bc9f52fbde17789fd844d7f1244e558150f01/chemprop/train/train.py#L17
    """
    
    model.train()  # push model to the training mode (changes dropout and batchnorm layers' behaviour)
    cumulative_loss = 0
    
    for batch in data_loader:
        # batch is a MoleculeDataset
        mol_batch, target_batch = batch.batch_graph(), batch.targets()
        targets = torch.tensor([[0 if x is None else x for x in tb] for tb in target_batch]) # shape(batch, tasks)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        output = model(mol_batch)        
    
        loss = loss_function(output, targets)
        loss.backward()

        cumulative_loss += len(batch) * loss.item()
        optimizer.step()

    return cumulative_loss


def predict(model, data_loader, device):
    model.eval()  # set dropout and batch normalization layers to evaluation mode before running inference

    pred_array = []
    true_array = []

    for batch in data_loader:
        # batch is a MoleculeDataset
        mol_batch, target_batch = batch.batch_graph(), batch.targets()
        targets = torch.tensor([[0 if x is None else x for x in tb] for tb in target_batch]) # shape(batch, tasks)
        output = model(mol_batch)

        pred_array.append(output.detach().cpu().numpy())
        true_array.append(targets.cpu().numpy())

    pred_array = np.concatenate(pred_array)
    true_array = np.concatenate(true_array)

    return true_array, pred_array
