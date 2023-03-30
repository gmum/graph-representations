import os
import os.path as osp

import json

import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
from torch_geometric.data import Data, DataLoader

from ..graphconv import GraphConvNetwork, predict
from ..config import utils_section, params_section
from ..data import load_data_from_smiles


# utils for:
# - storing and loading datapoints (ie. sample with an explanation) with JSON
# - loading and validating (ie. checking that it loaded correctly) torch models


def store_datapoint(save_dir, fname, d, node_mask=None, edge_mask=None, extras=None):
    # Saves Data d and its explanation as JSON
    pkeys = ['smiles', 'x', 'y', 'edge_index', 'pos', 'node_mask', 'edge_mask', 'extras']
    extras = extras if extras is not None else {}
    assert all([k not in pkeys for k in extras.keys()]), f"One of extras keys ({extras.keys()}) is protected ({pkeys})."
    
    datapoint = {'smiles':d.smiles,
                 'y': [d.y.item(), ], 'x': d.x.tolist(),
                 'edge_index': d.edge_index.tolist(),
                 'pos': d.pos if d.pos is None else d.pos.tolist(),
                 'node_mask': node_mask if node_mask is None else node_mask.tolist(),
                 'edge_mask': edge_mask if edge_mask is None else edge_mask.tolist(),
                 'extras': list(extras.keys())
                }
    datapoint.update(extras)

    with open(osp.join(save_dir, fname), 'w') as f:
        json.dump(datapoint, f, indent=1)


def load_datapoint(path, device=None):
    device = device if device is not None else ('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def as_datapoint(d):
        for k in ['x', 'y', 'edge_index', 'pos', 'edge_mask', 'node_mask']:
            d[k] = torch.tensor(d[k]).to(device) if d[k] is not None else None
        
        node_mask = d['node_mask']
        edge_mask = d['edge_mask']
        extras = {(k, d[k]) for k in d['extras']}
        
        keys_to_remove = d['extras'] + ['extras', 'node_mask', 'edge_mask']
        for k in keys_to_remove:
            del(d[k])
        
        return Data(**d), node_mask, edge_mask, extras

    with open(path) as f:
        datapoint = json.load(f, object_hook=as_datapoint)
    
    return datapoint


def datapoint_equal(d1, d2):
    """Checks equality of two datapoints."""
    d1d, d1n, d1e, _ = d1  # Data, node_mask, edge_mask, extras
    d2d, d2n, d2e, _ = d2
    return all([d1d.pos == d2d.pos,
                d1d.smiles == d2d.smiles,
                d1d.y == d2d.y,
                torch.all(d1d.x == d2d.x),
                torch.all(d1d.edge_index == d2d.edge_index),
                torch.all(d1n == d2n),
                torch.all(d1e == d2e)
               ])



# # # # # # # # # # # # #
# M O D E L   U T I L S #
# # # # # # # # # # # # #

def get_model(path, model_cfg, dataset, cuda=False, validation_kwargs=None):
    """Load model weights and optionally validate that it was done correctly."""
    device = 'cuda:0' if cuda else 'cpu'
    weights = [osp.join(path, h) for h in  os.listdir(path) if 'best_model_weights.pt' in h][0]
    state_dict = torch.load(weights, map_location=torch.device(device))

    model = GraphConvNetwork(input_dim=dataset[0].x.shape[1],
                             output_dim=dataset[0].y.shape[0],
                             **model_cfg[params_section])

    # we might try to load the model with different version of pytorch/geometric
    # therefore, we update the parameters accordingly
    try:
        print("Trying once...")
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        # torch 1.9+
        new_state_dict = {}
        for key in state_dict.keys():
            if 'bn' in key:
                new_key = '.'.join(key.split('.')[:2] + ['module', ] + key.split('.')[-1:])
                new_state_dict[new_key] = state_dict[key]
            else:
                new_state_dict[key] = state_dict[key]
        try:
            print("Trying twice...")
            model.load_state_dict(new_state_dict, strict=True)
            
        except RuntimeError:
            # torch 2.0+
            new_state_dict = {}
            for k in state_dict.keys():
                if 'layers' not in k:
                    new_key = '.'.join(k.split('.')[:-1] + ['module',] +  k.split('.')[-1:])
                    new_state_dict[new_key] = state_dict[k]
                elif 'conv' in k and 'bias' not in k:
                    new_key = '.'.join(k.split('.')[:-1] + ['lin',] +  k.split('.')[-1:])
                    new_state_dict[new_key] = state_dict[k].T  # tricks!
                else:
                    new_state_dict[k] = state_dict[k]
            print("Last try...")     
            model.load_state_dict(new_state_dict, strict=True)
                
    model.eval()
    
    if cuda:
        model = model.cuda()
        
    if validation_kwargs is not None:
        try:
            validate_model(model, path, **validation_kwargs)
            return model
        except AssertionError:
            raise RuntimeError("The model was not loaded properly. Abandon all hope.")

    return model


def validate_model(model, path, repr_cfg, device):
    "Does predictions of the model match predictions stored under path?"
    for pred_file in [osp.join(path, h) for h in  os.listdir(path) if 'predictions' in h]:
        pred_df = pd.read_csv(pred_file, sep='\t')

        for smiles in tqdm(pred_df.smiles, desc="Validating model..."):
            ref_pred = float(pred_df[pred_df.smiles==smiles].predicted.values[0])
            
            this_sample, this_smi = load_data_from_smiles([smiles[2:-2], ], [0, ], **repr_cfg[utils_section])
            _, pred = predict(model, DataLoader(this_sample, batch_size=1), device)

            assert np.isclose(pred, ref_pred, atol=0.0005), f"Prediction for {this_smi} should be {ref_pred} is {pred}."
