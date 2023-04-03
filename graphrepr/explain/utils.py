import os
import os.path as osp

from ..config import parse_data_config, parse_model_config, parse_representation_config
from ..config import utils_section, data_section, csv_section
from ..data import load_dataset


def get_all_subfolders(path, extend=False):
    subfolders = [folder for folder in os.listdir(path) if osp.isdir(osp.join(path, folder))]
    if extend:
        subfolders = [osp.join(path, f) for f in subfolders]
    return subfolders


def get_all_files(path, extend=False):
    files = [folder for folder in os.listdir(path) if osp.isfile(osp.join(path, folder))]
    if extend:
        files = [osp.join(path, f) for f in files]
    return files


def make_directory(where, dname):
    try:
        os.makedirs(osp.join(where, dname))
    except FileExistsError:
        pass
    return osp.join(where, dname)


def get_configs(path, dname=None):
    """Load model, data and representation configs from path"""
    configs = [osp.join(path, f) for f in  os.listdir(path) if osp.isfile(osp.join(path, f))]
    model_cfg = [cfg for cfg in configs if 'model' in osp.basename(cfg)][0]
    repr_cfg = [cfg for cfg in configs if 'repr' in osp.basename(cfg)][0]
    
    allowed = ['esol', 'human', 'rat', 'qm9'] if dname is None else [dname, ]
    data_cfg = [cfg for cfg in configs if any([d in osp.basename(cfg) for d in allowed])][0]

    model_cfg = parse_model_config(model_cfg)
    data_cfg = parse_data_config(data_cfg)
    repr_cfg = parse_representation_config(repr_cfg)
    
    return model_cfg, data_cfg, repr_cfg


def get_data(data_cfg, repr_cfg):
    """Load dataset."""
    if data_cfg[utils_section]['cv']:
        # joined train and validation; test
        parts = [(data_section, k) for k in data_cfg[data_section].keys()] + [(utils_section, 'test'),]
    else:
        # train, validation, test
        parts = [(data_section, 'train'), (data_section, 'valid'), (utils_section, 'test')]
    
    results = []
    for section, part in parts:
        data_path = data_cfg[section][part]
        dataset, smiles = load_dataset([data_path, ], **data_cfg[csv_section], **repr_cfg[utils_section])
        results.append((dataset, smiles))
    return results


def get_data_and_parts(data_cfg, repr_cfg):
    """Load dataset."""
    if data_cfg[utils_section]['cv']:
        # joined train and validation; test
        parts = [(data_section, k) for k in data_cfg[data_section].keys()] + [(utils_section, 'test'),]
    else:
        # train, validation, test
        parts = [(data_section, 'train'), (data_section, 'valid'), (utils_section, 'test')]
    
    results = []
    for section, part in parts:
        data_path = data_cfg[section][part]
        dataset, smiles = load_dataset([data_path, ], **data_cfg[csv_section], **repr_cfg[utils_section])
        results.append((dataset, smiles, part))
    return results
