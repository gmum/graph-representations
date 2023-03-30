import os.path as osp
import argparse
import numpy as np
import pandas as pd
import sklearn
import collections

from graphrepr.explain.utils import get_all_subfolders, get_all_files, get_configs
from graphrepr.explain.plot_utils import kolorex
from graphrepr.data import load_data_from_smiles
from graphrepr.config import utils_section

# run: python analyse_explanations.py results_path
# ex1: python analyse_explanations.py gcn_best_models


def usv(it):
    """Unpack single value"""
    assert isinstance(it, collections.abc.Iterable)
    if len(it) == 1:
        return it[0]
    elif len(it) == 0:
        return None
    else:
        raise ValueError(f'len(it)={len(it)}')
        

def make_statistics(df, repr_cfg):
    """
    Returns dictionary with confusion matrix-based statistics
    regarding how often (non-)zero-valued features are ignored.
    """ 
    # prepare data
    non_zero, over_mean, over_median = [], [], []
    for row in df.iterrows():
        smi = [row[-1].smi, ]
        mol, smiles = load_data_from_smiles(smi, [0]*len(smi), **repr_cfg[utils_section])

        non_zero.append((mol[0].x!=0).numpy().any(axis=0))
        scores = np.array([row[-1].loc[f'node_feat_mask_{i}'] for i in range(mol[0].x.shape[-1])])
        over_mean.append(scores>=np.mean(scores))

    non_zero, over_mean = np.array(non_zero), np.array(over_mean)
    
    # calculate statistics
    def _stats(confusion_matrix):
        def _ratio(numerator_idx, denominator_idx):
            # a/a+b
            denominator = confusion_matrix[numerator_idx] + confusion_matrix[denominator_idx]
            return np.nan if denominator==0 else confusion_matrix[numerator_idx]/denominator
        # confusion matrix:
        # empty unimportant, empty important, present unimportant, present important
        empty_important = _ratio(1, 0)
        present_ignored = _ratio(2, 3)
        return empty_important, present_ignored
    
    # dataset run fold part representation featurename featuregroup
    # confusion_matrix empty_important_ratio present_ignored_ratio
    dataset = usv(np.unique(df.dataset))
    run = usv(np.unique(df.run))
    fold = usv(np.unique(df.fold))
    part = usv(np.unique(df.part))
    repr_name = usv(np.unique(df.model_path)).split('_')[-1]
    fnames = kolorex[repr_name][-1]
    fgroups = []
    for group in kolorex[repr_name][0]:
        fgroups.extend([group.name,] * (group.end - group.start))
    
    records = []
    for i in range(non_zero.shape[-1]):
        conf_mat = sklearn.metrics.confusion_matrix(non_zero[:, i] , over_mean[:, i], labels=[False, True]).flatten()
        empty_important_ratio, present_ignored_ratio = _stats(conf_mat)
        empty_unimportant, empty_important, present_unimportant, present_important = conf_mat
        
        d = {'dataset': dataset, 'run': run, 'fold':fold, 'part':part, 'repr_name':repr_name,
             'featurename':fnames[i], 'featuregroup':fgroups[i],
             'empty_unimportant': empty_unimportant, 'empty_important': empty_important,
             'present_unimportant': present_unimportant, 'present_important': present_important,
             'empty_important_ratio': empty_important_ratio, 'present_ignored_ratio': present_ignored_ratio
            }
        records.append(d)
        
    return records


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=str, help='path to results')
    args = parser.parse_args()
    
    records = []
    for dataset_dir in get_all_subfolders(args.results, extend=True):
        for run_dir in get_all_subfolders(dataset_dir, extend=True):
            for model_dir in get_all_subfolders(run_dir, extend=True):
                print('\n\n', model_dir)
                _, _, repr_cfg = get_configs(model_dir)

                for masks_file in [f for f in get_all_files(model_dir, extend=True) if f.endswith('.csv')]:
                    masks_df = pd.read_csv(masks_file)
                    records.extend(make_statistics(masks_df, repr_cfg))
            
    df = pd.DataFrame.from_records(records)
    df.to_csv('analyse_explanations.csv')
