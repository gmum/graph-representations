import os
import os.path as osp

import sys
import time
import argparse
import pandas as pd
from tqdm import tqdm

import torch
from graphrepr.explain.gnnexplainer import GNNExplainer
from graphrepr.explain.utils import get_configs, get_data_and_parts, get_all_subfolders, make_directory
from graphrepr.explain.torchutils import get_model, store_datapoint, load_datapoint, datapoint_equal

import warnings  # silence!
warnings.filterwarnings('ignore')

# run: python calculate_explanations.py results_path dataset run fold [--savedir saving_path] [--supersafe]
# ex1: python calculate_explanations.py gcn_best_models esol-random run-1 fold1
# ex2  python calculate_explanations.py gcn_best_models esol-random run-1 fold1 --savedir explanations --supersafe


if __name__=="__main__":
    # PARSE ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=str, help='path to results')
    parser.add_argument('dataset', type=str)
    parser.add_argument('run', type=str)
    parser.add_argument('fold', type=str)
    parser.add_argument('--savedir', type=str, default=None,
                        help='allows to define alternative directory for storing results')
    parser.add_argument('--supersafe', action='store_const', const=True, default=False)
    args = parser.parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    timestamp = time.strftime('%Y-%m-%d')
    
    print(args.results)
    print(args.dataset, args.run, args.fold, device, timestamp)
    print('supersafe:', args.supersafe, '\n')
    
    path = osp.realpath(osp.join(args.results, args.dataset, args.run))
    
    # FOR EACH MODEL
    for model_path in [osp.join(path, d) for d in get_all_subfolders(path)]:
        print(f"Processing {model_path}")
        model_id = osp.basename(model_path)
        
        if args.savedir is not None:
            saving_dir = make_directory(args.savedir, f"{model_id}_{args.run}")
        else:
            saving_dir = model_path
        print(f"Saving to: {saving_dir}")
        
        # load data and model, create explainer
        model_cfg, data_cfg, repr_cfg = get_configs(model_path, args.dataset)
        datasets = get_data_and_parts(data_cfg, repr_cfg)
        model = get_model(osp.join(model_path, args.fold),
                          model_cfg,
                          datasets[0][0],
                          torch.cuda.is_available(),
                          validation_kwargs={'repr_cfg':repr_cfg, 'device':device})
        explainer = GNNExplainer(model, epochs=200, return_type='regression', allow_edge_mask=True, log=False)

        # over data parts (fold1, fold2, ..., test)
        for dataset, _, part in datasets:
            prefix = f"{args.fold}_{part}"
            if len([f for f in os.listdir(saving_dir) if prefix in f and 'csv' in f]) > 0:
                print(f"{model_id} {args.fold} {part} already calculated. Skipping.")
                continue
            
            samples_sdir = f"{prefix}_explanations"
            samples_path = make_directory(saving_dir, samples_sdir)
            
            # over datapoints
            results = []
            for idx in tqdm(range(len(dataset)), desc="Explaining graph..."):
                data = dataset[idx].to(device)
                node_feat_mask, edge_mask = explainer.explain_graph(data.x, data.edge_index)
                
                extras = {'dataset':args.dataset, 'run':args.run, 
                          'fold':args.fold, 'part':part, 
                          'model_path':model_path, 'timestamp':timestamp,
                          'smi_idx':idx, 'smi':data.smiles[0]}
                
                row = dict(extras)
                row.update( (f'node_feat_mask_{idxf}', f.item()) for idxf, f in enumerate(node_feat_mask) )
                results.append(row)
                    
                fname = f'{idx}.json'
                store_datapoint(samples_path, fname, data, node_feat_mask, edge_mask, extras)
                if args.supersafe:
                    assert datapoint_equal((data, node_feat_mask, edge_mask, extras),
                                        load_datapoint(osp.join(samples_path, fname), device))
            
            # S A V I N G
            dfname = f"{prefix}_nf-masks.csv"
            df = pd.DataFrame(results)
            df.to_csv(osp.join(saving_dir, dfname))
            
            tarname = f"{samples_sdir}.tar.bz2"
            _ = os.system(f'tar cfj {osp.join(saving_dir, tarname)} {samples_path}')

    print("Done.")
