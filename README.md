# Comparison of Atom Representations in Graph Neural Networks for Molecular Property Prediction

This repository accompanies article [*Comparison of Atom Representations in Graph Neural Networks for Molecular Property Prediction*](https://arxiv.org/abs/2012.04444) by Agnieszka Pocha, Tomasz Danel, Sabina Podlewska, Jacek Tabor and Łukasz Maziarka.

[:tv: spotlight talk](https://slideslive.com/38942399) for *Machine Learning for Molecules Workshop* at NeurIPS 2020

The extension of this work is under review.

## Main findings

We compare multiple atom representations for graph neural networks and evaluate them on the prediction of free energy, solubility, and metabolic stability in order to understand the impact of selected atomic features on the model's performance.

We find that not all atomic features are equally usefull and that removing some of them can improve the performance.

![Image](img/wilcoxon.png)

P-values of one-tailed Wilcoxon tests between the best models trained on each representation.

The value in i-th row and j-th column corresponds to the alternative hypothesis saying that the median squared error of i-th representation is greater than the median of j-th representation (superior representations have darker columns, and inferior ones have darker rows). The darkest cells are statistically significant with Bonferroni correction.


## How to use this code

### Environment

We use the following packages:

- python == 3.7
- cudatoolkit == 10.0
- pytorch == 1.4.0
- torchvision == 0.5.0
- torch-cluster == 1.4.5
- torch-geometric == 1.4.2
- torch-scatter == 2.0.3
- torch-sparse == 0.5.1
- torch-spline-conv == 1.1.1
- rdkit == 2019.09.3
- numpy == 1.18.1
- pandas == 1.0.1
- scikit-learn == 0.22.1
- matplotlib == 3.1.3

### Running the code

All scripts are in directory `scripts`.
In the following, we assume the code is run from the repository's root directory. 



#### Train graph convolutional neural networks

To train GCNs, use this command:

```bash
python scripts/main.py model.cfg dataset.cfg representation.cfg results_dir
```

The configuration files (`*.cfg`) are parsed by methods implemented in `graphrepr.config`.
Examples of configuration files and their description can be found in directory `configs` including the list of model architectures used in the study.

`results_dir` should point to a directory where the results should be saved.
For each experiment, a subdirectory called `[MODEL ID]-model_[DATASET ID]_[REPRESENTATION ID]-repr` (e.g. `1-model_esol-scaffold_7-repr`) will be created.
We encourage to create a different directory for each dataset.

**Example:** `python scripts/main.py configs/models/1-model.cfg configs/data/esol-scaffold.cfg configs/representations/7-repr.cfg results/gcn/esol-scaffold`



#### Train Directed Message Passing Neural Networks

Similarly, you can train D-MPNNs with the following command:

```bash
python scripts/main_dmpnn.py model.cfg dataset.cfg representation.cfg results_dir
```

**Example:** `python scripts/main_dmpnn.py configs/models/1-model.cfg configs/data/esol-scaffold.cfg configs/representations/7-repr.cfg results/dmpnn/esol-scaffold`



#### Run Wilcoxon analysis

To run Wilcoxon tests and generate p-value heatmaps, use this command:

```bash
python scripts/run_wilcoxon.py --model [model name] --source [results directory]
```

- `[results directory]` should point to a root directory with the results,
- `[model name]` should be a name of subdirectory for a given model (e.g. GCN or D-MPNN).


**Example:** `python scripts/run_wilcoxon.py --model gcn --source results`



#### Calculate explanations

To calculate explanations with GNNExplainer, use this command:

```bash
python scripts/calculate_explanations.py results_dir dataset run fold [--savedir saving_dir] [--supersafe]
```

Parameters `results_dir`, `dataset` and `run` are used to build a path to directory with results for all models.
`fold` determines for which fold the explanations should be calculated (e.g. `fold1`).
Option `--savedir` allows to define an alternative directory for storing results.
Otherwise the calculated explanations are saved in the directory that contains results for a specific model (e.g. `1-model_esol-scaffold_7-repr`).
If option `--supersafe` is used then equality between the calculated and the saved explanations is additionally checked.

**Example:** `python scripts/calculate_explanations.py results/gcn esol-scaffold run-1 fold1 --savedir my_explanations --supersafe`
**or** `python scripts/calculate_explanations.py results/gcn esol-scaffold run-1 fold1`


#### Calculate statistics of the explantations

To calculate statistics for GNNExplainer explanations, use this command:

```bash
python scripts/analyse_explanations.py results_dir
```

The statistics are saved to a file `analyse_explanations.csv`.

**Example:** `python scripts/analyse_explanations.py results/gcn`

## Citation

If you find our results useful, you can cite this work using the following BibTeX entry:

```
@inproceedings{pocha2021comparison,
  title={Comparison of atom representations in graph neural networks for molecular property prediction},
  author={Pocha, Agnieszka and Danel, Tomasz and Podlewska, Sabina and Tabor, Jacek and Maziarka, {\L}ukasz},
  booktitle={2021 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2021},
  organization={IEEE}
}
```
