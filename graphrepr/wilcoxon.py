import argparse
import json
import os
import re
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import from_levels_and_colors
from scipy.stats import wilcoxon


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calculates the squared error of the predictions"""
    return np.power(y_true - y_pred, 2)


def get_validation_results(datadir: str, metric: str = "mse"):
    df = pd.DataFrame()
    metric_id = 0 if metric == "mse" else 1
    for root, dirs, files in os.walk(datadir):
        for filename in files:
            if filename.endswith("mse-mae.json"):
                with open(os.path.join(root, filename), "r") as file:
                    content = json.load(file)
                score = content["val"][metric_id]
                try:
                    repr_id = re.findall(r"(\d+)-repr", root)[0]
                except:
                    # probably literature repr
                    continue
                fold_id = re.findall(r"fold(\d+)", root)[0]
                try:
                    model_id = re.findall(r"(\d+)-model", root)[0]
                except:
                    model_id = re.findall(r"mat-lr-(\d+)_", root)[0]
                run_id = re.findall(r"run-(\d+)", root)[0]
                df = df.append(
                    {
                        "repr": repr_id,
                        "model": model_id,
                        "run": run_id,
                        "fold": fold_id,
                        "score": score,
                    },
                    ignore_index=True,
                )
    return df


def get_best_models_for_reprs(df, n_top=3):
    return (
        df.groupby(["model", "repr"])
        .mean()
        .sort_values("score")
        .reset_index()
        .groupby("repr")
        .head(n_top)
        .sort_values("repr")
    )


def load_predictions(dataset, datadir, repr_id, run_id, fold_id, model_id):
    if "mat" in datadir:
        model_name = f"mat-lr-{model_id}"
    else:
        model_name = f"{model_id}-model"
    dirname = (
        f"{datadir}/repr-{repr_id}/run-{run_id}/"
        f"{model_name}_{dataset}_{repr_id}-repr/fold{fold_id}"
    )
    filename = [
        path for path in os.listdir(dirname) if path.endswith("test.predictions")
    ][0]
    df = pd.read_csv(os.path.join(dirname, filename), sep="\t", index_col=0)
    df["repr"] = repr_id
    df["run"] = run_id
    df["fold"] = fold_id
    df["model"] = model_id
    return df


def load_all_preds(dataset, datadir, df_top):
    dfs = []
    for i in range(len(df_top) // len(df_top.repr.unique())):
        df = pd.DataFrame()
        for repr_id in df_top.repr.unique():
            model_id = df_top[df_top.repr == repr_id].sort_values("score").iloc[i].model
            for run_id in range(1, 4):
                for fold_id in range(1, 2):
                    df_pred = load_predictions(
                        dataset, datadir, repr_id, run_id, fold_id, model_id
                    )
                    df = pd.concat([df, df_pred], axis=0)
        df["repr"] = df.repr.apply(lambda i: f"repr-{i}")
        dfs.append(df)
    return dfs


def test_representations_df(result_df, metric=mse, one_tailed=False, average=None):
    p_val = np.zeros([12, 12])

    for x, y in combinations(range(1, 13), 2):
        if average:
            if average == "mean":
                make_df = (
                    lambda i: result_df[result_df.repr == f"repr-{i}"]
                    .groupby(level=0)
                    .mean()
                )
            if average == "median":
                make_df = (
                    lambda i: result_df[result_df.repr == f"repr-{i}"]
                    .groupby(level=0)
                    .median()
                )
        else:
            make_df = lambda i: result_df[result_df.repr == f"repr-{i}"]

        if one_tailed:
            _, p = wilcoxon(
                metric(make_df(x).actual, make_df(x).predicted),
                metric(make_df(y).actual, make_df(y).predicted),
                alternative="greater",
            )
            p_val[x - 1, y - 1] = p
            _, p = wilcoxon(
                metric(make_df(y).actual, make_df(y).predicted),
                metric(make_df(x).actual, make_df(x).predicted),
                alternative="greater",
            )
            p_val[y - 1, x - 1] = p
        else:
            _, p = wilcoxon(
                metric(make_df(x).actual, make_df(x).predicted),
                metric(make_df(y).actual, make_df(y).predicted),
            )
            p_val[x - 1, y - 1] = p
            p_val[y - 1, x - 1] = p
    return p_val


def plot_pval(p_val, one_tailed=False, ax=None, cbar_ax=None, annot=True):
    mask = np.zeros_like(p_val, dtype=np.bool)
    mask[np.diag_indices_from(mask)] = True

    if one_tailed:
        colors = sns.color_palette("rocket", 6)
        levels = [0, 0.000375, 0.00075, 0.0015, 0.03, 0.05]
    else:
        colors = sns.color_palette("rocket", 5)
        levels = [0, 0.00075, 0.0015, 0.03, 0.05]
    cmap, norm = from_levels_and_colors(levels, colors, extend="max")

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        p_val,
        ax=ax,
        cbar_ax=cbar_ax,
        cmap=cmap,
        norm=norm,
        square=True,
        linewidths=0.05,
        annot=annot,
        fmt=".4f",
        xticklabels=range(1, 13),
        yticklabels=range(1, 13),
        mask=mask,
        cbar=True if cbar_ax or annot else False,
        vmax=0.05,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model name")
    parser.add_argument(
        "--source",
        type=str,
        default='data-graphrepr',
        help="name of the directory containing model predictions",
    )
    args = parser.parse_args()

    dfs = {}
    model = args.model
    for dataset in ("rat", "human", "qm9-random", "esol-random", "esol-scaffold"):
        df = get_validation_results(f"{args.source}/{model}/{dataset}")
        df_top = get_best_models_for_reprs(df, n_top=3)
        dfs[dataset] = pd.concat(
            load_all_preds(dataset, f"{args.source}/{model}/{dataset}", df_top)
        )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=6,
        figsize=(20, 4),
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 0.05]},
    )

    dataname_map = {
        "rat": "Rat",
        "human": "Human",
        "qm9-random": "QM9",
        "esol-random": "ESOL (random)",
        "esol-scaffold": "ESOL (scaffold)",
    }

    for dataset_name, ax in zip(
        ("rat", "human", "qm9-random", "esol-random", "esol-scaffold"), axes
    ):
        result_df = dfs[dataset_name]
        if model == "dmpnn":
            for column in ("actual", "predicted", "actual - predicted"):
                if result_df[column].dtype != float:
                    result_df[column] = result_df[column].apply(
                        lambda x: float(x[1:-1])
                    )
        ax.set_title(dataname_map[dataset_name])
        pval = test_representations_df(
            result_df, metric=mse, one_tailed=True, average="mean"
        )
        plot_pval(
            pval,
            ax=ax,
            annot=False,
            cbar_ax=axes[-1] if dataset_name == "rat" else None,
        )
    plt.savefig(f"wilcoxon-{model}.pdf")
    plt.show()
