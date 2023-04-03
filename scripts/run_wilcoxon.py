import argparse

import matplotlib.pyplot as plt
import pandas as pd

from graphrepr.wilcoxon import (
    get_best_models_for_reprs,
    get_validation_results,
    load_all_preds,
    mse,
    plot_pval,
    test_representations_df,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model name")
    parser.add_argument(
        "--source",
        type=str,
        default="data-graphrepr",
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
