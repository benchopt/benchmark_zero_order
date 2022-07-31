import re
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


usetex = mpl.checkdep_usetex(True)
params = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Computer Modern Roman"],
    "text.usetex": usetex,
}
mpl.rcParams.update(params)


SAVEFIG = True
figname = "zero_order_dim_20"

# RUN `benchopt run . --bench_config.yml` to produce the csv
# BENCH_NAME = "outputs/benchopt_run_2022-07-29_17h32m56.parquet"
BENCH_NAME = "outputs/benchopt_run_2022-07-31_10h42m19.parquet"

FLOATING_PRECISION = 1e-8
MIN_XLIM = 1e-3
MARKERS = list(plt.Line2D.markers.keys())[:-4]

SOLVERS = {
    "basinhopping[temperature=1]": "Bashin hopping",
    # "basinhopping[temperature=1]": "Bashin hopping[temperature=1]",
    # "basinhopping[temperature=10]": "Bashin hopping[temperature=10]",
    "nevergrad[seed=42,solver=NGOpt]": "Nevergrad[NGOpt]",
    "nevergrad[seed=42,solver=RandomSearch]": "Nevergrad[Randomsearch]",
    'nevergrad[seed=42,solver=ScrHammersleySearch]': 'nevergrad[ScrHammersleySearch]',
    'nevergrad[seed=42,solver=TwoPointsDE]': 'nevergrad[TwoPointsDE]',
    'nevergrad[seed=42,solver=CMA]': 'nevergrad[CMA]',
    "scipy[seed=42,solver=Nelder-Mead]": "scipy[Nelder-Mead]",
    "scipy[seed=42,solver=Powell]": "scipy[Powell]",
    "scipy[seed=42,solver=BFGS]": "scipy[BFGS]",
    # "nevergrad[solver=NGOpt]": "Nevergrad[solver=NGOpt]",
    # "nevergrad[solver=RandomSearch]": "Nevergrad[solver=Randomsearch]",
    # "scipy[solver=Nelder-Mead]": "scipy[solver=Nelder-Mead]",
    # "scipy[solver=Powell]": "scipy[solver=Powell]",
    # "scipy[solver=BFGS]": "scipy[solver=BFGS]",
    'optuna[seed=42,solver=cmaes]': 'optuna[CMA]',
    'optuna[seed=42,solver=TPE]': 'optuna[TPE]',
    'optuna[seed=42,solver=RandomSearch]': 'optuna[RandomSearch]',
}

all_solvers = SOLVERS.keys()

DICT_XLIM = {
    "FCN[dimension=20,function=ackley]": 1e-5,
    "FCN[dimension=20,function=rastrigin]": 5e0,
    "FCN[dimension=20,function=rosenbrock]": 5e1,
}

DICT_TITLE = {"Zero-order test functions": "Zero-order test functions"}

DICT_YLABEL = {
    "FCN[dimension=20,function=ackley]": "Ackley",
    "FCN[dimension=20,function=rastrigin]": "Rastrigin",
    "FCN[dimension=20,function=rosenbrock]": "Rosenbrock",
}

DICT_YTICKS = {
    "FCN[dimension=20,function=ackley]": [1e0, 1e-6, 1e-10],
    "FCN[dimension=20,function=rastrigin]": [1e2, 1e-6, 1e-12],
    "FCN[dimension=20,function=rosenbrock]": [1e3, 1e-3, 1e-8],
}

DICT_XTICKS = {
    "FCN[dimension=20,function=ackley]": np.geomspace(1e-4, 1e2, 4),
    "FCN[dimension=20,function=rastrigin]": np.geomspace(1e-4, 1e0, 4),
    "FCN[dimension=20,function=rosenbrock]": np.geomspace(1e-4, 1e1, 4),
}

CMAP = plt.get_cmap("tab20")
style = {solv: (CMAP(i), MARKERS[i]) for i, solv in enumerate(all_solvers)}


df = pd.read_parquet(BENCH_NAME)


df = df[df['solver_name'].isin(all_solvers)]
solvers = df["solver_name"].unique()
solvers = np.array(sorted(solvers, key=lambda key: SOLVERS[key].lower()))
datasets = [
    "FCN[dimension=20,function=ackley]",
    "FCN[dimension=20,function=rastrigin]",
    "FCN[dimension=20,function=rosenbrock]",
]

objectives = df["objective_name"].unique()

titlesize = 22
ticksize = 16
labelsize = 20
regex = re.compile(r"\[(.*?)\]")

plt.close("all")
fig1, axarr = plt.subplots(
    len(datasets),
    len(objectives),
    sharex=False,
    sharey="row",
    figsize=[9, 1 + 2 * len(datasets)],
    constrained_layout=True,
    squeeze=False,
)

for idx_data, dataset in enumerate(datasets):
    df1 = df[df["data_name"] == dataset]
    for idx_obj, objective in enumerate(objectives):
        df2 = df1[df1["objective_name"] == objective]
        ax = axarr[idx_data, idx_obj]
        c_star = np.min(df2["objective_value"]) - FLOATING_PRECISION
        for i, solver_name in enumerate(solvers):
            df3 = df2[df2["solver_name"] == solver_name]
            curve = df3.groupby("stop_val").median()

            y = curve["objective_value"] - c_star

            linestyle = "-"
            if solver_name in ("snapml[gpu=True]", "cuml[qn]", "cuml[cd]"):
                linestyle = "--"
            # ax.loglog(
            ax.semilogx(
                curve["time"],
                y,
                color=style[solver_name][0],
                marker=style[solver_name][1],
                markersize=6,
                label=SOLVERS[solver_name],
                linewidth=2,
                markevery=3,
                linestyle=linestyle,
            )

        # ax.set_xlim([DICT_XLIM.get(dataset, MIN_XLIM), ax.get_xlim()[1]])
        axarr[len(datasets) - 1, idx_obj].set_xlabel(
            "Time (s)", fontsize=labelsize
        )
        axarr[0, idx_obj].set_title(DICT_TITLE[objective], fontsize=labelsize)

        ax.grid()
        ax.set_xticks(DICT_XTICKS[dataset])
        ax.tick_params(axis="both", which="major", labelsize=ticksize)

    if regex.search(dataset) is not None:
        dataset_label = (
            regex.sub("", dataset)
            + "\n"
            + "\n".join(regex.search(dataset).group(1).split(","))
        )
    else:
        dataset_label = dataset
    axarr[idx_data, 0].set_ylabel(DICT_YLABEL[dataset], fontsize=labelsize)
    # axarr[idx_data, 0].set_yticks(DICT_YTICKS[dataset])

plt.show(block=False)


fig2, ax2 = plt.subplots(1, 1, figsize=(20, 4))
n_col = 3
if n_col is None:
    n_col = len(axarr[0, 0].lines)

ax = axarr[0, 0]
lines_ordered = list(
    itertools.chain(*[ax.lines[i::n_col] for i in range(n_col)])
)
legend = ax2.legend(
    lines_ordered,
    [line.get_label() for line in lines_ordered],
    ncol=n_col,
    loc="upper center",
)
fig2.canvas.draw()
fig2.tight_layout()
width = legend.get_window_extent().width
height = legend.get_window_extent().height
fig2.set_size_inches((width / 80, max(height / 80, 0.5)))
plt.axis("off")
plt.show(block=False)


if SAVEFIG:
    Path("figures").mkdir(exist_ok=True)
    fig1_name = f"figures/{figname}.pdf"
    fig1.savefig(fig1_name)

    fig2_name = f"figures/{figname}_legend.pdf"
    fig2.savefig(fig2_name)
