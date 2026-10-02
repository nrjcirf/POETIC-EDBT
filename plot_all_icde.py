# -*- coding: utf-8 -*-

"""
=========================================================
ICDE Visualization System (FULL FIXED VERSION)
=========================================================

适配：
    parsed_tdrive.json
    parsed_nyc.json

修复：
    ✔ KeyError
    ✔ 字段自动兼容
    ✔ 缺失字段自动跳过
    ✔ 不同parser格式兼容
    ✔ publication-ready

输出：
    figures/*.pdf

=========================================================
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# Global Style
# =========================================================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["pdf.fonttype"] = 42


COLORS = {
    "POETIC-Full": "#d62728",
    "POETIC-NoDecouple": "#1f77b4",
    "Truthful-DP-MAB": "#2ca02c",
    "Non-Private Optimal": "#9467bd",
}

MARKERS = {
    "POETIC-Full": "o",
    "POETIC-NoDecouple": "s",
    "Truthful-DP-MAB": "^",
    "Non-Private Optimal": "D",
}


# =========================================================
# Utils
# =========================================================

def ensure_dir(path):

    if not os.path.exists(path):
        os.makedirs(path)


ensure_dir("figures")


def load_json(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# Robust Value Reader
# =========================================================

def get_metric(d, candidates, default=0):

    for c in candidates:

        if c in d:
            return d[c]

    return default


# =========================================================
# Figure 2
# Overall Performance
# =========================================================

def plot_fig2_overall(td, nyc):

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 8)
    )

    datasets = [
        ("T-Drive", td),
        ("NYC", nyc)
    ]

    for row, (dataset_name, data) in enumerate(datasets):

        exp1 = data.get("Exp1", {})

        methods = list(exp1.keys())

        utility = [
            get_metric(
                exp1[m],
                ["Val", "Utility", "FinalVal"]
            )
            for m in methods
        ]

        esterr = [
            get_metric(
                exp1[m],
                ["EstErr", "Error"]
            )
            for m in methods
        ]

        regret = [
            max(utility) - u
            for u in utility
        ]

        # =====================================
        # Utility
        # =====================================

        ax = axes[row][0]

        ax.bar(
            methods,
            utility,
            color=[
                COLORS.get(m, "gray")
                for m in methods
            ]
        )

        ax.set_title(
            f"{dataset_name} - Utility"
        )

        ax.tick_params(
            axis='x',
            rotation=15
        )

        # =====================================
        # Regret
        # =====================================

        ax = axes[row][1]

        ax.bar(
            methods,
            regret,
            color=[
                COLORS.get(m, "gray")
                for m in methods
            ]
        )

        ax.set_title(
            f"{dataset_name} - Regret"
        )

        ax.tick_params(
            axis='x',
            rotation=15
        )

        # =====================================
        # EstErr
        # =====================================

        ax = axes[row][2]

        ax.bar(
            methods,
            esterr,
            color=[
                COLORS.get(m, "gray")
                for m in methods
            ]
        )

        ax.set_title(
            f"{dataset_name} - Estimation Error"
        )

        ax.tick_params(
            axis='x',
            rotation=15
        )

    plt.tight_layout()

    plt.savefig(
        "figures/icde_Fig2_Overall.pdf",
        bbox_inches="tight"
    )

    plt.close()


# =========================================================
# Figure 3
# Strategic Worker
# =========================================================

def plot_fig3_strategic(td, nyc):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4)
    )

    datasets = [
        ("T-Drive", td),
        ("NYC", nyc)
    ]

    ratios = [0, 10, 30, 50, 70]

    for dataset_name, data in datasets:

        exp7 = data.get("Exp7", [])

        methods = sorted(list(set([
            x["method"]
            for x in exp7
        ])))

        # =====================================
        # Utility
        # =====================================

        ax = axes[0]

        for method in methods:

            ys = []

            for r in ratios:

                found = [
                    x for x in exp7
                    if x["method"] == method
                    and x["ratio"] == r
                ]

                if len(found) == 0:
                    ys.append(np.nan)
                else:
                    ys.append(
                        get_metric(
                            found[0],
                            ["Val", "Utility"]
                        )
                    )

            ax.plot(
                ratios,
                ys,
                marker=MARKERS.get(method, "o"),
                linewidth=2,
                label=f"{dataset_name}-{method}"
            )

        ax.set_title("Utility")

        ax.set_xlabel("Strategic Worker Ratio (%)")

        ax.set_ylabel("Cumulative Utility")

        # =====================================
        # EstErr
        # =====================================

        ax = axes[1]

        for method in methods:

            ys = []

            for r in ratios:

                found = [
                    x for x in exp7
                    if x["method"] == method
                    and x["ratio"] == r
                ]

                if len(found) == 0:
                    ys.append(np.nan)
                else:
                    ys.append(
                        get_metric(
                            found[0],
                            ["EstErr"]
                        )
                    )

            ax.plot(
                ratios,
                ys,
                marker=MARKERS.get(method, "o"),
                linewidth=2,
                label=f"{dataset_name}-{method}"
            )

        ax.set_title("Estimation Error")

        ax.set_xlabel("Strategic Worker Ratio (%)")

        ax.set_ylabel("Estimation Error")

        # =====================================
        # Variance
        # =====================================

        ax = axes[2]

        for method in methods:

            ys = []

            for r in ratios:

                found = [
                    x for x in exp7
                    if x["method"] == method
                    and x["ratio"] == r
                ]

                if len(found) == 0:
                    ys.append(np.nan)
                else:
                    ys.append(
                        get_metric(
                            found[0],
                            ["Var", "Variance"]
                        )
                    )

            ax.plot(
                ratios,
                ys,
                marker=MARKERS.get(method, "o"),
                linewidth=2,
                label=f"{dataset_name}-{method}"
            )

        ax.set_title("Variance")

        ax.set_xlabel("Strategic Worker Ratio (%)")

        ax.set_ylabel("Variance")

    axes[0].legend(fontsize=7)

    plt.tight_layout()

    plt.savefig(
        "figures/icde_Fig3_Strategic.pdf",
        bbox_inches="tight"
    )

    plt.close()


# =========================================================
# Figure 7
# Privacy-Learning Tradeoff
# =========================================================

def plot_fig7_privacy(td, nyc):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8)
    )

    datasets = [
        ("T-Drive", td),
        ("NYC", nyc)
    ]

    for row, (dataset_name, data) in enumerate(datasets):

        exp5 = data.get("Exp5", [])

        methods = sorted(list(set([
            x["method"]
            for x in exp5
        ])))

        # =====================================
        # Utility
        # =====================================

        ax = axes[row][0]

        for method in methods:

            xs = []
            ys = []

            for item in exp5:

                if item["method"] == method:

                    eps = item["epsilon"]

                    if eps == "inf":
                        eps = 20

                    xs.append(float(eps))

                    ys.append(
                        get_metric(
                            item,
                            ["Val", "Utility"]
                        )
                    )

            order = np.argsort(xs)

            xs = np.array(xs)[order]
            ys = np.array(ys)[order]

            ax.plot(
                xs,
                ys,
                marker=MARKERS.get(method, "o"),
                linewidth=2,
                label=method
            )

        ax.set_title(
            f"{dataset_name} - Utility"
        )

        ax.set_xlabel("Privacy Budget ε")

        ax.set_ylabel("Utility")

        # =====================================
        # EstErr
        # =====================================

        ax = axes[row][1]

        for method in methods:

            xs = []
            ys = []

            for item in exp5:

                if item["method"] == method:

                    eps = item["epsilon"]

                    if eps == "inf":
                        eps = 20

                    xs.append(float(eps))

                    ys.append(
                        get_metric(
                            item,
                            ["EstErr"]
                        )
                    )

            order = np.argsort(xs)

            xs = np.array(xs)[order]
            ys = np.array(ys)[order]

            ax.plot(
                xs,
                ys,
                marker=MARKERS.get(method, "o"),
                linewidth=2,
                label=method
            )

        ax.set_title(
            f"{dataset_name} - Estimation Error"
        )

        ax.set_xlabel("Privacy Budget ε")

        ax.set_ylabel("Estimation Error")

    axes[0][0].legend(fontsize=8)

    plt.tight_layout()

    plt.savefig(
        "figures/icde_Fig7_Privacy.pdf",
        bbox_inches="tight"
    )

    plt.close()


# =========================================================
# Figure 8
# Ablation
# =========================================================

def plot_fig8_ablation(td, nyc):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4)
    )

    datasets = [
        ("T-Drive", td),
        ("NYC", nyc)
    ]

    metrics = [
        ("Val", 0),
        ("EstErr", 1),
        ("Var", 2)
    ]

    for dataset_name, data in datasets:

        exp6 = data.get("Exp6", {})

        methods = list(exp6.keys())

        for metric, idx in metrics:

            ax = axes[idx]

            ys = []

            for m in methods:

                ys.append(
                    get_metric(
                        exp6[m],
                        [metric]
                    )
                )

            ax.bar(
                np.arange(len(methods)),
                ys,
                alpha=0.7,
                label=dataset_name
            )

            ax.set_xticks(
                np.arange(len(methods))
            )

            ax.set_xticklabels(
                methods,
                rotation=25
            )

            ax.set_title(metric)

    axes[0].legend()

    plt.tight_layout()

    plt.savefig(
        "figures/icde_Fig8_Ablation.pdf",
        bbox_inches="tight"
    )

    plt.close()


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    print("Loading parsed json...")

    td = load_json("parsed_tdrive.json")

    nyc = load_json("parsed_nyc.json")

    print("Generating figures...")

    plot_fig2_overall(td, nyc)

    plot_fig3_strategic(td, nyc)

    plot_fig7_privacy(td, nyc)

    plot_fig8_ablation(td, nyc)

    print("All figures generated successfully.")