#!/usr/bin/env python3
from palimpzest.profiler import Profiler, StatsProcessor
import palimpzest as pz


from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from tabulate import tabulate

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import argparse
import json
import shutil
import subprocess
import time
import os
import pdb


def get_color(workload, result_dict, plan_idx):
    color = "black"
    if (
        workload != "real-estate"
        and len(set(filter(None, result_dict["plan_info"]["models"]))) > 1
    ):
        color = "green"

    elif (
        workload != "real-estate"
        and len(set(filter(None, result_dict["plan_info"]["models"]))) > 2
    ):
        color = "green"

    elif "codegen-with-fallback" in result_dict["plan_info"]["query_strategies"]:
        color = "green"

    elif any(
        [
            budget is not None and budget < 1.0
            for budget in result_dict["plan_info"]["token_budgets"]
        ]
    ):
        color = "green"

    elif workload == "real-estate":
        # give color to logical re-orderings on real-estate
        if result_dict["plan_info"]["models"][1] == "gpt-4-vision-preview":
            color = "green"

    elif workload == "enron" and plan_idx == 0:
        color = "green"

    return color


def get_pareto_indices(result_dicts, col):
    pareto_indices = []
    for idx, result_dict in enumerate(result_dicts):
        col_i, quality_i = result_dict[col], result_dict["f1_score"]
        paretoFrontier = True

        # check if any other plan dominates plan i
        for j, _result_dict in enumerate(result_dicts):
            col_j, quality_j = _result_dict[col], _result_dict["f1_score"]
            if idx == j:
                continue

            # if plan i is dominated by plan j, set paretoFrontier = False and break
            if col_j <= col_i and quality_j >= quality_i:
                paretoFrontier = False
                break

        # add plan i to pareto frontier if it's not dominated
        if paretoFrontier:
            pareto_indices.append(idx)

    return pareto_indices


def plot_runtime_cost_vs_quality(results):
    # Filter to only include real-estate data
    results = {'real-estate': results.get('real-estate', [])}

    # Create figure with adjusted subplot configuration
    fig_text, axs_text = plt.subplots(nrows=2, ncols=1, figsize=(7, 4))  # Adjusted for a single column
    fig_clean_mc, axs_clean_mc = plt.subplots(nrows=2, ncols=1, figsize=(7, 4))  # Adjusted for a single column

    # Since we only have one workload, we can simplify the plotting logic
    workload = 'real-estate'
    result_tuples = results[workload]
    col = 0  # Only one column

    for plan_idx, result_dict in result_tuples:
        runtime = result_dict["runtime"]
        cost = result_dict["cost"]
        f1_score = result_dict["f1_score"]
        text = f"{plan_idx}"

        # Set label and color
        color = get_color(workload, result_dict, plan_idx)
        marker = "D" if color == "black" else None

        # Plot runtime vs. f1_score and cost vs. f1_score
        axs_text[0].scatter(f1_score, runtime, alpha=0.6, color=color)
        axs_text[1].scatter(f1_score, cost, alpha=0.6, color=color)
        axs_clean_mc[0].scatter(f1_score, runtime, alpha=0.6, color=color, marker=marker)
        axs_clean_mc[1].scatter(f1_score, cost, alpha=0.6, color=color, marker=marker)

        # Add annotations
        axs_text[0].annotate(text, (f1_score, runtime))
        axs_text[1].annotate(text, (f1_score, cost))

        # Compute pareto frontiers across all optimizations
        all_result_dicts = list(map(lambda tup: tup[1], result_tuples))
        cost_pareto_lst_indices = get_pareto_indices(all_result_dicts, "cost")
        runtime_pareto_lst_indices = get_pareto_indices(all_result_dicts, "runtime")

        # Plot line for pareto frontiers
        cost_pareto_qualities = [all_result_dicts[idx]["f1_score"] for idx in cost_pareto_lst_indices]
        pareto_costs = [all_result_dicts[idx]["cost"] for idx in cost_pareto_lst_indices]
        cost_pareto_curve = sorted(zip(cost_pareto_qualities, pareto_costs), key=lambda tup: tup[0])
        pareto_cost_xs, pareto_cost_ys = zip(*cost_pareto_curve)

        runtime_pareto_qualities = [all_result_dicts[idx]["f1_score"] for idx in runtime_pareto_lst_indices]
        pareto_runtimes = [all_result_dicts[idx]["runtime"] for idx in runtime_pareto_lst_indices]
        runtime_pareto_curve = sorted(zip(runtime_pareto_qualities, pareto_runtimes), key=lambda tup: tup[0])
        pareto_runtime_xs, pareto_runtime_ys = zip(*runtime_pareto_curve)

        axs_text[0].plot(pareto_runtime_xs, pareto_runtime_ys, color="#ef9b20", linestyle="--")
        axs_text[1].plot(pareto_cost_xs, pareto_cost_ys, color="#ef9b20", linestyle="--")
        axs_clean_mc[0].plot(pareto_runtime_xs, pareto_runtime_ys, color="#ef9b20", linestyle="--")
        axs_clean_mc[1].plot(pareto_cost_xs, pareto_cost_ys, color="#ef9b20", linestyle="--")

        # Set x,y-lim for each workload
        left, right = 0.72, 0.85
        axs_text[0].set_xlim(left, right)
        axs_text[0].set_ylim(ymin=0,ymax=1700)
        axs_text[1].set_xlim(left, right)
        axs_text[1].set_ylim(ymin=0,ymax=5.8)
        axs_clean_mc[0].set_xlim(left, right)
        axs_clean_mc[0].set_ylim(ymin=0,ymax=1700)
        axs_clean_mc[1].set_xlim(left, right)
        axs_clean_mc[1].set_ylim(ymin=0,ymax=5.8)

        # Turn on grid lines
        axs_text[0].grid(True, alpha=0.4)
        axs_text[1].grid(True, alpha=0.4)
        axs_clean_mc[0].grid(True, alpha=0.4)
        axs_clean_mc[1].grid(True, alpha=0.4)

    # Set labels and titles
    axs_text[0].set_ylabel("Single-Threaded\nRuntime (seconds)", fontsize=12)
    axs_text[1].set_ylabel("Cost (USD)", fontsize=12)
    axs_text[1].set_xlabel("F1 Score", fontsize=12)
    axs_clean_mc[0].set_ylabel("Single-Threaded\nRuntime (seconds)", fontsize=12)
    axs_clean_mc[1].set_ylabel("Cost (USD)", fontsize=12)
    axs_clean_mc[1].set_xlabel("F1 Score", fontsize=12)

    # Save figures
    fig_text.savefig("final-eval-results/plots/real-estate-text.png", dpi=500, bbox_inches="tight")
    fig_clean_mc.savefig("final-eval-results/plots/real-estate-clean-mc.png", dpi=500, bbox_inches="tight")

def plot_reopt(results, workload):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 1))  # Adjusted figsize for smaller height


    # parse results into fields
    results_df = pd.DataFrame(results)
    plan_to_ord = {"Baseline": 0, "PZ": 1, "Best": 2}
    results_df["plan_ord"] = results_df.plan.apply(lambda plan: plan_to_ord[plan])

    plots = [
        ("real-estate", "runtime", 0),
        ("real-estate", "cost", 1),
        ("real-estate", "f1_score", 2),
    ]

    for workload, metric, col in plots:
        data_df = results_df[(results_df.workload == workload) & (results_df.metric == metric)]
        if metric == "runtime":
            data_df["value"] = data_df.value / 60.0  # Convert runtime to minutes

        policy_to_label_col = {
            "max-quality-at-fixed-cost": "Policy A",
            "max-quality-at-fixed-runtime": "Policy B",
            "min-cost-at-fixed-quality": "Policy C",
        }

        data_df["label_col"] = data_df.apply(
            lambda row: (policy_to_label_col[row["policy"]] if row["plan"] == "PZ" else "Baseline"),
            axis=1,
        )

        label_col_to_ord = {"Baseline": 0, "Policy A": 1, "Policy B": 2, "Policy C": 3}
        data_df["label_col_ord"] = data_df.label_col.apply(lambda label: label_col_to_ord[label])

        # drop duplicates for baseline, which is replicated across policies
        data_df.drop_duplicates(subset=["label_col"], inplace=True)
        data_df.sort_values(by=["plan_ord", "label_col_ord"], inplace=True)

        # rename Baseline --> GPT-4 for plot
        data_df["label_col"] = data_df.label_col.apply(lambda label_col: "GPT-4" if label_col == "Baseline" else label_col)

        g = sns.barplot(
            data=data_df,
            x="value",
            y="label_col",
            hue="plan",
            alpha=0.6,
            ax=axs[col],
            width=0.8  # Adjust bar width here
        )

        # Set x-labels
        xlabel = "Cost (USD)" if metric == "cost" else metric.capitalize()
        if metric == "runtime":
            xlabel = "Single-Threaded Runtime (minutes)"
        elif metric == "f1_score":
            xlabel = "F1-Score"
        g.set_xlabel(xlabel, fontsize=12)

        # Remove legends and tick labels if not the first plot
        if col > 0:
            g.set_yticklabels([])
            g.legend_.remove()
        else:
            g.legend(title='')  # Remove the legend title

        # Set x-limits for F1 score
        if metric == "f1_score":
            axs[col].set_xlim(0, 1.05)

        # axs[col].set_title(xlabel, fontsize=15)
        g.set_ylabel('')  # Remove y-axis label

    fig.savefig(f"final-eval-results/plots/reopt_real_estate.png", dpi=500, bbox_inches="tight")

if __name__ == "__main__":
    # parse arguments
    startTime = time.time()
    parser = argparse.ArgumentParser(description="Run the evaluation(s) for the paper")
    parser.add_argument("--all", default=False, action="store_true", help="")
    parser.add_argument("--reopt", default=False, action="store_true", help="")
    # parser.add_argument('--workload', type=str, help='The workload: one of ["biofabric", "enron", "real-estate"]')
    # parser.add_argument('--opt' , type=str, help='The optimization: one of ["model", "codegen", "token-reduction"]')

    args = parser.parse_args()

    # create directory for intermediate results
    os.makedirs(f"final-eval-results/plots", exist_ok=True)

    if args.all:
        # # opt and workload to # of plots
        # opt_workload_to_num_plans = {
        #     "model": {
        #         "enron": 11,
        #         "real-estate": 10,
        #         "biofabric": 14,
        #     },
        #     "codegen": {
        #         "enron": 6,
        #         "real-estate": 7,
        #         "biofabric": 6,
        #     },
        #     "token-reduction": {
        #         "enron": 12,
        #         "real-estate": 16,
        #         "biofabric": 16,
        #     },
        # }

        # # read results file(s) generated by evaluate_pz_plans
        # results = {
        #     "enron": {
        #         "model": [],
        #         "codegen": [],
        #         "token-reduction": [],
        #     },
        #     "real-estate": {
        #         "model": [],
        #         "codegen": [],
        #         "token-reduction": [],
        #     },
        #     "biofabric": {
        #         "model": [],
        #         "codegen": [],
        #         "token-reduction": [],
        #     },
        # }
        # for workload in ["enron", "real-estate", "biofabric"]:
        #     for opt in ["model", "codegen", "token-reduction"]:
        #         num_plans = opt_workload_to_num_plans[opt][workload]
        #         for plan_idx in range(num_plans):
        #             if workload == "biofabric" and opt == "codegen" and plan_idx == 3:
        #                 continue

        #             with open(f"final-eval-results/{opt}/{workload}/results-{plan_idx}.json", 'r') as f:
        #                 result = json.load(f)
        #                 results[workload][opt].append((plan_idx, result))
        # opt and workload to # of plots
        workload_to_num_plans = {
            "enron": 20,
            "real-estate": 20,
            "biofabric": 20,
        }

        # read results file(s) generated by evaluate_pz_plans
        results = {
            "enron": [],
            "real-estate": [],
            "biofabric": [],
        }
        for workload in ["enron", "real-estate", "biofabric"]:
            num_plans = workload_to_num_plans[workload]
            for plan_idx in range(num_plans):
                with open(
                    f"final-eval-results/{workload}/results-{plan_idx}.json", "r"
                ) as f:
                    result = json.load(f)
                    results[workload].append((plan_idx, result))

        plot_runtime_cost_vs_quality(results)

    if args.reopt:
        policy_to_plan = {
            "max-quality-at-fixed-cost": {
                "real-estate": "final-eval-results/reoptimization/real-estate/max-quality-at-fixed-cost.json",
            },
            "max-quality-at-fixed-runtime": {
                "real-estate": "final-eval-results/real-estate/results-0.json",
            },
            "min-cost-at-fixed-quality": {
                "real-estate": "final-eval-results/real-estate/results-3.json",
            },
        }

        policy_to_naive_plan = {
            "max-quality-at-fixed-cost": {"real-estate": 8},
            "max-quality-at-fixed-runtime": {"real-estate": 8},
            "min-cost-at-fixed-quality": {"real-estate": 8},
        }

        results = []
        for workload in ["real-estate"]:
            for policy in ["max-quality-at-fixed-cost", "max-quality-at-fixed-runtime", "min-cost-at-fixed-quality"]:
                fp = policy_to_plan[policy][workload]
                with open(fp, "r") as f:
                    result_dict = json.load(f)
                    results.extend([
                        {"plan": "PZ", "policy": policy, "workload": workload, "metric": "f1_score",
                         "value": result_dict["f1_score"]},
                        {"plan": "PZ", "policy": policy, "workload": workload, "metric": "cost",
                         "value": result_dict["cost"]},
                        {"plan": "PZ", "policy": policy, "workload": workload, "metric": "runtime",
                         "value": result_dict["runtime"]},
                    ])

                naive_plan_idx = policy_to_naive_plan[policy][workload]
                with open(f"final-eval-results/{workload}/results-{naive_plan_idx}.json", "r") as f:
                    result_dict = json.load(f)
                    results.extend([
                        {"plan": "Baseline", "policy": policy, "workload": workload, "metric": "f1_score",
                         "value": result_dict["f1_score"]},
                        {"plan": "Baseline", "policy": policy, "workload": workload, "metric": "cost",
                         "value": result_dict["cost"]},
                        {"plan": "Baseline", "policy": policy, "workload": workload, "metric": "runtime",
                         "value": result_dict["runtime"]},
                    ])

        plot_reopt(results, workload)