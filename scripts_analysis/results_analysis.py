from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

from src.utils import (
    load_run_train_val_metrics,
    load_run_evaluation_metrics,
    load_folds_data,
    get_train_val_metric,
    get_label_probs_subset,
    evaluation_metrics_at_5,
    evaluation_metrics_optimal_threshold,
    write_latex_table,
)
from src.evaluate import (
    find_per_fold_optimal_threshold,
    evaluate_confusion_matrix,
    evaluate_precision_recall_curve,
    evaluate_roc_curve,
)
from src.plot_utils import (
    plot_train_val_curves,
    plot_train_val_mean_std_curves,
    plot_folds_rocs,
    plot_folds_prcs,
    plot_folds_conf_matrices,
    plot_probability_distribution,
)

target_names = ["No Chagas", "Chagas"]
metrics = ["Loss", "Threshold", "Recall", "Precision", "F1-Score", "F2-Score"]#, "mcc"]
# metrics = ["Loss", "Threshold", "Recall", "Precision", "F2-Score"]

all_experiment_names = ["architecture_comparison", "architecture_modification", "data_augmentation"]
all_run_names = [
    [
    "hubert_linear_probing",
    "founder_linear_probing",
    "hubert_fine_tune",
    "founder_fine_tune",
    "hubert_train",
    "founder_train",
    "hubert_fine_tune_samiptb_recall",
    "founder_fine_tune_samiptb_recall",
    ],
    [
    "founder_fine_tune",
    "founder_fine_tune_bn",
    "founder_fine_tune_do_25",
    "founder_fine_tune_do_50",
    "founder_fine_tune_de",
    "founder_fine_tune_dlr",
    ],
    [
    "founder_fine_tune",
    "founder_fine_tune_tm",
    "founder_fine_tune_lm",
    "founder_fine_tune_ms",
    "founder_fine_tune_gn",
    "founder_fine_tune_ls",
    "founder_fine_tune_rls",
    ],
]

all_display_run_names = [
    [
        "HuBERT-ECG LP",
        "ECG Founder LP",
        "HuBERT-ECG FT",
        "ECG Founder FT",
        "HuBERT-ECG ST",
        "ECG Founder ST",
        "HuBERT-ECG FTS",
        "ECG Founder FTS",
    ],
    [
    "Baseline",
    "Batch Normalisation",
    "Dropout 25%",
    "Dropout 50%",
    "Demographic Encoder",
    "Discriminative Learning Rate",
    ],
    [
    "Baseline",
    "Temporal Masking",
    "Lead Masking",
    "Magnitude Scaling",
    "Gaussian Noise",
    "Label Smoothing",
    "Refined Label Smoothing",
    ],
]

subset_stats_recall = []
subset_stats_f2 = []
subset_stats_run_names = []
for experiment_name, run_names, display_run_names in zip(all_experiment_names, all_run_names, all_display_run_names):

    run_name_display_map = {
        run_name: display_run_name
        for run_name, display_run_name in zip(run_names, display_run_names)
    }

    table_convergence = pd.DataFrame()
    table_at_5 = pd.DataFrame()
    table_at_5_samiptb = pd.DataFrame()
    table_at_optimal = pd.DataFrame()
    table_at_optimal_code = pd.DataFrame()
    table_at_optimal_samiptb = pd.DataFrame()
    table_extra_metrics = pd.DataFrame()
    for run_name in run_names:

        run_save_path = Path(f"evaluations/{run_name}/aggregated/")
        run_save_path.mkdir(exist_ok=True)

        train_folds_stats, validation_folds_stats = load_run_train_val_metrics(run_name)
        evaluation_folds_stats = load_run_evaluation_metrics(run_name)

        train_loss_stats = get_train_val_metric(train_folds_stats, "Loss")
        validation_loss_stats = get_train_val_metric(validation_folds_stats, "Loss")
        train_recall_stats = get_train_val_metric(train_folds_stats, "Recall")
        validation_recall_stats = get_train_val_metric(validation_folds_stats, "Recall")

        convergence_epochs = np.nanargmax(validation_recall_stats, axis=1)
        run_convergence = pd.DataFrame(
            [
                {
                    f"Convergence epoch median (meadian absolute deviation)": \
                    f"{np.median(convergence_epochs):.1f} ({median_abs_deviation(convergence_epochs):.1f})"
                }
            ],
            index=[run_name_display_map[run_name]],
        )
        table_convergence = pd.concat([table_convergence, run_convergence])

        try:
            evaluation_folds_stats_transformed = {
                metric: np.array([stats[metric] for stats in evaluation_folds_stats])
                for metric in metrics
            }
        except KeyError as e:
            if e.args[0] == "F2-Score":
                refined_metrics = metrics.copy()
                refined_metrics.remove("F2-Score")
                evaluation_folds_stats_transformed = {
                    metric: np.array([stats[metric] for stats in evaluation_folds_stats])
                    for metric in refined_metrics
                }

                evaluation_folds_stats_transformed["F2-Score"] = (
                    (2**2 + 1)
                    * evaluation_folds_stats_transformed["Recall"]
                    * evaluation_folds_stats_transformed["Precision"]
                ) / (
                    evaluation_folds_stats_transformed["Recall"]
                    + 2**2 * evaluation_folds_stats_transformed["Precision"]
                )
            else:
                raise

        run_at_5 = pd.DataFrame(
            [
                {
                    f"{key}@5% μ (σ)": f"{item.mean():.3f} ({item.std():.3f})"
                    for key, item in evaluation_folds_stats_transformed.items()
                }
            ],
            index=[run_name_display_map[run_name]],
        )
        run_at_5.drop("Loss@5% μ (σ)", axis=1, inplace=True)
        table_at_5 = pd.concat([table_at_5, run_at_5])

        folds_evaluation_data_list, folds_labels_list, folds_probs_list = load_folds_data(
            run_name
        )
        fold_optimal_threshold = find_per_fold_optimal_threshold(
            folds_labels_list, folds_probs_list
        )

        metrics_at_optimal = evaluation_metrics_optimal_threshold(
            folds_labels_list, folds_probs_list, fold_optimal_threshold
        )
        run_at_optimal = pd.DataFrame(
            [
                {
                    f"{key} μ (σ)": f"{item.mean():.3f} ({item.std():.3f})"
                    for key, item in metrics_at_optimal.items()
                }
            ],
            index=[run_name_display_map[run_name]],
        )
        table_at_optimal = pd.concat([table_at_optimal, run_at_optimal])

        plot_train_val_curves(
            train_loss_stats,
            validation_loss_stats,
            "Loss",
            run_name_display_map[run_name],
            run_save_path / "loss_curves.png",
        )
        plot_train_val_curves(
            train_recall_stats,
            validation_recall_stats,
            "Recall",
            run_name_display_map[run_name],
            run_save_path / "recall_curves.png",
        )
        plot_train_val_mean_std_curves(
            train_loss_stats,
            validation_loss_stats,
            "Loss",
            run_name_display_map[run_name],
            run_save_path / "loss_mean_std_curves.png",
        )
        plot_train_val_mean_std_curves(
            train_recall_stats,
            validation_recall_stats,
            "Recall",
            run_name_display_map[run_name],
            run_save_path / "recall_mean_std_curves.png",
        )

        plot_probability_distribution(
            folds_labels_list,
            folds_probs_list,
            run_name_display_map[run_name],
            run_save_path / "prob_distribution.png",
        )

        conf_matrices_at_5 = {"cm": [], "cm_norm": [], "thresholds": evaluation_folds_stats_transformed["Threshold"]}
        conf_matrices_at_optimal = {"cm": [], "cm_norm": [], "thresholds": fold_optimal_threshold}
        prcs = {
            "precision": [],
            "recall": [],
            "thresholds": [],
            "pr_auc": [],
            "baseline": [],
        }
        rocs = {"fpr": [], "tpr": [], "thresholds": [], "roc_auc": []}

        for i, (probs, labels, optimal_threshold, threshold_at_5) in enumerate(
            zip(
                folds_probs_list,
                folds_labels_list,
                fold_optimal_threshold,
                evaluation_folds_stats_transformed["Threshold"],
            )
        ):
            cm, cm_normalized = evaluate_confusion_matrix(
                probs, labels, threshold=optimal_threshold
            )
            cm_5, cm_normalized_5 = evaluate_confusion_matrix(
                probs, labels, threshold=threshold_at_5
            )
            precision, recall, thresholds, pr_auc, baseline = (
                evaluate_precision_recall_curve(probs, labels)
            )
            fpr, tpr, thresholds, roc_auc = evaluate_roc_curve(probs, labels)

            conf_matrices_at_5["cm"].append(cm_5)
            conf_matrices_at_5["cm_norm"].append(cm_normalized_5)

            conf_matrices_at_optimal["cm"].append(cm)
            conf_matrices_at_optimal["cm_norm"].append(cm_normalized)

            prcs["precision"].append(precision)
            prcs["recall"].append(recall)
            prcs["thresholds"].append(thresholds)
            prcs["pr_auc"].append(pr_auc)
            prcs["baseline"].append(baseline)

            rocs["fpr"].append(fpr)
            rocs["tpr"].append(tpr)
            rocs["thresholds"].append(thresholds)
            rocs["roc_auc"].append(roc_auc)

        # aggregated Loss, ROC, PRC
        run_extras = {}
        run_extras["Loss"] = evaluation_folds_stats_transformed["Loss"]
        run_extras["ROC_AUC"] = np.array(rocs["roc_auc"])
        run_extras["PRC_AUC"] = np.array(prcs["pr_auc"])
        run_extras = pd.DataFrame(
            [
                {
                    f"{key.replace("_", " ")} μ (σ)": f"{item.mean():.3f} ({item.std():.3f})"
                    for key, item in run_extras.items()
                }
            ],
            index=[run_name_display_map[run_name]],
        )
        table_extra_metrics = pd.concat([table_extra_metrics, run_extras])

        # Plot roc, prc, conf_matrices
        plot_folds_rocs(rocs, run_name_display_map[run_name], run_save_path / "fold_rocs.png")
        plot_folds_prcs(prcs, run_name_display_map[run_name], run_save_path / "fold_prcs.png")
        plot_folds_conf_matrices(conf_matrices_at_optimal, run_name_display_map[run_name], target_names, run_save_path / "fold_cms_at_optimal.png")
        plot_folds_conf_matrices(conf_matrices_at_5, run_name_display_map[run_name], target_names, run_save_path / "fold_cms_at_5.png")

        if "samiptb" in run_name:
            continue
        
        samiptb_upsampled_folds_labels_list, samiptb_upsampled_folds_probs_list = get_label_probs_subset(
            folds_evaluation_data_list,
            folds_labels_list,
            folds_probs_list,
            sources=["SaMi-Trop", "PTB-XL"],
            upsample_negative=True,
        )

        code_folds_labels_list, code_folds_probs_list = get_label_probs_subset(
            folds_evaluation_data_list,
            folds_labels_list,
            folds_probs_list,
            sources=["CODE-15%"],
        )
        samiptb_folds_labels_list, samiptb_folds_probs_list = get_label_probs_subset(
            folds_evaluation_data_list,
            folds_labels_list,
            folds_probs_list,
            sources=["SaMi-Trop", "PTB-XL"],
        )

        
        metrics_at_5_samiptb = evaluation_metrics_at_5(
            samiptb_upsampled_folds_labels_list, samiptb_upsampled_folds_probs_list
        )
        
        # save for Stats tests
        subset_stats_recall.append(metrics_at_5_samiptb["Recall"])
        subset_stats_f2.append(metrics_at_5_samiptb["F2-Score"])
        subset_stats_run_names.append(run_name)

        run_at_5_samiptb = pd.DataFrame(
            [
                {
                    f"{key}@5% μ (σ)": f"{item.mean():.3f} ({item.std():.3f})"
                    for key, item in metrics_at_5_samiptb.items()
                }
            ],
            index=[run_name_display_map[run_name]],
        )
        table_at_5_samiptb = pd.concat([table_at_5_samiptb, run_at_5_samiptb])

        # ! subset optimal threshold
        # fold_optimal_threshold = find_per_fold_optimal_threshold(
        #     code_folds_labels_list, code_folds_probs_list
        # )
        code_metrics_at_optimal = evaluation_metrics_optimal_threshold(
            code_folds_labels_list, code_folds_probs_list, fold_optimal_threshold
        )
        run_at_optimal_code = pd.DataFrame(
            [
                {
                    f"{key} μ (σ)": f"{item.mean():.3f} ({item.std():.3f})"
                    for key, item in code_metrics_at_optimal.items()
                }
            ],
            index=[run_name_display_map[run_name]],
        )
        table_at_optimal_code = pd.concat([table_at_optimal_code, run_at_optimal_code])

        # ! subset optimal threshold
        # fold_optimal_threshold = find_per_fold_optimal_threshold(
        #     samiptb_folds_labels_list, samiptb_folds_probs_list
        # )
        samiptb_metrics_at_optimal = evaluation_metrics_optimal_threshold(
            samiptb_folds_labels_list, samiptb_folds_probs_list, fold_optimal_threshold
        )
        run_at_optimal_samiptb = pd.DataFrame(
            [
                {
                    f"{key} μ (σ)": f"{item.mean():.3f} ({item.std():.3f})"
                    for key, item in samiptb_metrics_at_optimal.items()
                }
            ],
            index=[run_name_display_map[run_name]],
        )
        table_at_optimal_samiptb = pd.concat(
            [table_at_optimal_samiptb, run_at_optimal_samiptb]
        )

    extra_table_at_5 = table_at_5.filter(regex = "Precision|F1-Score|Threshold")
    table_at_5 = table_at_5.filter(regex = "Recall|F2-Score|Threshold")

    extra_table_at_5_samiptb = table_at_5_samiptb.filter(regex = "Precision|F1-Score|Threshold")
    table_at_5_samiptb = table_at_5_samiptb.filter(regex = "Recall|F2-Score|Threshold")

    write_latex_table(
        table_at_5,
        rf"{experiment_name.capitalize().replace("_", " ")} experiment results at 5% threshold",
        f"tab:{experiment_name}_metrics_at_5",
        f"evaluations/aggregated_{experiment_name}/",
        "table_at_5.tex",
    )

    write_latex_table(
        extra_table_at_5,
        rf"{experiment_name.capitalize().replace("_", " ")} experiment additional results at 5% threshold",
        f"tab:{experiment_name}_extra_metrics_at_5",
        f"evaluations/aggregated_{experiment_name}/",
        "extra_table_at_5.tex",
    )
    write_latex_table(
        table_at_5_samiptb,
        rf"{experiment_name.capitalize().replace("_", " ")} experiment results at 5% threshold on SaMi-Trop and upsampled PTB-XL",
        f"tab:{experiment_name}_metrics_at_5_samiptb",
        f"evaluations/aggregated_{experiment_name}/",
        "table_at_5_samiptb.tex",
    )
    write_latex_table(
        extra_table_at_5_samiptb,
        rf"{experiment_name.capitalize().replace("_", " ")} experiment additional results at 5% threshold on SaMi-Trop and upsampled PTB-XL",
        f"tab:{experiment_name}_extra_metrics_at_5_samiptb",
        f"evaluations/aggregated_{experiment_name}/",
        "extra_table_at_5_samiptb.tex",
    )

    write_latex_table(
        table_extra_metrics,
        rf"{experiment_name.capitalize().replace("_", " ")} experiment additional metrics",
        f"tab:{experiment_name}_extra_metrics",
        f"evaluations/aggregated_{experiment_name}/",
        "table_extra_metrics.tex",
    )

    write_latex_table(
        table_at_optimal,
        rf"{experiment_name.capitalize().replace("_", " ")} experiment results at optimal threshold by F2-score",
        f"tab:{experiment_name}_metrics_at_optimal",
        f"evaluations/aggregated_{experiment_name}/",
        "table_at_optimal.tex",
    )
    write_latex_table(
        table_at_optimal_code,
        rf"{experiment_name.capitalize().replace("_", " ")} experiment results at optimal threshold by F2-score on Code-15%",
        f"tab:{experiment_name}_metrics_at_optimal_code",
        f"evaluations/aggregated_{experiment_name}/",
        "table_at_optimal_code.tex",
        # "table_at_code_optimal_code.tex",
    )
    write_latex_table(
        table_at_optimal_samiptb,
        rf"{experiment_name.capitalize().replace("_", " ")} experiment results at optimal threshold by F2-score on SaMi-Trop and PTB-XL",
        f"tab:{experiment_name}_metrics_at_optimal_samiptb",
        f"evaluations/aggregated_{experiment_name}/",
        "table_at_optimal_samiptb.tex",
        # "table_at_samiptb_optimal_samiptb.tex",
    )
    write_latex_table(
        table_convergence,
        rf"{experiment_name.capitalize().replace("_", " ")} experiment convergence epoch results",
        f"tab:{experiment_name}_convergence",
        f"evaluations/aggregated_{experiment_name}/",
        "table_convergence.tex",
    )

# DFs for stats tests
pd.DataFrame(np.array(subset_stats_recall).T, columns=subset_stats_run_names).sort_index(axis=1).to_csv("evaluations/experiments_subset_recalls.csv", index=False)
pd.DataFrame(np.array(subset_stats_f2).T, columns=subset_stats_run_names).sort_index(axis=1).to_csv("evaluations/experiments_subset_f2s.csv", index=False)