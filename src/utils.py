import re
import json
from typing import List
from pathlib import Path
from itertools import zip_longest

import numpy as np
import pandas as pd

from scipy.signal import freqz, lfilter
from scipy.fftpack import fft

from src.evaluate import evaluate_at_threshold, compute_metrics_epoch
from src.constants import RANDOM_SEED


def generate_filter_response(b, a, fs):
    """Usage example:
    # Filter parameters
    filter_order = 4
    bandpass_range = [0.5, 50]
    fs = 400.0

    # Design filter
    b, a = butter(N=filter_order, Wn=bandpass_range, btype="bandpass", fs=fs)
    generate_filter_response(b, a, fs)
    Args:
        b (_type_): _description_
        a (_type_): _description_
        fs (_type_): _description_
    """

    # frequency phase response
    freq, h = freqz(b, a, fs=fs)

    # Impulse response
    impulse = np.zeros(int(fs))
    impulse[0] = 1
    impulse_response = lfilter(b, a, impulse)
    impulse_time = np.arange(int(fs)) / fs

    return freq, h, impulse_time, impulse_response


def generate_signal_spectrum(signal, fs):
    n = len(signal)
    freqs = np.fft.fftfreq(n, 1 / fs)[: n // 2]
    fft_values = np.abs(fft(signal, axis=0))[: n // 2]
    return freqs, fft_values


def generate_signal_time(signal, fs):
    return np.linspace(0, signal.shape[0] / fs, num=signal.shape[0], endpoint=True)


# Results Analysis


def load_run_evaluation_metrics(run_name):
    evaluation_metrics_paths = sorted(
        [
            metric_path
            for metric_path in Path(f"evaluations/{run_name}").glob(
                "*/evaluation_metrics.json"
            )
        ],
        key=(lambda p: int(p.parent.__str__().split("_")[-1])),
    )
    evaluation_folds_stats = []
    for metrics_path in evaluation_metrics_paths:
        with open(metrics_path, "r") as f:
            evaluation_metrics = json.load(f)
        evaluation_folds_stats.append(evaluation_metrics)

    return evaluation_folds_stats


def load_run_train_val_metrics(run_name):
    training_metrics_paths = sorted(
        [
            metric_path
            for metric_path in Path(f"evaluations/{run_name}").glob(
                "*/train_metrics.json"
            )
        ],
        key=(lambda p: int(p.parent.__str__().split("_")[-1])),
    )

    validation_metrics_paths = sorted(
        [
            metric_path
            for metric_path in Path(f"evaluations/{run_name}").glob(
                "*/validation_metrics.json"
            )
        ],
        key=(lambda p: int(p.parent.__str__().split("_")[-1])),
    )
    train_folds_stats = []
    for metrics_path in training_metrics_paths:
        with open(metrics_path, "r") as f:
            train_metrics = json.load(f)
        result = {key: [d[key] for d in train_metrics] for key in train_metrics[0]}
        train_folds_stats.append(result)

    validation_folds_stats = []
    for metrics_path in validation_metrics_paths:
        with open(metrics_path, "r") as f:
            validation_metrics = json.load(f)
        result = {
            key: [d[key] for d in validation_metrics] for key in validation_metrics[0]
        }
        validation_folds_stats.append(result)
    return train_folds_stats, validation_folds_stats


def load_folds_data(run_name):
    run_fold_paths = sorted(
        [metric_path for metric_path in Path(f"evaluations/{run_name}").glob("fold_*")],
        key=(lambda p: int(p.__str__().split("_")[-1])),
    )

    all_labels = []
    all_probs = []
    all_evaluation_data = []
    for rf_path in run_fold_paths:
        with open(rf_path / "val_labels.npy", "rb") as f:
            labels = np.load(f)
        with open(rf_path / "val_probs.npy", "rb") as f:
            probs = np.load(f)
        evaluation_data = pd.read_csv(rf_path / "evaluation_data.csv")
        all_labels.append(labels)
        all_probs.append(probs)
        all_evaluation_data.append(evaluation_data)
    return all_evaluation_data, all_labels, all_probs


def get_train_val_metric(folds_stats, metric_name: str):
    stats = [i[metric_name] for i in folds_stats]
    stats = np.array(list(zip_longest(*stats, fillvalue=float("NaN")))).T
    return stats


def evaluation_metrics_at_5(folds_labels_list, folds_probs_list):
    metrics = {
        "Threshold": [],
        "Recall": [],
        "Precision": [],
        "F1-Score": [],
        "F2-Score": [],
    }

    for labels, probs in zip(folds_labels_list, folds_probs_list):
        fold_metrics = compute_metrics_epoch(labels, probs, num_permutations=10**3)

        metrics["Threshold"].append(fold_metrics["Threshold"])
        metrics["Precision"].append(fold_metrics["Precision"])
        metrics["Recall"].append(fold_metrics["Recall"])
        metrics["F1-Score"].append(fold_metrics["F1-Score"])
        metrics["F2-Score"].append(fold_metrics["F2-Score"])

    metrics["Threshold"] = np.array(metrics["Threshold"])
    metrics["Precision"] = np.array(metrics["Precision"])
    metrics["Recall"] = np.array(metrics["Recall"])
    metrics["F1-Score"] = np.array(metrics["F1-Score"])
    metrics["F2-Score"] = np.array(metrics["F2-Score"])

    return metrics


def evaluation_metrics_optimal_threshold(
    folds_labels_list, folds_probs_list, thresholds
):
    metrics = {
        "Threshold": thresholds,
        "Recall": [],
        "Precision": [],
        "F1-Score": [],
        "F2-Score": [],
        "MCC": [],
    }

    for labels, probs, threshold in zip(
        folds_labels_list, folds_probs_list, thresholds
    ):
        precision, recall, f1_score, f2_score, mcc = evaluate_at_threshold(
            labels, probs, threshold
        )

        metrics["Precision"].append(precision)
        metrics["Recall"].append(recall)
        metrics["F1-Score"].append(f1_score)
        metrics["F2-Score"].append(f2_score)
        metrics["MCC"].append(mcc)

    metrics["Precision"] = np.array(metrics["Precision"])
    metrics["Recall"] = np.array(metrics["Recall"])
    metrics["F1-Score"] = np.array(metrics["F1-Score"])
    metrics["F2-Score"] = np.array(metrics["F2-Score"])
    metrics["MCC"] = np.array(metrics["MCC"])

    return metrics


def get_label_probs_subset(
    folds_evaluation_data_list,
    folds_labels_list,
    folds_probs_list,
    sources: List[str],
    upsample_negative=False,
):
    subset_folds_labels_list = []
    subset_folds_probs_list = []
    for data, labels, probs in zip(
        folds_evaluation_data_list, folds_labels_list, folds_probs_list
    ):
        subset_data = data[data["source"].isin(sources)]
        if upsample_negative:
            subset_data = pd.concat(
                (
                    subset_data,
                    subset_data[subset_data["chagas_label"] == False].sample(
                        frac=1.7, replace=True, random_state=RANDOM_SEED
                    ),
                ),
                ignore_index=False,
            )

        indices = subset_data.index.to_numpy()
        subset_folds_labels_list.append(labels[indices])
        subset_folds_probs_list.append(probs[indices])
    return subset_folds_labels_list, subset_folds_probs_list


def write_latex_table(table, caption, label, save_path, file_name):
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)

    rows, columns = table.shape

    table = (
        table.style.to_latex(
            column_format="c" + "|c" * (len(table.columns)),
            position="!ht",
            hrules=False,
            position_float="centering",
            caption=caption,
        )
        .replace("%", r"\%")
        .replace(r"\end{tabular}", r"\end{TAB}")
        .replace(
            r"\end{TAB}",
            f"\\end{{TAB}}\n\\label{{{label}}}",
        )
    )
    pattern = r"\\begin\{tabular\}\{.*?\}"
    col_pattern = "c" + "|c" * columns
    row_pattern = "c" + "|c" * rows
    template = rf"\\begin{{TAB}}(r,0.8cm,0.8cm)[4pt]{{{col_pattern}}}{{{row_pattern}}}"
    table = re.sub(pattern, template, table)

    with open(save_path / file_name, "w") as f:
        f.write(table)
