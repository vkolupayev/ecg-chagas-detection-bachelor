import logging

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    matthews_corrcoef,
    fbeta_score,
)

logger = logging.getLogger(__name__)


# Coppied from helper_code
def compute_challenge_score(
    labels, outputs, fraction_capacity=0.05, num_permutations=10**4, seed=12345
):
    # Check the data.
    assert len(labels) == len(outputs)
    num_instances = len(labels)
    capacity = int(fraction_capacity * num_instances)

    # Convert the data to NumPy arrays, as needed, for easier indexing.
    labels = np.asarray(labels, dtype=np.float64)
    outputs = np.asarray(outputs, dtype=np.float64)

    # Permute the labels and outputs so that we can approximate the expected confusion matrix for "tied" probabilities.
    tp = np.zeros(num_permutations)
    fp = np.zeros(num_permutations)
    fn = np.zeros(num_permutations)
    tn = np.zeros(num_permutations)

    if seed is not None:
        np.random.seed(seed)

    for i in range(num_permutations):
        permuted_idx = np.random.permutation(np.arange(num_instances))
        permuted_labels = labels[permuted_idx]
        permuted_outputs = outputs[permuted_idx]

        ordered_idx = np.argsort(permuted_outputs, stable=True)[::-1]
        ordered_labels = permuted_labels[ordered_idx]

        tp[i] = np.sum(ordered_labels[:capacity] == 1)
        fp[i] = np.sum(ordered_labels[:capacity] == 0)
        fn[i] = np.sum(ordered_labels[capacity:] == 1)
        tn[i] = np.sum(ordered_labels[capacity:] == 0)

    tp = np.mean(tp)
    fp = np.mean(fp)
    fn = np.mean(fn)
    tn = np.mean(tn)

    threshold = outputs[capacity]

    # Compute the true positive rate.
    if tp + fn > 0:
        tpr = tp / (tp + fn)
    else:
        tpr = float("nan")

    return tpr, threshold, tp, fp, fn, tn


# Coppied from helper_code
def compute_old_challenge_score(labels, outputs, max_fraction_positive=0.05):
    # Check the data.
    assert len(labels) == len(outputs)
    num_instances = len(labels)
    max_num_positive_instances = int(max_fraction_positive * num_instances)

    # Convert the data to NumPy arrays, as needed, for easier indexing.
    labels = np.asarray(labels, dtype=np.float64)
    outputs = np.asarray(outputs, dtype=np.float64)

    # Collect the unique output values as the thresholds for the positive and negative classes.
    thresholds = np.unique(outputs)
    thresholds = np.append(thresholds, thresholds[-1] + 1)
    thresholds = thresholds[::-1]
    num_thresholds = len(thresholds)

    idx = np.argsort(outputs)[::-1]

    # Initialize the TPs, FPs, FNs, and TNs with no positive outputs.
    tp = np.zeros(num_thresholds)
    fp = np.zeros(num_thresholds)
    fn = np.zeros(num_thresholds)
    tn = np.zeros(num_thresholds)

    tp[0] = 0
    fp[0] = 0
    fn[0] = np.sum(labels == 1)
    tn[0] = np.sum(labels == 0)

    # Update the TPs, FPs, FNs, and TNs using the values at the previous threshold.
    i = 0
    for j in range(1, num_thresholds):
        tp[j] = tp[j - 1]
        fp[j] = fp[j - 1]
        fn[j] = fn[j - 1]
        tn[j] = tn[j - 1]

        while i < num_instances and outputs[idx[i]] >= thresholds[j]:
            if labels[idx[i]] == 1:
                tp[j] += 1
                fn[j] -= 1
            else:
                fp[j] += 1
                tn[j] -= 1
            i += 1

    # Find the true positive rate so that the number of positive model outputs are no more than 5% of the total instances.
    k = num_thresholds
    for j in range(1, num_thresholds):
        if tp[j] + fp[j] > max_num_positive_instances:
            k = j - 1
            break

    if tp[k] + fn[k] > 0:
        tpr = tp[k] / (tp[k] + fn[k])
    else:
        tpr = float("nan")

    return tpr, thresholds[k], tp[k], fp[k], fn[k], tn[k]


def compute_metrics_epoch(labels, probs, num_permutations=10**4):
    recall_k, threshold, tp_k, fp_k, fn_k, tn_k = compute_challenge_score(
        labels, probs, num_permutations=num_permutations
    )

    precision_k = tp_k / (tp_k + fp_k) if tp_k + fp_k > 0 else float("nan")
    f1_k = 2 * (recall_k * precision_k) / (recall_k + precision_k)
    f2_k = ((2**2 + 1) * recall_k * precision_k) / (recall_k + 2**2 * precision_k)

    metrics_dict = {
        "Threshold": threshold,
        "Recall": recall_k,
        "Precision": precision_k,
        "F1-Score": f1_k,
        "F2-Score": f2_k,
        "TP": int(tp_k),
        "FP": int(fp_k),
        "FN": int(fn_k),
        "TN": int(tn_k),
    }

    return metrics_dict


# Redundant calculations using sklearn, but fine for now
def evaluate_roc_curve(probs, y_true):
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, thresholds, roc_auc


def evaluate_precision_recall_curve(probs, y_true):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    pr_auc = auc(recall, precision)
    baseline = y_true[y_true == 1].shape[0] / y_true.shape[0]

    return precision, recall, thresholds, pr_auc, baseline


def evaluate_confusion_matrix(probs, y_true, threshold):
    y_preds = probs >= threshold
    cm_normalized = confusion_matrix(y_true, y_preds, normalize="true")
    cm = confusion_matrix(y_true, y_preds)
    return cm, cm_normalized


def evaluate_classification_report(
    probs, y_true, threshold, target_names=["No Chagas", "Chagas"]
):
    y_preds = probs >= threshold
    report = classification_report(
        y_true, y_preds, target_names=target_names, output_dict=True
    )
    mcc = matthews_corrcoef(y_true, y_preds)
    logger.info(
        f"\n{classification_report(y_true, y_preds, target_names=target_names)}"
    )
    logger.info(f"Matthews Correlation Coeficient: {mcc:5f}")
    logger.info(f"Threshold: {threshold:.5f}")
    return report, mcc


def find_optimal_threshold(folds_labels_list, folds_probs_list):
    labels = (
        np.concatenate(folds_labels_list)
        if isinstance(folds_labels_list, list)
        else folds_labels_list
    )
    probs = (
        np.concatenate(folds_probs_list)
        if isinstance(folds_probs_list, list)
        else folds_probs_list
    )

    thresholds = np.arange(0.01, 1, 0.001)
    f_scores = []
    for threshold in thresholds:
        f_scores.append(fbeta_score(labels, probs >= threshold, beta=2))
    f_scores = np.array(f_scores)
    # print(f"Optimal Threshold: {thresholds[f_scores.argmax()]:.4f}\tF2-Score: {f_scores.max():.4f}")
    return thresholds[f_scores.argmax()]


def find_per_fold_optimal_threshold(folds_labels_list, folds_probs_list):
    thresholds = np.arange(0.01, 1, 0.001)
    optimal_thresholds = []
    for labels, probs in zip(folds_labels_list, folds_probs_list):
        f_scores = []
        for threshold in thresholds:
            f_scores.append(fbeta_score(labels, probs >= threshold, beta=2))
        f_scores = np.array(f_scores)
        optimal_thresholds.append(thresholds[f_scores.argmax()])
        # print(f"Optimal Threshold: {thresholds[f_scores.argmax()]:.4f}\tF2-Score: {f_scores.max():.4f}")

    return np.array(optimal_thresholds)


def evaluate_at_threshold(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)
    labels = labels.astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()  # .tolist()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1_score = 2 * precision * recall / (precision + recall + 1e-9)
    f2_score = ((2**2 + 1) * recall * precision) / (recall + 2**2 * precision)
    # to avoid mcc overflow
    tn = tn.astype(np.float64)
    fp = fp.astype(np.float64)
    fn = fn.astype(np.float64)
    tp = tp.astype(np.float64)
    mcc = (tp * tn - fp * fn) / (
        np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-9
    )
    return precision, recall, f1_score, f2_score, mcc
