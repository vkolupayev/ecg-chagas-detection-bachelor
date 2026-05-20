import logging

import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from matplotlib import rcParams
from scipy.fftpack import fft

logger = logging.getLogger(__name__)

rcParams["axes.spines.right"] = False
rcParams["axes.spines.top"] = False
rcParams["axes.spines.left"] = False


def plot_wfdb_record(file_path, save_file_path=None, title="Normal PTB-XL Record"):
    import wfdb

    record = wfdb.rdrecord(file_path, physical=True)
    return_figure = True if save_file_path else False
    wfdb.plot_wfdb(record, figsize=(12, 10), title=title, return_fig=return_figure)
    if return_figure:
        plt.savefig(save_file_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_signal_channel(signal, title, sample_rate, figsize=(10, 3), clean_sig=None):
    time = np.linspace(0, len(signal) / sample_rate, len(signal))
    # Plot the channel data
    plt.figure(figsize=figsize)
    plt.plot(time, signal, c="black")
    plt.xlabel("Time, sec")
    plt.ylabel("Amplitude")
    plt.title(title)
    if clean_sig is not None:
        plt.plot(time, clean_sig, c="red")
    plt.show()


def plot_signal_channel_spectrum(signal, fs):
    n = len(signal)
    freqs = np.fft.fftfreq(n, 1 / fs)[: n // 2]
    fft_values = np.abs(fft(signal))[: n // 2]

    plt.figure(figsize=(10, 5))
    plt.plot(freqs, fft_values, color="b")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, fs // 2)  # Limit to Nyquist
    plt.grid()
    plt.show()


def plot_filter_response(
    freq,
    h,
    time,
    impulse_response,
    fs,
    title,
    file_path=None,
    focus_spectrum_to_hz=None,
):
    fig, ax = plt.subplots(4, 1, figsize=(12, 10), constrained_layout=True)
    # fig.tight_layout()
    title = title + "\nFrequency, Phase, Impulse Response"
    fig.suptitle(title, fontsize=16)
    with np.errstate(divide="ignore"):
        db = 20 * np.log10(abs(h))

    # ax[0].set_title("Frequency Response")
    ax[0].plot(freq, db, color="blue", label="Frequency Response")
    ax[0].set_ylabel("Amplitude [dB]", color="blue")
    if focus_spectrum_to_hz:
        ax[0].set_xlim([0, focus_spectrum_to_hz])
    else:
        ax[0].set_xlim([0, fs / 2])

    ax[1].plot(freq, abs(h), color="blue", label="Frequency Response")
    ax[1].set_ylabel("Amplitude", color="blue")
    if focus_spectrum_to_hz:
        ax[1].set_xlim([0, focus_spectrum_to_hz])
    else:
        ax[1].set_xlim([0, fs / 2])
    # ax[1].set_ylim([0, 1.2])
    # ax[0].set_ylim([-20, 10])

    # ax[2].set_title("Phase Response")
    ax[2].plot(
        freq, np.degrees(np.unwrap(np.angle(h))), color="green", label="Phase Response"
    )
    ax[2].set_ylabel("Phase [deg]", color="green")
    ax[2].set_xlabel("Frequency [Hz]")
    if focus_spectrum_to_hz:
        ax[2].set_xlim([0, focus_spectrum_to_hz])
    else:
        ax[2].set_xlim([0, fs / 2])

    # ax[3].set_title("Impulse Response")
    ax[3].stem(time, impulse_response, "r", label="Impulse Response")
    ax[3].set_ylabel("Amplitude", color="red")
    ax[3].set_xlabel("Time [s]")

    for axes in ax:
        axes.grid(True)
        axes.legend()

    if file_path:
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_signal_time_spectrum(
    signal,
    signal_time,
    fft_values,
    freqs,
    lead_names,
    title=None,
    focus_spectrum_to_hz=None,
    file_path=None,
):

    fig, ax = plt.subplots(
        12,
        2,
        figsize=(20, 30),
        constrained_layout=True,  # sharex="col"
    )
    if title:
        fig.suptitle(title, fontsize=16)
    ax[0][0].set_title("Signal")
    ax[0][1].set_title("Signal Spectrum")
    for i in range(fft_values.shape[1]):
        ax[i][0].plot(signal_time, signal[:, i], color="r")
        ax[i][0].grid(True)

        ax[i][1].plot(freqs, fft_values[:, i], color="b")
        if focus_spectrum_to_hz:
            ax[i][1].set_xlim([0, focus_spectrum_to_hz])
        ax[i][1].grid(True)

        ax[i][0].set_ylabel("Lead " + lead_names[i] + " amplitude")
        ax[i][1].set_ylabel("Lead " + lead_names[i] + " amplitude")

    ax[-1][1].set_xlabel("Frequency [Hz]")
    ax[-1][0].set_xlabel("Time [s]")
    if file_path:
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_signal_time_spectrum_overlapped(
    signals,
    signals_times,
    signals_fft_values,
    signals_freqs,
    signals_names,
    lead_names,
    title=None,
    focus_spectrum_to_hz=None,
    file_path=None,
):
    fig, ax = plt.subplots(
        12,
        2,
        figsize=(20, 30),
        constrained_layout=True,  # sharex="col"
    )
    if title:
        fig.suptitle(title, fontsize=16)
    ax[0][0].set_title("Signal")
    ax[0][1].set_title("Signal Spectrum")
    for i in range(12):

        for signal_time, signal, freqs, fft_values, signal_names in zip(
            signals_times, signals, signals_freqs, signals_fft_values, signals_names
        ):
            ax[i][0].plot(signal_time, signal[:, i], label=signal_names, alpha=0.7)
            ax[i][0].legend()
            ax[i][1].plot(freqs, fft_values[:, i], label=signal_names, alpha=0.7)
            ax[i][1].legend()

        if focus_spectrum_to_hz:
            ax[i][1].set_xlim([0, focus_spectrum_to_hz])

        ax[i][0].grid(True)
        ax[i][1].grid(True)

        ax[i][0].set_ylabel("Lead " + lead_names[i] + " amplitude")
        ax[i][1].set_ylabel("Lead " + lead_names[i] + " amplitude")

    ax[-1][1].set_xlabel("Frequency [Hz]")
    ax[-1][0].set_xlabel("Time [s]")
    if file_path:
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    cm, cm_normalized, threshold, classes, path_to_save, model_name
):
    plt.figure(figsize=(8, 8), layout="tight")
    cmap = plt.cm.Blues
    plt.imshow(cm_normalized, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
    plt.title(
        f"{model_name}\nConfusion Matrix (Threshold: {threshold:.3f})\n", fontsize=16
    )
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)
    for i, j in product(range(cm_normalized.shape[0]), range(cm_normalized.shape[1])):
        # format(cm_norm[i, j], '.2f')+'\n'+format(cm[i,j], 'd')
        plt.text(
            j,
            i,
            f"{cm_normalized[i, j]:.2f}\n{cm[i, j]:_}",
            horizontalalignment="center",
            color="black",
            fontsize=12,
        )

    plt.tight_layout()
    plt.ylabel("True label", fontsize=12, color="black")
    plt.xlabel("Predicted label", fontsize=12, color="black")
    np.set_printoptions(precision=2)
    plt.margins(0.2)
    image_path = path_to_save / "confusion_matrix.png"
    plt.savefig(image_path, bbox_inches="tight")
    # plt.show()
    plt.close()


def plot_roc_curve(fpr, tpr, roc_auc, path_to_save, model_name):
    logger.info(f"ROC AUC: {roc_auc:.4f}")

    # Plot ROC AUC
    plt.figure(figsize=(8, 8), layout="tight")
    plt.title(f"{model_name}\nReceiver Operating Characteristic\n", fontsize=18)
    plt.plot(fpr, tpr, "b", label="AUC = %0.4f" % roc_auc)
    plt.plot([0, 1], [0, 1], "r--", label="chance level (AUC = 0.5)")
    plt.legend(loc="lower right")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate", fontsize=14, color="black")
    plt.xlabel("False Positive Rate", fontsize=14, color="black")
    plt.margins(0.2)
    image_path = path_to_save / "roc_curve.png"
    plt.savefig(image_path, bbox_inches="tight")
    # plt.show()
    plt.close()


def plot_precision_recall_curve(
    precision, recall, pr_auc, baseline, path_to_save, model_name
):
    logger.info(f"Precision Recall Curve Area Under the Curve: {pr_auc:.4f}")

    # Plot ROC AUC
    plt.figure(figsize=(8, 8), layout="tight")
    plt.title(f"{model_name}\nPrecision Recall Curve\n", fontsize=18)
    plt.plot(recall, precision, label=f"AUC = {pr_auc:.4f}")
    plt.plot([0, 1], [baseline, baseline], linestyle="--", label="Baseline")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("Recall", fontsize=14, color="black")
    plt.ylabel("Precision", fontsize=14, color="black")
    plt.legend(loc="lower right")
    plt.margins(0.2)
    image_path = path_to_save / "precision_recall.png"
    plt.savefig(image_path, bbox_inches="tight")
    # plt.show()
    plt.close()


def plot_train_val_curves(
    train_folds_metrics,
    validatiation_folds_metrics,
    metric_name="Loss",
    title=None,
    file_path=None,
):

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True, sharey="col")
    if title:
        fig.suptitle(title, fontsize=16)
    ax[0].set_title("Train")
    ax[1].set_title("Validation")

    for i, (train_stats, validation_stats) in enumerate(
        zip(train_folds_metrics, validatiation_folds_metrics), 1
    ):
        ax[0].plot(train_stats, alpha=0.5, label=f"fold {i}")
        ax[1].plot(validation_stats, linestyle="--", alpha=0.5, label=f"fold {i}")

    for i in range(2):
        ax[i].set_xlabel("Epoch")
        ax[i].set_ylabel(metric_name)
        ax[i].legend()
        ax[i].grid(True)

    if file_path:
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_train_val_mean_std_curves(
    train_folds_metrics,
    validatiation_folds_metrics,
    metric_name="Loss",
    title=None,
    file_path=None,
):

    mean_train = np.nanmean(train_folds_metrics, axis=0)
    std_train = np.nanstd(train_folds_metrics, axis=0)

    mean_validation = np.nanmean(validatiation_folds_metrics, axis=0)
    std_validation = np.nanstd(validatiation_folds_metrics, axis=0)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True, sharey="col")
    if title:
        fig.suptitle(title, fontsize=16)
    ax[0].set_title("Train")
    ax[1].set_title("Validation")

    ax[0].plot(mean_train, label="Fold Mean")
    ax[0].fill_between(
        range(len(mean_train)),
        mean_train - std_train,
        mean_train + std_train,
        alpha=0.3,
    )
    ax[1].plot(mean_validation, label="Fold Mean")
    ax[1].fill_between(
        range(len(mean_validation)),
        mean_validation - std_validation,
        mean_validation + std_validation,
        alpha=0.3,
    )

    for i in range(2):
        ax[i].set_xlabel("Epoch")
        ax[i].set_ylabel(metric_name)
        ax[i].legend()
        ax[i].grid(True)

    if file_path:
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_folds_rocs(rocs, run_name, file_path=None):
    # Plot ROC AUC
    plt.figure(figsize=(8, 8), layout="tight")
    plt.title(
        f"Receiver Operating Characteristic of {run_name}\nROC AUC: {np.array(rocs["roc_auc"]).mean():.3f} (± {np.array(rocs["roc_auc"]).std():.3f})",
        fontsize=16,
    )
    for i, (fpr, tpr, roc_auc) in enumerate(
        zip(rocs["fpr"], rocs["tpr"], rocs["roc_auc"])
    ):
        plt.plot(
            fpr,
            tpr,
            alpha=0.9,
            label=f"Fold {i} AUC = {roc_auc:.3f}",
        )
    plt.plot([0, 1], [0, 1], "r--", label="chance level (AUC = 0.5)")
    plt.legend(loc="lower right")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate", fontsize=14, color="black")
    plt.xlabel("False Positive Rate", fontsize=14, color="black")
    plt.margins(0.2)

    if file_path:
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_folds_prcs(prcs, run_name, file_path=None):
    plt.figure(figsize=(8, 8), layout="tight")
    plt.title(
        f"Precision Recall Curve of {run_name}\nPRC AUC: {np.array(prcs["pr_auc"]).mean():.3f} (± {np.array(prcs["pr_auc"]).std():.3f})",
        fontsize=16,
    )
    for i, (recall, precision, pr_auc) in enumerate(
        zip(prcs["recall"], prcs["precision"], prcs["pr_auc"])
    ):
        plt.plot(recall, precision, alpha=0.7, label=f"Fold {i} AUC = {pr_auc:.3f}")

    plt.plot(
        [0, 1],
        [np.array(prcs["baseline"]).mean(), np.array(prcs["baseline"]).mean()],
        linestyle="--",
        label="Mean Baseline",
    )

    plt.fill_between(
        range(recall.shape[0]),
        np.array(prcs["baseline"]).mean() - np.array(prcs["baseline"]).std(),
        np.array(prcs["baseline"]).mean() + np.array(prcs["baseline"]).std(),
        alpha=0.3,
    )
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("Recall", fontsize=14, color="black")
    plt.ylabel("Precision", fontsize=14, color="black")
    plt.legend(loc="center right")
    plt.margins(0.2)
    if file_path:
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_folds_conf_matrices(conf_matrices, run_name, target_names, file_path=None):

    mean_cm, std_cm = np.array(conf_matrices["cm"]).mean(axis=0), np.array(
        conf_matrices["cm"]
    ).std(axis=0)
    mean_cm_norm, std_cm_norm = np.array(conf_matrices["cm_norm"]).mean(
        axis=0
    ), np.array(conf_matrices["cm_norm"]).std(axis=0)
    mean_threshold, std_threshold = (
        np.array(conf_matrices["thresholds"]).mean(),
        np.array(conf_matrices["thresholds"]).std(),
    )

    plt.figure(figsize=(8, 8), layout="tight")
    cmap = plt.cm.Blues
    plt.imshow(mean_cm_norm, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
    plt.title(
        f"{run_name} \nConfusion Matrix at Threshold {mean_threshold:.3f} (± {std_threshold:.3f})",
        fontsize=15,
    )
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45, fontsize=12)
    plt.yticks(tick_marks, target_names, fontsize=12)
    for i, j in product(range(mean_cm_norm.shape[0]), range(mean_cm_norm.shape[1])):
        # format(cm_norm[i, j], '.2f')+'\n'+format(cm[i,j], 'd')
        plt.text(
            j,
            i,
            f"{mean_cm_norm[i, j]:.2f} (± {std_cm_norm[i, j]:.2f})\n{mean_cm[i, j]:.0f} (± {std_cm[i, j]:.0f})",
            horizontalalignment="center",
            color="black",
            fontsize=12,
        )

    plt.tight_layout()
    plt.ylabel("True label", fontsize=12, color="black")
    plt.xlabel("Predicted label", fontsize=12, color="black")
    np.set_printoptions(precision=2)
    plt.margins(0.2)
    if file_path:
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_probability_distribution(
    folds_labels_list, folds_probs_list, run_name, file_path=None
):
    labels, probs = np.concatenate(folds_labels_list), np.concatenate(folds_probs_list)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True, sharey="col")
    fig.suptitle(run_name, fontsize=16)

    ax[0].set_title("Chagas Negative")
    ax[1].set_title("Chagas Positive")

    ax[0].hist(probs[labels == 0], bins=100, label="Chagas Negative")
    ax[1].hist(
        probs[labels == 1],
        bins=100,
        label="Chagas Positive",
        color="r",  # alpha=0.7,
    )

    for i in range(2):
        ax[i].set_xlabel("Probability of being positive class")
        ax[i].set_ylabel("Number of records in bucket")
        ax[i].set_xlim([0, 1])
        ax[i].grid(True)
    if file_path:
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_demographics_count(
    df_list, column="age_groups", sub_title_list="", title="", file_path=None
):

    plot_count = len(df_list)
    plot_cols = plot_count // 2
    plot_rows = plot_cols + plot_count % 2

    fig, axes = plt.subplots(
        plot_rows,
        plot_cols,
        figsize=(10, 5 * plot_cols),
        constrained_layout=True,
    )
    axes = axes.flatten()

    fig.suptitle(title, fontsize=16)

    for ax, df, sub_title in zip(axes, df_list, sub_title_list):
        counts = df[column].value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_xlabel(column.capitalize().replace("_", " "))
        ax.set_ylabel("Count")
        ax.set_title(sub_title)
        if len(counts.index) > 3:
            ax.tick_params(axis="x", labelrotation=45)

    if plot_count % 2 != 0:
        axes[plot_count].axis("off")
    if file_path:
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
