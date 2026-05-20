import os
import logging
from typing import Any

from src.constants import RANDOM_SEED, seed_everything
from src.data import CleanLevel, ChagasDataset
from src.model.ecg_founder import Net1D
from src.model.hubert import ChagasHuBERT
from src.loss import BinaryFocalLoss
from src.train import (
    full_train,
    validate_epoch,
    log_epoch_metrics,
    compute_metrics_epoch,
)

from src.evaluate import (
    evaluate_classification_report,
    evaluate_confusion_matrix,
    evaluate_precision_recall_curve,
    evaluate_roc_curve,
)
from src.plot_utils import (
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from src.train_utils import EarlyStopping, SaveModel

# from src.meta_data import prep_exams_data, prepare_dataset_metrics, upsample_samitrop
from src.meta_data import add_refined_label_smooth, add_demographic_dummies
import json
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


def _set_dlr(experiment_parameters, model):
    optimizer = AdamW(
        [
            {
                "params": list(model.parameters())[:-2],
                "lr": torch.tensor(experiment_parameters["lr"] / 10),
            },
            {"params": list(model.parameters())[-2:]},
        ],
        lr=torch.tensor(experiment_parameters["lr"]),
        weight_decay=torch.tensor(experiment_parameters["weight_decay"]),
    )
    return optimizer


def _load_model(experiment_parameters: dict[str, Any], demo_feat_len=None):
    if experiment_parameters["model"] == "ecg_founder":
        model = Net1D(
            in_channels=12,
            base_filters=64,  # 32 64
            ratio=1,
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
            kernel_size=16,
            stride=2,
            groups_width=16,
            verbose=False,
            use_bn=experiment_parameters["use_bn"],
            use_do=experiment_parameters["use_do"],
            do_prob=(
                experiment_parameters["do_prob"]
                if experiment_parameters["use_do"]
                else 0.0
            ),
            use_demo_encoder=experiment_parameters["use_de"],
            demo_feat_len=demo_feat_len,
            n_classes=1,
        )
        if experiment_parameters["train_mode"] != "ST":
            state_dict = torch.load(
                "pretrained_models/12_lead_ECGFounder.pth", weights_only=False
            )["state_dict"]
            # partial load
            state_dict = {
                k: v for k, v in state_dict.items() if not k.startswith("dense.")
            }
            model.load_state_dict(state_dict, strict=False)
    else:
        model = ChagasHuBERT(
            simple_classifier=True,
            pretrained=True if experiment_parameters["train_mode"] != "ST" else False,
            model_path="pretrained_models/hubert-ecg-small",
        )
    if experiment_parameters["train_mode"] == "LP":

        if experiment_parameters["model"] == "ecg_founder":
            # Train only Head (Linear Probing)
            for param in model.parameters():
                param.requires_grad = False
            for param in model.dense.parameters():
                param.requires_grad = True
        if experiment_parameters["model"] == "hubert_ecg":
            for param in model.hubert_ecg.parameters():
                param.requires_grad = False

    return model


def run_experiment(experiment_parameters: dict[str, Any]) -> None:
    logger = logging.getLogger(__name__)
    seed_everything(RANDOM_SEED)
    logger.info("Start Training")
    clean_level = CleanLevel.REFINED

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Experiment Parameters:")
    for key, value in experiment_parameters.items():
        logger.info(f"{key}: {value}")

    run_name = experiment_parameters["run_name"]
    data_path = Path("data/training_data/")
    meta_data_path = data_path / "meta_data"
    evaluation_path = Path(f"evaluations/{run_name}/")
    model_path = Path(f"models/{run_name}/")
    logger.info(f"Models will be saved to {model_path}")
    logger.info(f"Evaluations will be saved to {evaluation_path}")

    logger.info("Read Meta Data")

    meta_data = pd.read_csv(
        meta_data_path / "meta_data.csv",
        dtype={"record_name": str, "base_time": str, "base_date": str},
    )

    # load exam info
    logger.info("Loading exam info")
    db_exams = pd.read_csv(meta_data_path / "db_exams.csv", low_memory=False)

    logger.info("Merging meta and exams data")
    meta_data = pd.merge(
        meta_data,
        db_exams,
        how="left",
        left_on="record_name",
        right_on="exam_id",
        validate="1:1",
    )

    relevant_cols = [
        "record_name",
        "patient_id",
        "data_path",
        "fs",
        "source",
        "chagas_label",
        "age",
        "sex",
        "RBBB",
        "1dAVb",
        "AF",
    ]
    if experiment_parameters["rls"]:
        logger.info("Adding refined label smooth")
        meta_data = add_refined_label_smooth(meta_data)
        relevant_cols.append("soft_label")

    metrics = pd.read_csv(meta_data_path / "channel_metrics.csv").drop("source", axis=1)
    meta_data = meta_data.merge(metrics, on="data_path")

    logger.info("Outlier removal")

    meta_data = meta_data[meta_data.sig_len >= 1500]

    # Attention mask solves missing channels
    # meta_data = meta_data.loc[~(meta_data.filter(regex="std|pp_amp") == 0).any(axis=1)]

    Q1 = meta_data.filter(regex="pp_amp").quantile(0.25)
    Q3 = meta_data.filter(regex="pp_amp").quantile(0.75)
    IQR = Q3 - Q1

    for numeric_col in meta_data.filter(regex="pp_amp").columns:
        meta_data = meta_data.loc[
            (meta_data[numeric_col] >= Q1[numeric_col] - 1.5 * IQR[numeric_col])
            & (meta_data[numeric_col] <= Q3[numeric_col] + 1.5 * IQR[numeric_col])
        ]

    logger.info("Select only relevant columns")
    meta_data = meta_data.loc[:, relevant_cols]

    if experiment_parameters["train_mode"] == "FTS":
        meta_data = meta_data.loc[meta_data["source"].isin(["PTB-XL", "SaMi-Trop"])]

    dummies_len = None
    if experiment_parameters["use_de"]:
        logger.info("Constructing demographic dummies and merging to meta data")
        meta_data, dummies_len = add_demographic_dummies(meta_data)

    if experiment_parameters["test_run"]:
        logger.info("Test run 10% sample")
        meta_data = meta_data.sample(frac=0.1)

    loss = BinaryFocalLoss(
        pos_weight=experiment_parameters["loss_pos_weight"],
        alpha=experiment_parameters["loss_alpha"],
        gamma=experiment_parameters["loss_gamma"],
        reduction="sum",
    )
    loss.to(device)

    fs = 500 if experiment_parameters["model"] == "ecg_founder" else 100
    max_sig_len = fs * 10

    logger.info("Create Data Folds")
    str_grp_k_fold = StratifiedGroupKFold(
        n_splits=10, shuffle=True, random_state=RANDOM_SEED
    )

    for i, split in enumerate(
        str_grp_k_fold.split(
            meta_data, meta_data["chagas_label"], meta_data["patient_id"]
        ),
        start=1,
    ):
        fold_name = f"fold_{i}"
        fold_evaluation_path = evaluation_path / fold_name
        fold_model_name = f"best_model_weights_{fold_name}"

        logger.info(f"{i} Fold")

        train_dataset = meta_data.iloc[split[0]]
        val_dataset = meta_data.iloc[split[1]]

        if experiment_parameters["train_mode"] == "FTS":
            logger.info(
                "Upsample PTB-XL to retain a similar label proportion to the full dataset"
            )
            train_dataset = pd.concat(
                (
                    train_dataset,
                    train_dataset[train_dataset["chagas_label"] == False].sample(
                        frac=1.7, replace=True, random_state=RANDOM_SEED
                    ),
                ),
                ignore_index=True,
            )
            val_dataset = pd.concat(
                (
                    val_dataset,
                    val_dataset[val_dataset["chagas_label"] == False].sample(
                        frac=1.7, replace=True, random_state=RANDOM_SEED
                    ),
                ),
                ignore_index=True,
            )

        train_dataset = ChagasDataset(
            train_dataset,
            fs=fs,
            max_sig_len=max_sig_len,
            clean_level=clean_level,
            code_soft_label_positive_term=0.8 if experiment_parameters["ls"] else 1.0,
            refined_label_smooth_flag=experiment_parameters["rls"],
            normalize_flag=(
                True if experiment_parameters["model"] == "ecg_founder" else False
            ),
            scaling_flag=(
                True if experiment_parameters["model"] == "hubert_ecg" else False
            ),
            to_tensor_flag=True,
            attention_mask_flag=(
                True if experiment_parameters["model"] == "hubert_ecg" else False
            ),
            z_mean=None,
            z_std=None,
            random_crop_probability=0.0,
            max_sig_seconds=None,
            random_temporal_mask_prob=0.5 if experiment_parameters["tm_aug"] else 0.0,
            random_lead_mask_prob=0.5 if experiment_parameters["lm_aug"] else 0.0,
            random_magnitude_scale_prob=0.5 if experiment_parameters["ms_aug"] else 0.0,
            random_gaussian_noise_prob=0.5 if experiment_parameters["gn_aug"] else 0.0,
            return_demo_feats=experiment_parameters["use_de"],
            zero_demo_feat_prob=0.3 if experiment_parameters["use_de"] else 0.0,
            verbose=False,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=experiment_parameters["bs"],
            shuffle=True,
            num_workers=6,
            prefetch_factor=2,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )
        val_dataset = ChagasDataset(
            val_dataset,
            fs=fs,
            max_sig_len=max_sig_len,
            clean_level=clean_level,
            code_soft_label_positive_term=1.0,
            normalize_flag=(
                True if experiment_parameters["model"] == "ecg_founder" else False
            ),
            scaling_flag=(
                True if experiment_parameters["model"] == "hubert_ecg" else False
            ),
            to_tensor_flag=True,
            attention_mask_flag=(
                True if experiment_parameters["model"] == "hubert_ecg" else False
            ),
            z_mean=None,
            z_std=None,
            random_crop_probability=0.0,
            max_sig_seconds=None,
            return_demo_feats=experiment_parameters["use_de"],
            zero_demo_feat_prob=0.0,
            verbose=False,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=experiment_parameters["bs"],
            shuffle=False,
            num_workers=6,
            prefetch_factor=2,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )

        logger.info("Initialize Model, Hyper-Parameters, Optimizer")

        model = _load_model(experiment_parameters, dummies_len)
        logger.info(
            f"Full Model param count: {sum(p.numel() for p in model.parameters()):_}"
        )

        model.to(device)

        early_stopping = EarlyStopping(
            patience=experiment_parameters["early_stop_patience"], maximize=True
        )
        save_model = SaveModel(model_path, model_name=fold_model_name)

        if experiment_parameters["use_dlr"]:
            optimizer = _set_dlr(experiment_parameters, model)
        else:
            optimizer = AdamW(
                model.parameters(),
                lr=torch.tensor(experiment_parameters["lr"]),
                weight_decay=torch.tensor(experiment_parameters["weight_decay"]),
            )
        scheduler = ReduceLROnPlateau(
            optimizer,
            patience=experiment_parameters["early_stop_patience"] // 2,
            factor=0.1,
            min_lr=experiment_parameters["lr"] / 100,
            cooldown=1,
            mode="max",
        )

        logger.info(f"Training {experiment_parameters["display_run_name"]} Fold {i}")
        model, train_metrics, val_metrics, best_epoch = full_train(
            model=model,
            criterion=loss,
            optimizer=optimizer,
            epochs=experiment_parameters["epochs"],
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            device=device,
            reduction="sum",
            transformer=(
                True if experiment_parameters["model"] == "hubert_ecg" else False
            ),
            use_demo_feats=experiment_parameters["use_de"],
            mixed_precision=False,
            early_stop=early_stopping,
            scheduler=scheduler,
            save_model=save_model,
        )
        logger.info("Training Finished")
        logger.info(f"Best Model epoch: {best_epoch+1}")

        fold_evaluation_path.mkdir(parents=True, exist_ok=True)
        with open(fold_evaluation_path / "train_metrics.json", "w") as f:
            json.dump(train_metrics, f, indent=4)
        with open(fold_evaluation_path / "validation_metrics.json", "w") as f:
            json.dump(val_metrics, f, indent=4)

        logger.info(f"Additional Evaluation of the Best Model at epoch {best_epoch+1}:")

        all_val_labels, all_val_probs, total_val_loss = validate_epoch(
            model,
            loss,
            0,
            1,
            val_dataloader,
            device,
            reduction="sum",
            transformer=(
                True if experiment_parameters["model"] == "hubert_ecg" else False
            ),
            use_demo_feats=experiment_parameters["use_de"],
            mixed_precision=False,
        )
        all_val_labels = np.array(all_val_labels)
        all_val_probs = np.array(all_val_probs)

        with open(fold_evaluation_path / "val_labels.npy", "wb") as f:
            np.save(f, all_val_labels)
        with open(fold_evaluation_path / "val_probs.npy", "wb") as f:
            np.save(f, all_val_probs)

        val_dataset.meta_data.to_csv(
            fold_evaluation_path / "evaluation_data.csv", index=True
        )

        val_metrics_epoch = compute_metrics_epoch(all_val_labels, all_val_probs)
        val_loss_dict = {"Loss": total_val_loss}
        val_metrics_dict = dict(val_loss_dict, **val_metrics_epoch)
        log_epoch_metrics(best_epoch, val_metrics_dict, f"Validation {fold_name}")
        target_names = ["No Chagas", "Chagas"]

        report, mcc = evaluate_classification_report(
            all_val_probs,
            all_val_labels,
            threshold=val_metrics_dict["Threshold"],
            target_names=target_names,
        )

        val_metrics_dict["mcc"] = mcc
        val_metrics_dict["clasification_report"] = report

        with open(fold_evaluation_path / "evaluation_metrics.json", "w") as f:
            json.dump(val_metrics_dict, f, indent=4)

        cm, cm_normalized = evaluate_confusion_matrix(
            all_val_probs, all_val_labels, threshold=val_metrics_dict["Threshold"]
        )
        precision, recall, thresholds, pr_auc, baseline = (
            evaluate_precision_recall_curve(all_val_probs, all_val_labels)
        )
        fpr, tpr, thresholds, roc_auc = evaluate_roc_curve(
            all_val_probs, all_val_labels
        )

        plot_confusion_matrix(
            cm,
            cm_normalized,
            threshold=val_metrics_dict["Threshold"],
            classes=target_names,
            path_to_save=fold_evaluation_path,
            model_name=f"{experiment_parameters["display_run_name"]} Fold {i}",
        )
        plot_roc_curve(
            fpr,
            tpr,
            roc_auc,
            fold_evaluation_path,
            f"{experiment_parameters["display_run_name"]} Fold {i}",
        )
        plot_precision_recall_curve(
            precision,
            recall,
            pr_auc,
            baseline,
            fold_evaluation_path,
            f"{experiment_parameters["display_run_name"]} Fold {i}",
        )
        if experiment_parameters["test_run"]:
            break
