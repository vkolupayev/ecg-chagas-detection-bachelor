
import os
import logging
from src.logger import setup_logger

setup_logger(f"{os.path.splitext(os.path.basename(__file__))[0]}")

from src.data import CleanLevel, ChagasDataset
from src.model.ecg_founder import Net1D
from src.model.hubert import ChagasHuBERT
from src.loss import BinaryFocalLoss
from src.train import (
    validate_epoch,
    compute_metrics_epoch,
)

from src.evaluate import (
    evaluate_precision_recall_curve,
    evaluate_roc_curve,
)
from src.utils import find_optimal_threshold, evaluation_at_threshold
from src.constants import RANDOM_SEED, seed_everything

import json
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader


RUN_NAME = "founder_fine_tune_samiptb_recall"
# RUN_NAME = "founder_fine_tune_samiptb_f2"
# RUN_NAME = "hubert_fine_tune_samiptb_recall"

def main():

    logger = logging.getLogger(__name__)
    seed_everything(RANDOM_SEED)
    
    logger.info(RUN_NAME)

    experiment_parameters = {
        "Batch Size": 256,
        "Pos Weight": 45.0,
        "Alpha": 0.0,
        "Gamma": 1.5,
        "Soft Label Term": 1.0,
    }
    clean_level = CleanLevel.REFINED

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    data_path = Path("data/training_data/")
    meta_data_path = data_path / "meta_data"

    model_paths = sorted(Path(f"models/{RUN_NAME}/").glob("*"), key=(lambda p: int(p.with_suffix('').parts[-1].split("_")[-1])))

    logger.info("Read Meta Data")
    meta_data = pd.read_csv(
        meta_data_path / "meta_data.csv",
        dtype={"record_name": str, "base_time": str, "base_date": str},
    )
    meta_data = meta_data[meta_data.sig_len >= 1500]
    metrics = pd.read_csv(meta_data_path / "channel_metrics.csv").drop("source", axis=1)
    meta_data = meta_data.merge(metrics, on="data_path")

    logger.info("Outlier removal")

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

    logger.info("Select only relevant columns")
    relevant_cols = [
        "record_name",
        "patient_id",
        "data_path",
        "fs",
        "source",
        "chagas_label",
    ]
    meta_data = meta_data.loc[:, relevant_cols]

    meta_data = meta_data[meta_data.source == "CODE-15%"]

    # Test run sample
    # logger.info("Test run 1% sample")
    # meta_data = meta_data.sample(frac=0.01)

    val_dataset = ChagasDataset(
        meta_data,
        fs=500 if "founder" in RUN_NAME else 100,
        max_sig_len=5000 if "founder" in RUN_NAME else 1000,
        clean_level=clean_level,
        code_soft_label_positive_term=1.0,
        normalize_flag=True if "founder" in RUN_NAME else False,
        scaling_flag= False if "founder" in RUN_NAME else True,
        to_tensor_flag=True,
        attention_mask_flag=False if "founder" in RUN_NAME else True,
        z_mean=None,
        z_std=None,
        random_crop_probability=0.0,
        max_sig_seconds=None,
        verbose=False,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=experiment_parameters["Batch Size"],
        shuffle=False,
        num_workers=6,
        prefetch_factor=2,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )
    
    loss = BinaryFocalLoss(
        pos_weight=experiment_parameters["Pos Weight"],
        alpha=experiment_parameters["Alpha"],
        gamma=experiment_parameters["Gamma"],
        reduction="sum",
    )
    loss.to(device)

    metrics = []

    for i, model_path in enumerate(model_paths, start=1):
        logger.info(f"Model from Fold {i}")
        if "founder" in RUN_NAME:
            model = Net1D(
                in_channels=12, 
                base_filters=64, #32 64
                ratio=1, 
                filter_list=[64,160,160,400,400,1024,1024],
                m_blocks_list=[2,2,2,3,3,4,4],
                kernel_size=16, 
                stride=2, 
                groups_width=16,
                verbose=False, 
                use_bn=False,
                use_do=False,
                n_classes=1
            )
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)
        else:
            model = ChagasHuBERT(
               simple_classifier=True, pretrained=True,
                model_path="pretrained_models/hubert-ecg-small"
            )
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)
        
        model.to(device)

        all_val_labels, all_val_probs, total_val_loss = validate_epoch(
            model,
            loss,
            0,
            1,
            val_dataloader,
            device,
            reduction="sum",
            transformer=False if "founder" in RUN_NAME else True,
            mixed_precision=False,
        )
        all_val_labels = np.array(all_val_labels)
        all_val_probs = np.array(all_val_probs)

        val_metrics_epoch = compute_metrics_epoch(all_val_labels, all_val_probs)
        val_loss_dict = {"Loss": total_val_loss}
        val_metrics_dict = dict(val_loss_dict, **val_metrics_epoch)

        precision, recall, thresholds, pr_auc, baseline = (
            evaluate_precision_recall_curve(all_val_probs, all_val_labels)
        )
        fpr, tpr, thresholds, roc_auc = evaluate_roc_curve(
            all_val_probs, all_val_labels
        )
        val_metrics_dict["ROC_AUC"] = roc_auc
        val_metrics_dict["PRC_AUC"] = pr_auc

        # optimal threshold
        logger.info("Searching for Optimal Threshold")
        optimal_threshold = find_optimal_threshold(all_val_labels, all_val_probs)
        logger.info("Running Evaluations at Optimal Threshold")
        precision, recall, f1_score, f2_score, mcc = evaluation_at_threshold(
            all_val_labels, all_val_probs, optimal_threshold
        )

        val_metrics_dict["Optimal_Threshold"] = optimal_threshold
        val_metrics_dict["Precision_Optimal"] = precision
        val_metrics_dict["Recall_Optimal"] = recall
        val_metrics_dict["F1_Optimal"] = f1_score
        val_metrics_dict["F2_Optimal"] = f2_score
        val_metrics_dict["MCC_Optimal"] = mcc

        metrics.append(val_metrics_dict)
    
    eval_path = Path(f"evaluations/{RUN_NAME}/aggregated/")
    eval_path.mkdir(parents=True, exist_ok=True)
    with open(eval_path / "code_evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()