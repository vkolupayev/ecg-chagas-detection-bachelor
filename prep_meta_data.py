import os
import sys
import argparse
import logging
from pathlib import Path

from src.logger import setup_logger
setup_logger(f"{os.path.splitext(os.path.basename(__file__))[0]}")

from src.data import CleanLevel
from src.meta_data import prepare_dataset_metrics, prepare_dataset_meta_data, prep_exams_data

# Parse arguments.
def get_parser():
    description = "Prepare dataset meta data."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-d", "--data_folder_path", type=str, required=True)
    return parser


# Run the code.
def run(args):
    logger = logging.getLogger(__name__)
    # Check if valid path
    # Check if files are present
    data_path = Path(args.data_folder_path)

    if not data_path.exists():
        raise Exception("Path does not exist")
    
    save_meta_data_path = data_path / "meta_data"
    save_meta_data_path.mkdir(parents=True, exist_ok=True)

    logger.info("Preparing Dataset Meta Data")
    meta_data = prepare_dataset_meta_data(data_path)
    logger.info(f"Saving Dataset Meta Data to: {save_meta_data_path / "meta_data.csv"}")
    meta_data.to_csv(save_meta_data_path / "meta_data.csv", index=False)

    meta_data = meta_data[meta_data.sig_len >= 1500]
    
    logger.info("Preparing Dataset Exams Data")
    exams_data = prep_exams_data(data_path / "db_exams")
    logger.info(f"Saving Dataset Exams Data to: {save_meta_data_path / "db_exams.csv"}")
    exams_data.to_csv(save_meta_data_path / "db_exams.csv", index=False)


    logger.info("Preparing Original Sample Rate Dataset Metrics")
    metrics_df, mean_metrics_df = prepare_dataset_metrics(
        meta_data,
        fs=None,
        max_sig_len=2934, # 55% Quantile falls into this sig_len, 400Hz 7.335s, 500Hz 5.868s
        clean_level=CleanLevel.REFINED,
        calc_pp_amps=True,
        return_df=True,
        return_full_data_metrics=True,
    )

    logger.info(f"Saving Original Sample Rate Dataset Metrics to: {save_meta_data_path / "channel_metrics.csv"}")
    metrics_df.to_csv(save_meta_data_path / "channel_metrics.csv", index=False)
    logger.info(f"Saving Original Sample Rate Dataset Mean Metrics to: {save_meta_data_path / "mean_channel_metrics.csv"}")
    mean_metrics_df.to_csv(save_meta_data_path / "mean_channel_metrics.csv", index=True)

    logger.info("Preparing Original Sample Rate Unfiltered Dataset Metrics")
    metrics_df, mean_metrics_df = prepare_dataset_metrics(
        meta_data,
        fs=None,
        max_sig_len=2934, # 55% Quantile falls into this sig_len, 400Hz 7.335s, 500Hz 5.868s
        clean_level=CleanLevel.NONE,
        calc_pp_amps=True,
        return_df=True,
        return_full_data_metrics=True,
    )

    logger.info(f"Saving Original Sample Rate Unfiltered Dataset Metrics to: {save_meta_data_path / "unfiltered_channel_metrics.csv"}")
    metrics_df.to_csv(save_meta_data_path / "unfiltered_channel_metrics.csv", index=False)
    logger.info(f"Saving Original Sample Rate Unfiltered Dataset Mean Metrics to: {save_meta_data_path / "unfiltered_mean_channel_metrics.csv"}")
    mean_metrics_df.to_csv(save_meta_data_path / "unfiltered_mean_channel_metrics.csv", index=True)


    logger.info("Preparing HuBERT-ECG Dataset Metrics")
    metrics_df, mean_metrics_df = prepare_dataset_metrics(
        meta_data,
        fs=100,
        max_sig_len=734, # 55% Quantile falls into this sig_len. 100Hz 7.34s
        clean_level=CleanLevel.REFINED,
        calc_pp_amps=True,
        return_df=True,
        return_full_data_metrics=True,
    )

    logger.info(f"Saving HuBERT-ECG Dataset Metrics to: {save_meta_data_path / "hubert_channel_metrics.csv"}")
    metrics_df.to_csv(save_meta_data_path / "hubert_channel_metrics.csv", index=False)
    logger.info(f"Saving HuBERT-ECG Dataset Mean Metrics to: {save_meta_data_path / "hubert_mean_channel_metrics.csv"}")
    mean_metrics_df.to_csv(save_meta_data_path / "hubert_mean_channel_metrics.csv", index=True)

    logger.info("Preparing ECG Founder Dataset Metrics")
    metrics_df, mean_metrics_df = prepare_dataset_metrics(
        meta_data,
        fs=500,
        max_sig_len=3668, # 55% Quantile falls into this sig_len. 500Hz 7.34s
        clean_level=CleanLevel.REFINED,
        calc_pp_amps=True,
        return_df=True,
        return_full_data_metrics=True,
    )

    logger.info(f"Saving ECG Founder Dataset Metrics to: {save_meta_data_path / "founder_channel_metrics.csv"}")
    metrics_df.to_csv(save_meta_data_path / "founder_channel_metrics.csv", index=False)
    logger.info(f"Saving ECG Founder Dataset Mean Metrics to: {save_meta_data_path / "founder_mean_channel_metrics.csv"}")
    mean_metrics_df.to_csv(save_meta_data_path / "founder_mean_channel_metrics.csv", index=True)


if __name__ == "__main__":
    run(get_parser().parse_args(sys.argv[1:]))
