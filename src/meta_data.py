import logging
from pathlib import Path
from multiprocessing import cpu_count

import wfdb
import torch
import numpy as np
import pandas as pd
import numpy.typing as npt
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.data import ChagasDataset, CleanLevel

logger = logging.getLogger(__name__)


def convert_numeric(value):
    if value.isnumeric():
        return int(value)
    try:
        return float(value)
    except ValueError as e:
        return value


def prep_record_meta(record_path: Path) -> dict:
    data_path = record_path.with_suffix("")
    record_dict = wfdb.rdrecord(data_path, physical=True).__dict__
    for key, value in record_dict.items():
        if isinstance(value, list):
            if len(set(value)) == 1:
                record_dict[key] = value[0]

    comments = record_dict.pop("comments")
    for comment in comments:
        key, value = comment.split(": ")
        key = key.lower().replace(" ", "_")
        value = convert_numeric(value)
        record_dict[key] = value
    record_dict.pop("p_signal")
    record_dict.pop("d_signal")
    record_dict.pop("e_p_signal")
    record_dict.pop("e_d_signal")
    record_dict["data_path"] = str(data_path)
    # pprint(record_dict)
    return record_dict


def prepare_dataset_meta_data(data_path: str, save: bool = False):
    record_paths = Path(data_path).glob("**/*.dat")
    record_paths = [x for x in record_paths if x.is_file()]

    if len(record_paths) == 0:
        raise Exception("No signals were read")

    meta_data: list[dict] = []
    progress_bar = tqdm(
        record_paths,
        unit="record",
        total=len(record_paths),
        desc="Prep ECG Meta Data",
        ncols=80,
    )
    for record_path in progress_bar:
        meta_data.append(prep_record_meta(record_path))
    logger.info(str(progress_bar))

    if len(meta_data) == 0:
        raise Exception("Problem populating dictionary")

    meta_data = pd.DataFrame.from_records(meta_data)

    try:
        meta_data["chagas_label"] = meta_data["chagas_label"].astype(str) == "True"
    except KeyError as e:
        raise e

    # all records of the same type
    meta_data["record_name"] = meta_data["record_name"].apply(str)
    # nans -> ""
    meta_data["base_time"] = meta_data["base_time"].apply(str)
    meta_data["base_date"] = meta_data["base_date"].apply(str)

    if save:
        meta_data.to_csv(data_path + "meta_data.csv", index=False)
    return meta_data


def df_append_resampled_signal_len(meta_data: pd.DataFrame, fs: int):
    # calculate and populate resampled signal length
    meta_data.loc[(meta_data["fs"] != fs), "sig_len_resampled"] = (
        fs * meta_data["sig_len"]
    ) / meta_data["fs"]

    meta_data.fillna({"sig_len_resampled": meta_data["sig_len"]}, inplace=True)

    # populate resampled singal fs
    meta_data.loc[:, "fs_resampled"] = fs

    meta_data["sig_len_resampled"] = meta_data["sig_len_resampled"].astype(int)


def prep_exams_data(
    exams_data_path: str | Path, save_to_csv: bool = False
) -> pd.DataFrame:
    exams_data_path = Path(exams_data_path)
    ptb_exams = pd.read_csv(
        exams_data_path / "ptbxl_database.csv", dtype={"ecg_id": str, "patient_id": int}
    )
    ptb_exams["patient_id"] = ptb_exams["patient_id"].astype(str)

    code_exams = pd.read_csv(
        exams_data_path / "code_exams.csv", dtype={"exam_id": str, "patient_id": str}
    )
    sami_exams = pd.read_csv(exams_data_path / "sami_exams.csv", dtype={"exam_id": str})

    ptb_exams["exam_id"] = ptb_exams["filename_hr"].str.split("/").str[-1]

    ptb_exams["patient_id"] = "ptb_" + ptb_exams["patient_id"]
    code_exams["patient_id"] = "code_" + code_exams["patient_id"]
    sami_exams["patient_id"] = pd.Series(
        (f"sami_{i}" for i in range(sami_exams.shape[0]))
    )

    # age, sex are already in the meta_data
    code_exams.drop(
        columns=["trace_file", "nn_predicted_age", "age", "is_male"], inplace=True
    )
    sami_exams.drop(columns=["age", "is_male", "nn_predicted_age"], inplace=True)
    ptb_exams.drop(
        columns=["age", "sex", "strat_fold", "filename_lr", "filename_hr", "ecg_id"],
        inplace=True,
    )

    db_exams = pd.concat(
        [code_exams, sami_exams, ptb_exams],
        ignore_index=True,
        axis="rows",
    )

    if save_to_csv:
        db_exams.to_csv(exams_data_path / "db_exams.csv", index=False)
    return db_exams


def calculate_metrics(
    meta_data: pd.DataFrame,
    fs: int,
    max_sig_len: int,
    clean_level: CleanLevel,
    calc_pp_amps: bool,
):
    meta_data = ChagasDataset(
        meta_data,
        fs=fs,
        pad_value=0.0,
        clean_level=clean_level,
        code_soft_label_positive_term=1.0,
        normalize_flag=False,
        scaling_flag=False,
        to_tensor_flag=False,
        attention_mask_flag=False,
        max_sig_len=max_sig_len,
    )
    dataloader = DataLoader(
        meta_data,
        batch_size=64,
        num_workers=min(6, cpu_count() - 6),
        prefetch_factor=2,
        drop_last=False,
        shuffle=False,
    )

    n_signals = len(meta_data)
    n_channels = meta_data[0]["signal"].shape[1]

    means = torch.empty((n_signals, n_channels), dtype=torch.float64)
    stds = torch.empty((n_signals, n_channels), dtype=torch.float64)
    if calc_pp_amps:
        pp_amps = torch.empty((n_signals, n_channels), dtype=torch.float64)

    progress_bar = tqdm(
        dataloader,
        desc="Calculating ECG Metrics",
        ncols=80,
    )
    idx = 0
    for batch in progress_bar:
        signal = batch["signal"]
        batch_shape = signal.shape[0]

        ch_means = signal.mean(dim=1)
        ch_stds = signal.std(dim=1)
        means[idx : idx + batch_shape] = ch_means
        stds[idx : idx + batch_shape] = ch_stds

        if calc_pp_amps:
            ch_ppas = signal.max(dim=1).values - signal.min(dim=1).values
            pp_amps[idx : idx + batch_shape] = ch_ppas

        idx += batch_shape
    logger.info(str(progress_bar))

    if calc_pp_amps:
        return means.numpy(), stds.numpy(), pp_amps.numpy()
    else:
        return means.numpy(), stds.numpy()


def create_metrics_df(
    meta_data: pd.DataFrame,
    means: npt.NDArray[np.float64],
    stds: npt.NDArray[np.float64],
    pp_amps: npt.NDArray[np.float64] | None = None,
) -> pd.DataFrame:
    records_data_paths = meta_data["data_path"].to_list()
    records_source = meta_data["source"].to_list()

    metrics_dict = {"data_path": records_data_paths, "source": records_source}

    means = {f"mean_{channel_nr + 1}": means[:, channel_nr] for channel_nr in range(12)}
    stds = {f"std_{channel_nr + 1}": stds[:, channel_nr] for channel_nr in range(12)}

    metrics = metrics_dict | means | stds

    if pp_amps is not None:
        pp_amps = {
            f"pp_amp_{channel_nr + 1}": pp_amps[:, channel_nr]
            for channel_nr in range(12)
        }
        metrics = metrics | pp_amps

    return pd.DataFrame(metrics)


def create_mean_metrics_df(metrics_df: pd.DataFrame) -> pd.DataFrame:
    all_datasets = metrics_df.iloc[:, 2:]
    n_all = all_datasets.shape[0]
    all_datasets = all_datasets.mean()

    metrics_sources = metrics_df["source"].unique()

    df_list = [all_datasets]
    df_n_list = [n_all]
    for metrics_source in metrics_sources:
        df = metrics_df.loc[(metrics_df.source == metrics_source)].iloc[:, 2:]
        n = df.shape[0]
        df = df.mean()

        df_list.append(df)
        df_n_list.append(n)

    mean_metrics_df = pd.concat(
        df_list,
        axis="columns",
    )

    dataset_names = ["All"] + metrics_sources.tolist()
    mean_metrics_df.columns = dataset_names

    n_data = pd.DataFrame([df_n_list], columns=dataset_names)
    n_data.index = ("n",)

    mean_metrics_df = pd.concat([mean_metrics_df, n_data]).T
    mean_metrics_df["n"] = mean_metrics_df["n"].astype(int)

    return mean_metrics_df


def prepare_dataset_metrics(
    meta_data: pd.DataFrame,
    fs: int = None,
    max_sig_len: int = 2934,
    clean_level: CleanLevel = CleanLevel.REFINED,
    calc_pp_amps: bool = False,
    return_df: bool = False,
    return_full_data_metrics: bool = False,
):
    """

    Prepare Dataset metrics.

    Args:
        meta_data (pd.DataFrame): Filtered DataFrame on which metric calculation is performed.
        fs (int, optional): Resample signals to this sample rate. If None keep original signal sample rate.
        max_sig_len (int):
        calc_pp_amps (bool, optional): Flag whether to calulate Peak to Peak Amplitudes. Defaults to False.
        return_df (bool, optional): Flag whether to return DataFrame. Defaults to False.
        return_full_data_metrics (bool, optional): Flag whether to return metrics for each record. Defaults to False.

    Returns:
        DataFrame with metrics of each record
        or
        DataFrames with metrics of each record and mean metrics
        or
        Tuples of numpy arrays consisting of either mean metrics or metrics for each record
    """

    metrics = calculate_metrics(meta_data, fs, max_sig_len, clean_level, calc_pp_amps)

    if return_df:
        metrics_df = create_metrics_df(meta_data, *metrics)
        mean_metrics_df = create_mean_metrics_df(metrics_df)

        if return_full_data_metrics:
            return metrics_df, mean_metrics_df
        else:
            return mean_metrics_df
    else:
        if return_full_data_metrics:
            return metrics
        else:
            return tuple(metric.mean(axis=0) for metric in metrics)


def upsample_samitrop(meta_data: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [meta_data, meta_data[meta_data["source"] == "SaMi-Trop"]],
        ignore_index=True,
        axis="rows",
    )


def _cardiac_condition_chagas_bias(df):

    df = df.copy().fillna(False, inplace=False)

    if (df["RBBB"]) and (df["AF"] or df["1dAVb"]):
        return 0.25
    elif (df["RBBB"]) and not (df["AF"] or df["1dAVb"]):
        return 0.15
    else:
        return 0


def clip(value, upper):
    return upper if value > upper else value


# TODO: Refactor to a vectorized version
def add_refined_label_smooth(meta_data: pd.DataFrame) -> pd.DataFrame:
    """AIChagas Team approach from:

    Managing Label Uncertainty in the Detection of Chagas Disease from the ECG

    Args:
        meta_data (pd.DataFrame): meta data with demographic information
    """
    meta_data = meta_data.copy()
    nunique_code_label_patient_ids = (
        meta_data.loc[meta_data["source"] == "CODE-15%", ["patient_id", "chagas_label"]]
        .groupby("patient_id")
        .nunique()
    )
    nunique_code_label_patient_ids = nunique_code_label_patient_ids[
        nunique_code_label_patient_ids["chagas_label"] > 1
    ].index.to_list()

    meta_data["soft_label"] = meta_data["chagas_label"].astype(float)

    for index, row in meta_data.loc[
        meta_data.patient_id.isin(nunique_code_label_patient_ids)
    ].iterrows():
        patient_chagas = meta_data.loc[
            meta_data["patient_id"] == row["patient_id"], "chagas_label"
        ]
        total, chagas_positive = patient_chagas.count(), patient_chagas.sum()
        meta_data.at[index, "soft_label"] = (
            row["soft_label"] * 0.51 + 0.49 * chagas_positive / total
        )

    for index, row in meta_data.loc[
        (meta_data["source"] == "CODE-15%") & (meta_data["chagas_label"] == False)
    ].iterrows():
        meta_data.at[index, "soft_label"] = clip(
            row["soft_label"] + _cardiac_condition_chagas_bias(row), 0.49
        )

    return meta_data


def add_demographic_dummies(meta_data: pd.DataFrame, return_dummy_len=True):

    bins = [0, 20] + list(range(25, 90, 5)) + [101]

    labels = ["[0-20)"] + [f"[{i-5}-{i})" for i in range(25, 90, 5)] + ["[85-101)"]
    meta_data["age_groups"] = pd.cut(
        meta_data["age"], bins=bins, labels=labels, right=False
    )

    demo_dummies = pd.get_dummies(meta_data[["age_groups", "sex"]], dummy_na=False)

    meta_data = pd.concat(
        [meta_data, demo_dummies],
        axis="columns",
    )

    return meta_data, demo_dummies.shape[1] if return_dummy_len else meta_data
