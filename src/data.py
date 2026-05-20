from enum import Enum
from typing import Dict
import logging

import torch
import numpy as np
import pandas as pd
import wfdb

from torch.utils.data import Dataset
from src.data_preprocess import (
    clean_signal_multi_channel,
    normalize,
    pad,
    truncate,
    scaling,
    apply_filter,
    generate_attention_mask,
    filter_bandpass,
    filter_pipeline,
    resample_polyphase,
)

logger = logging.getLogger(__name__)


class CleanLevel(Enum):
    NONE = 0
    PARTIAL = 1
    FULL = 2
    HUBERT = 3
    ECG_FOUNDER = 4
    REFINED = 5


# TODO: Padding value selection
class PadValue(Enum):
    ZERO = "zero"
    MEAN = "mean"
    MEDIAN = "median"


# TODO: Refactor ops to transforms
class ChagasDataset(Dataset):
    def __init__(
        self,
        meta_data: pd.DataFrame,
        fs: int = 400,
        max_sig_len: int = 2934,
        pad_value: float | None = 0.0,
        clean_level: CleanLevel = CleanLevel.FULL,
        code_soft_label_positive_term=0.8,
        refined_label_smooth_flag=False,
        normalize_flag=True,
        scaling_flag=False,
        to_tensor_flag=True,
        attention_mask_flag=True,
        z_mean=np.array(
            [
                3.93754815e-04,
                1.97562670e-04,
                -1.96428850e-04,
                -2.90809028e-04,
                2.98538327e-04,
                3.64035677e-06,
                -6.31806771e-04,
                -4.73925188e-04,
                -2.64163562e-04,
                2.21555235e-04,
                4.07253257e-04,
                3.01333987e-04,
            ]
        ),
        z_std=np.array(
            [
                0.22254967,
                0.2438941,
                0.17276358,
                0.21906845,
                0.1550138,
                0.17801439,
                0.26204946,
                0.35109691,
                0.41220576,
                0.41708225,
                0.3917299,
                0.32531538,
            ]
        ),
        random_crop_probability=0.0,
        max_sig_seconds=None,
        return_demo_feats=False,
        zero_demo_feat_prob=0.0,
        random_gaussian_noise_prob=0.0,
        random_temporal_mask_prob=0.0,
        random_lead_mask_prob=0.0,
        random_magnitude_scale_prob=0.0,
        transforms=None,
        verbose=False,
    ):
        self.meta_data = meta_data
        self.fs = fs
        self.max_sig_len = max_sig_len
        self.pad_value = pad_value
        self.clean_level = clean_level
        self.code_soft_label_positive_term = code_soft_label_positive_term
        self.refined_label_smooth_flag = refined_label_smooth_flag
        self.normalize_flag = normalize_flag
        self.scaling_flag = scaling_flag
        self.to_tensor_flag = to_tensor_flag
        self.attention_mask_flag = attention_mask_flag
        self.z_mean = z_mean
        self.z_std = z_std
        self.random_crop_probability = random_crop_probability
        self.max_sig_seconds = max_sig_seconds
        self.return_demo_feats = return_demo_feats
        self.zero_demo_feat_prob = zero_demo_feat_prob
        self.random_gaussian_noise_prob = random_gaussian_noise_prob
        self.random_temporal_mask_prob = random_temporal_mask_prob
        self.random_lead_mask_prob = random_lead_mask_prob
        self.random_magnitude_scale_prob = random_magnitude_scale_prob
        self.transforms = transforms
        self.verbose = verbose

    def __len__(self) -> int:
        return self.meta_data.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        signal = wfdb.rdrecord(
            self.meta_data["data_path"].iloc[idx], physical=True
        ).__dict__["p_signal"]
        # signal = wfdb.rdrecord(
        #     self.meta_data["data_path"].iloc[idx], physical=False
        # ).__dict__["d_signal"].astype(float)
        # signal (samples, channel)
        if self.verbose:
            logger.info("Signal shape WFDB read")
            logger.info(signal.shape)
        label = self.meta_data["chagas_label"].iloc[idx].astype(float)

        label_smooth = label.copy()

        if self.refined_label_smooth_flag:
            label_smooth = self.meta_data["soft_label"].iloc[idx]

        if (self.meta_data["source"].iloc[idx] == "CODE-15%") and (
            self.code_soft_label_positive_term < 1.0
        ):
            label_smooth = (
                np.clip(
                    self.code_soft_label_positive_term * label_smooth,
                    a_min=0.51,
                    a_max=None,
                )
                if label == 1.0
                else np.clip(
                    1 - self.code_soft_label_positive_term + label_smooth,
                    a_max=0.49,
                    a_min=None,
                )
            )

        signal_fs = self.meta_data["fs"].iloc[idx]

        if self.max_sig_seconds:
            signal_max_len = int(self.max_sig_seconds * self.fs)
        else:
            signal_max_len = self.max_sig_len

        signal = np.nan_to_num(signal, nan=0.0)
        match self.clean_level:
            case CleanLevel.FULL:
                signal = clean_signal_multi_channel(
                    signal, fs=signal_fs, dwt=True, wav="db4", level=4
                )
            case CleanLevel.PARTIAL:
                signal = clean_signal_multi_channel(signal, fs=signal_fs, dwt=False)
            case CleanLevel.HUBERT:
                signal = apply_filter(
                    signal.T, [0.05, 47], signal_fs
                ).T  # this band focuses on dominant component of ecg waves
            case CleanLevel.ECG_FOUNDER:
                signal = filter_bandpass(signal.T, signal_fs).T
            case CleanLevel.REFINED:
                # array.copy() to avoid negative strides...
                signal = filter_pipeline(signal, signal_fs).copy()

        # signal (samples, channel) after cleaning
        if self.verbose:
            logger.info("Signal shape after cleaning")
            logger.info(signal.shape)

        if self.fs:
            # array.copy() to avoid negative strides...
            signal = (
                signal
                if signal_fs == self.fs
                else resample_polyphase(signal, signal_fs, self.fs).copy()
            )

        if self.verbose:
            logger.info("Signal shape after resampling")
            logger.info(signal.shape)
        # signal (samples, channel) after resampling

        # Data augs/prep/norm/scale
        # Random lead masking, temporal masking
        # Random crop / padding / Truncation
        # Random magnitude scaling, gaussian noise

        _random_temporal_mask_prob = np.random.rand(1)[0]
        _random_lead_mask_prob = np.random.rand(1)[0]
        _random_magnitude_scale_prob = np.random.rand(1)[0]
        _random_gaussian_noise_prob = np.random.rand(1)[0]

        if _random_temporal_mask_prob < self.random_temporal_mask_prob:
            # 10-20%
            mask_size = np.random.uniform(low=0.1, high=0.2, size=None)
            mask_size = round((signal.shape[0] * mask_size))

            start = np.random.randint(0, signal.shape[0] - mask_size + 1)
            signal[start : start + mask_size, :] = 0

        if _random_lead_mask_prob < self.random_lead_mask_prob:
            # 1-4 leads
            lead_prob = np.random.rand(1)[0]
            mask_n_lead = 0
            if lead_prob < 0.4:
                mask_n_lead = 1
            elif lead_prob < 0.7:
                mask_n_lead = 2
            elif lead_prob < 0.9:
                mask_n_lead = 3
            else:
                mask_n_lead = 4
            mask_lead_idx = np.random.choice(
                list(range(0, 12)), size=mask_n_lead, replace=False, p=None
            )
            signal[:, mask_lead_idx] = 0.0

        original_signal_len = signal.shape[0]

        if (original_signal_len < signal_max_len) and (self.pad_value is not None):
            signal = pad(signal, signal_max_len, self.pad_value)
        elif original_signal_len > signal_max_len:
            # random crop
            random_crop_prob = np.random.rand(1)[0]
            if self.verbose:
                logger.info(random_crop_prob)
            if random_crop_prob < self.random_crop_probability:
                if self.verbose:
                    logger.info("random crop")
                start = np.random.randint(0, signal.shape[0] - signal_max_len + 1)
                signal = signal[start : start + signal_max_len, :]
            else:
                signal = truncate(signal, signal_max_len)
        # signal is correct size else: pass...
        # signal (samples, channel) after resizing
        if self.verbose:
            logger.info("Signal shape after resizing")
            logger.info(signal.shape)

        # when using non global z score, scaling
        # do it before noise, scaler or similar augmentations.
        # hubert uses scaling instead of normalization
        if self.normalize_flag:
            signal = normalize(signal, self.z_mean, self.z_std)
        if self.scaling_flag:
            signal = scaling(signal)

        if _random_magnitude_scale_prob < self.random_magnitude_scale_prob:
            # 1.1 - 1.3
            scaler = np.random.uniform(low=1.1, high=1.3, size=None)
            signal = signal * scaler
        if _random_gaussian_noise_prob < self.random_gaussian_noise_prob:
            mu, sigma = 0, 0.1  # mean and standard deviation
            gauss_noise = np.random.normal(mu, sigma, signal.shape)
            signal = signal + gauss_noise
        # when using global z score, scaling
        # do it after augmentations.
        # hubert uses scaling instead of normalization
        # if self.normalize_flag:
        #     signal = normalize(signal, self.z_mean, self.z_std)

        if self.attention_mask_flag:
            # TODO: attention mask based on lead, temporal masking as well?
            attention_mask = generate_attention_mask(signal, original_signal_len)
            if self.verbose:
                logger.info("Attention Mask shape")
                logger.info(attention_mask.shape)

        if self.return_demo_feats:
            demo_feats = (
                self.meta_data.iloc[idx]
                .filter(regex="age_groups_|sex_")
                .to_numpy(float)
            )

            age_prob = np.random.rand(1)[0]
            sex_prob = np.random.rand(1)[0]
            if age_prob < self.zero_demo_feat_prob:
                demo_feats[:-2] = 0
            if sex_prob < self.zero_demo_feat_prob:
                demo_feats[-2:] = 0

        if self.to_tensor_flag:
            # to tensor transposes signals from (samples, channel) to (channel, samples)
            signal = signal.T
            if self.verbose:
                logger.info("Signal shape after transpose")
                logger.info(signal.shape)
            if self.attention_mask_flag:
                # Transformer expects a flattened signal
                attention_mask = attention_mask.T
                if self.verbose:
                    logger.info("Attention Mask shape after transpose")
                    logger.info(signal.shape)

                attention_mask = attention_mask.reshape(-1)
                signal = signal.reshape(-1)
                if self.verbose:
                    logger.info("Signal shape after flattening")
                    logger.info(signal.shape)
                    logger.info("Attention Mask shape after flattening")
                    logger.info(signal.shape)

                attention_mask = torch.tensor(attention_mask, dtype=torch.int)

            signal = torch.tensor(signal, dtype=torch.float32)
            label_smooth = torch.tensor(label_smooth, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.float32)
            if self.return_demo_feats:
                demo_feats = torch.tensor(demo_feats, dtype=torch.float32)

        if self.verbose:
            logger.info("Signal shape after flags")
            logger.info(signal.shape)

        data = {"signal": signal, "label_train": label_smooth, "label": label}
        if self.attention_mask_flag:
            data["attention_mask"] = attention_mask
        if self.return_demo_feats:
            data["demo_feats"] = demo_feats

        if self.verbose:
            logger.info(data)

        return data

    def sources(self):
        return self.meta_data["source"].to_list()
