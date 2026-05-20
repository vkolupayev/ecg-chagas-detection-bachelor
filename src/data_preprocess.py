import numpy as np
import pywt

from scipy.signal import iirnotch, butter, filtfilt, medfilt, sosfiltfilt, resample_poly
from biosppy.signals.tools import filter_signal
from wfdb.processing import resample_sig


def filter_pipeline(
    ecg_signal,
    fs=400,
    notch_quality_factor=30,
    butter_order=4,
    bandpass_range=[0.5, 50],
):
    # axis=0 assumes shape is (time, channels)
    assert fs > 2
    if fs / 2 >= 60:
        b, a = iirnotch(60, notch_quality_factor, fs)
        ecg_signal = filtfilt(b, a, ecg_signal, axis=0)
    if fs / 2 >= 50:
        b, a = iirnotch(50, notch_quality_factor, fs)
        ecg_signal = filtfilt(b, a, ecg_signal, axis=0)
        # Filter ecg noise keeping the most important freqs
        # Design butterworth filter
        sos = butter(
            N=butter_order, Wn=bandpass_range, btype="bandpass", fs=fs, output="sos"
        )
        ecg_signal = sosfiltfilt(sos, ecg_signal, axis=0)
    else:
        sos = butter(
            N=butter_order, Wn=bandpass_range[0], btype="highpass", fs=fs, output="sos"
        )
        ecg_signal = sosfiltfilt(sos, ecg_signal, axis=0)

    return ecg_signal


def resample_polyphase(signal, original_fs, target_fs):

    gcd = np.gcd(original_fs, target_fs)
    up = target_fs // gcd
    down = original_fs // gcd

    # resample_poly applies zero-phase FIR filtering by default, preserving ECG phase
    # axis=0 assumes shape is (time, channels)
    return resample_poly(signal, up, down, axis=0)


def normalize(signal, mean=None, std=None):
    """Normalizes ecg signals by z score. Default - normalizing per channel.

    Args:
        signal (np.array): 2D numpy array of shape (time, channel)
        mean (np.array, optional): per channel means of dataset signals. Defaults to None.
        std (np.array, optional): per channel standard deviations of dataset signals. Defaults to None.

    Returns:
        (np.array): 2D numpy array of shape (time, channel)
    """
    if mean is None:
        mean = signal.mean(axis=0)
    if std is None:
        std = signal.std(axis=0)

    std[std == 0] = 1  # avoid division by zero

    return (signal - mean) / std


def pad(signal: np.array, max_sig_len: int, pad_value: float) -> np.array:
    return np.pad(
        signal,
        ((0, max_sig_len - signal.shape[0]), (0, 0)),
        "constant",
        constant_values=pad_value,
    )


def truncate(signal: np.array, max_sig_len: int):
    return signal[:max_sig_len]


def scaling(seq, smooth=1e-8):
    """Min Max Scaling

    Args:
        seq (np.array): 2D numpy array of shape (time, channel)
        smooth (float, optional): smoothing factor to avoid division by 0. Defaults to 1e-8.

    Returns:
        (np.array): Scaled 2D numpy array of shape (time, channel)
    """
    return (
        2
        * (seq - np.min(seq, axis=0))
        / (np.max(seq, axis=0) - np.min(seq, axis=0) + smooth)
        - 1
    )


def generate_attention_mask(signal, original_signal_len):
    """
    Generates attention mask after padding and truncation. Checks if a record channel is empty(all equal to 0).

    Args:
        signal (np.array): padded and truncated signal. Shape: [signal_len, signal_channels]
        original_signal_len (int): signal length before padding and truncation.

    Returns:
        np.array: attention_mask
    """

    if signal.shape[0] <= original_signal_len:
        attention_mask = np.ones_like(signal, dtype=int)
    else:
        attention_mask_true = np.ones((original_signal_len, signal.shape[1]), dtype=int)
        attention_mask_false = np.zeros(
            (signal.shape[0] - original_signal_len, signal.shape[1]), dtype=int
        )
        attention_mask = np.concat([attention_mask_true, attention_mask_false])

    empty_channel_indices = np.where((~np.any(signal, axis=0)))[0]
    if empty_channel_indices.size:
        attention_mask[:, empty_channel_indices] = np.zeros(
            (attention_mask.shape[0], empty_channel_indices.size), dtype=int
        )

    return attention_mask


# OLD


def resample(x: np.array, current_fs: int, fs_to_resample: int):
    resampled_signal = []
    for chan in range(x.shape[1]):
        resampled_x, _ = resample_sig(x[:, chan], current_fs, fs_to_resample)
        resampled_signal.append(resampled_x)

    return np.column_stack(resampled_signal)


# HuBERT-ECG
def apply_filter(signal, filter_bandwidth, fs=500):
    """
    Bandpass filtering to remove noise, artifacts etc
    :param signal: 2D numpy array of shape (channels, time)
    :param filter_bandwidth: Cutoff frequencies
    :param fs: Sampling frequency
    :return: filtered signal of shape (channels, time)
    """
    # Calculate filter order
    order = int(0.3 * fs)
    # Filter signal
    signal, _, _ = filter_signal(
        signal=signal,
        ftype="FIR",
        band="bandpass",
        order=order,
        frequency=filter_bandwidth,
        sampling_rate=fs,
    )
    return signal


# ECG Founder
def filter_bandpass(signal, fs):
    """
    Bandpass filter
    :param signal: 2D numpy array of shape (channels, time)
    :param fs: sampling frequency
    :return: filtered signal of shape (channels, time)
    """
    # Remove power-line interference
    b, a = iirnotch(50, 30, fs)
    filtered_signal = np.zeros_like(signal)
    for c in range(signal.shape[0]):
        filtered_signal[c] = filtfilt(b, a, signal[c])

    # Simple bandpass filter
    b, a = butter(N=4, Wn=[0.67, 40], btype="bandpass", fs=fs)
    for c in range(signal.shape[0]):
        filtered_signal[c] = filtfilt(b, a, filtered_signal[c])

    # Remove baseline wander
    baseline = np.zeros_like(filtered_signal)
    for c in range(filtered_signal.shape[0]):
        kernel_size = int(0.4 * fs) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure kernel size is odd
        baseline[c] = medfilt(filtered_signal[c], kernel_size=kernel_size)
    filter_ecg = filtered_signal - baseline

    return filter_ecg


def notch_filter(signal, fs, notch_freq=50, Q=30):
    """
    Cleans utility frequencies 50, 60 Hz freq.
    Since this noise is stationary, we need a precise and non-local filter
    """

    b, a = iirnotch(notch_freq, Q, fs)
    return filtfilt(b, a, signal)


def highpass_filter(signal, fs, cutoff=0.5, order=2):
    """
    Cleans extreme baseline drift.
    This is so out dwt would be better.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return filtfilt(b, a, signal)


def get_coef_threshold(coeffs, N):
    """
    Calculates the universal noise threshold.
    Excludes coeffs[0] to prevent excessive filtering.
    """
    all_detail_coeffs = np.concatenate(coeffs[1:])
    median = np.median(all_detail_coeffs)
    mad = np.median(np.abs(all_detail_coeffs - median))
    return mad * 1.4826 * np.sqrt(2 * np.log(N))  # Universal threshold


def threshold_coeffs(coeffs, threshold):
    """
    Applies soft thresholding to detail coefficients.
    """
    coeffs_thresholded = [coeffs[0]]
    for c in coeffs[1:]:
        c_thresholded = np.sign(c) * np.maximum(np.abs(c) - threshold, 0)
        coeffs_thresholded.append(c_thresholded)
    return coeffs_thresholded


def clean_signal(ecg_signal, fs=400, dwt=True, wav="db4", level=4):
    """
    Full cleaning pipeline. For now this is a bit unfinished, no fs/level handling
    """
    ecg_signal_filt = notch_filter(ecg_signal, fs)  # Remove 50Hz noise
    if int(fs / 2) > 60:
        ecg_signal_filt = notch_filter(
            ecg_signal, fs, notch_freq=60
        )  # Remove 60Hz noise
    ecg_signal_filt = highpass_filter(
        ecg_signal_filt, fs, cutoff=0.5
    )  # Remove baseline drift

    if dwt:
        coeffs = pywt.wavedec(
            ecg_signal_filt, wav, level=level, mode="per"
        )  # Set wavelet level
        threshold = get_coef_threshold(coeffs, len(ecg_signal))
        coeffs_filtered = threshold_coeffs(coeffs, threshold)
        return pywt.waverec(
            coeffs_filtered, wav, mode="per"
        )  # Ensure reconstruction works
    else:
        return ecg_signal_filt


def clean_signal_multi_channel(ecg_signal, fs=400, dwt=True, wav="db4", level=4):
    mc_cleaned_signal = []
    for i in range(ecg_signal.shape[1]):
        mc_cleaned_signal.append(clean_signal(ecg_signal[:, i], fs, dwt, wav, level))
    return np.column_stack(mc_cleaned_signal)
