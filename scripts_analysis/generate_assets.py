from pathlib import Path
from itertools import product

import wfdb
import pandas as pd
from scipy.signal import butter, iirnotch

from src.utils import generate_filter_response, generate_signal_spectrum, generate_signal_time
from src.plot_utils import plot_filter_response, plot_signal_time_spectrum_overlapped, plot_signal_time_spectrum, plot_demographics_count, plot_wfdb_record
from src.data_preprocess import filter_pipeline, resample_polyphase
from src.meta_data import add_demographic_dummies

def main():

    path = Path("assets/filter_response/")
    path.mkdir(parents=True, exist_ok=True)

    fss = [400, 500]
    filter_hzs = [50, 60]
    for fs, filter_hz in product(fss, filter_hzs):
        title = f"IIR Notch Q=30 Filter {filter_hz}Hz"
        b, a = iirnotch(60, 30, fs)
        freq, h, impulse_time, impulse_response = generate_filter_response(b, a, fs)
        plot_filter_response(freq, h, impulse_time, impulse_response, fs, title=title, file_path=path/f"iir-notch-30-{filter_hz}-{fs}.png")
    
    filter_orders = [3, 4]
    bandpass_ranges = [[0.5, 50], 0.5]
    for fs, filter_order, bandpass_range in product(fss, filter_orders, bandpass_ranges):
        filter_type = "bandpass" if isinstance(bandpass_range, list) else "highpass"
        title = f"Butterworth {filter_type} {filter_order}{"rd" if filter_order == 3 else "th"} order filter {bandpass_range}"
        file_name = f"butter-{filter_order}{"rd" if filter_order == 3 else "th"}-{"bp" if filter_type == "bandpass" else "hp"}-{str(bandpass_range).replace(".", "_").strip("[]").replace(", ", "-")}-{fs}.png"
        b, a = butter(N=filter_order, Wn=bandpass_range, btype=filter_type, fs=fs)
        freq, h, impulse_time, impulse_response = generate_filter_response(b, a, fs)
        plot_filter_response(freq, h, impulse_time, impulse_response, fs, title=title, file_path=path/file_name, focus_spectrum_to_hz=None if "bandpass" else 10)

    path = Path("assets/signal_analysis/")
    path.mkdir(parents=True, exist_ok=True)

    wfdb_record_samitrop = wfdb.rdrecord(
    "./data/training_data/samitrop_wfdb/4991", physical=True
    )
    wfdb_record_ptbxl = wfdb.rdrecord(
        "./data/training_data/ptbxl_500_wfdb/00000/00001_hr", physical=True
    )
    wfdb_record_code15 = wfdb.rdrecord(
        "./data/training_data/code15_wfdb/exams_part0/116", physical=True
    )
    wfdb_records = [wfdb_record_samitrop, wfdb_record_ptbxl, wfdb_record_code15]
    wfdb_records_source = ["SaMi-Trop", "PTB-XL", "Code-15%"]

    for (wfdb_record, wfdb_record_source) in zip(wfdb_records, wfdb_records_source):

        signal = wfdb_record.__dict__["p_signal"]
        fs = wfdb_record.__dict__["fs"]
        lead_names = wfdb_record.sig_name
        signal_time = generate_signal_time(signal, fs)
        freqs, fft_values = generate_signal_spectrum(signal, fs)

        title = f"{wfdb_record_source} record {wfdb_record.record_name}"
        file_path = path / f"{wfdb_record_source.lower().split("-")[0]}_original.png"
        plot_signal_time_spectrum(signal, signal_time, fft_values, freqs, lead_names, title=title, file_path=file_path)

        signals_names = ["Original", "Filtered"]
        signals = [signal]
        signals_time = [signal_time]
        signals_fft_values = [fft_values]
        signals_freqs = [freqs]


        signal = filter_pipeline(signal, fs)
        freqs, fft_values = generate_signal_spectrum(signal, fs)

        signals.append(signal)
        signals_time.append(signal_time)
        signals_fft_values.append(fft_values)
        signals_freqs.append(freqs)

        resample_fs = 500
        signal = resample_polyphase(signal, fs, resample_fs)
        signal_time = generate_signal_time(signal, resample_fs)
        freqs, fft_values = generate_signal_spectrum(signal, resample_fs)

        title = f"{wfdb_record_source} record: {wfdb_record.record_name} Filtered and Resampled to {resample_fs}Hz"
        file_path = path / f"{wfdb_record_source.lower().split("-")[0]}_filt_{resample_fs}hz.png"
        plot_signal_time_spectrum(signal, signal_time, fft_values, freqs, lead_names, title=title, file_path=file_path)

        signals.append(signal)
        signals_time.append(signal_time)
        signals_fft_values.append(fft_values)
        signals_freqs.append(freqs)
        signals_names.append(f"Filtered Resampled {resample_fs}Hz")

        title = f"{wfdb_record_source} record {wfdb_record.record_name}"
        file_path = path / f"{wfdb_record_source.lower().split("-")[0]}_data_pipeline_overlapped_{resample_fs}hz.png"
        plot_signal_time_spectrum_overlapped(signals, signals_time, signals_fft_values, signals_freqs, signals_names, lead_names, title=title, file_path=file_path)


        signals.pop()
        signals_time.pop()
        signals_fft_values.pop()
        signals_freqs.pop()
        signals_names.pop()

        signal = signals[-1]
        resample_fs = 100
        signal = resample_polyphase(signal, fs, resample_fs)
        signal_time = generate_signal_time(signal, resample_fs)
        freqs, fft_values = generate_signal_spectrum(signal, resample_fs)

        title = f"{wfdb_record_source} record: {wfdb_record.record_name} Filtered and Resampled to {resample_fs}Hz"
        file_path = path / f"{wfdb_record_source.lower().split("-")[0]}_filt_{resample_fs}hz.png"
        plot_signal_time_spectrum(signal, signal_time, fft_values, freqs, lead_names, title=title, file_path=file_path)

        signals.append(signal)
        signals_time.append(signal_time)
        signals_fft_values.append(fft_values)
        signals_freqs.append(freqs)
        signals_names.append(f"Filtered Resampled {resample_fs}Hz")

        title = f"{wfdb_record_source} record {wfdb_record.record_name}"
        file_path = path / f"{wfdb_record_source.lower().split("-")[0]}_data_pipeline_overlapped_{resample_fs}hz.png"
        plot_signal_time_spectrum_overlapped(signals, signals_time, signals_fft_values, signals_freqs, signals_names, lead_names, title=title, file_path=file_path)

    # Outlier and Normal ECG
    plot_wfdb_record("data/training_data/ptbxl_500_wfdb/00000/00369_hr", path / "normal_ptb_ecg.png", title = "Normal PTB-XL Record")
    plot_wfdb_record("data/training_data/code15_wfdb/exams_part15/2837969", path / "outlier_code_ecg.png", title = "Outlier from CODE-15%")

    path = Path("assets/demographic_analysis/")
    path.mkdir(parents=True, exist_ok=True)

    data_path = Path("data/training_data/")
    meta_data_path = data_path / "meta_data"

    meta_data = pd.read_csv(
        meta_data_path / "meta_data.csv",
        dtype={"record_name": str, "base_time": str, "base_date": str},
    )
    meta_data = meta_data[meta_data.sig_len >= 1500]
    metrics = pd.read_csv(meta_data_path / "channel_metrics.csv").drop("source", axis=1)
    meta_data = meta_data.merge(metrics, on="data_path")


    Q1 = meta_data.filter(regex="pp_amp").quantile(0.25)
    Q3 = meta_data.filter(regex="pp_amp").quantile(0.75)
    IQR = Q3 - Q1

    for numeric_col in meta_data.filter(regex="pp_amp").columns:
        meta_data = meta_data.loc[
            (meta_data[numeric_col] >= Q1[numeric_col] - 1.5 * IQR[numeric_col])
            & (meta_data[numeric_col] <= Q3[numeric_col] + 1.5 * IQR[numeric_col])
        ]

    db_exams = pd.read_csv(meta_data_path / "db_exams.csv", low_memory=False)

    meta_data = pd.merge(
        meta_data,
        db_exams,
        how="left",
        left_on="record_name",
        right_on="exam_id",
        validate="1:1",
    )
    meta_data, _ = add_demographic_dummies(meta_data)
    demo_list = [meta_data.loc[meta_data["source"] == "SaMi-Trop"], meta_data.loc[meta_data["source"] == "PTB-XL"], meta_data.loc[(meta_data["source"] == "CODE-15%") & (meta_data["chagas_label"]==True)], meta_data.loc[(meta_data["source"] == "CODE-15%") & (meta_data["chagas_label"]==False)]]#, meta_data.loc[meta_data["source"] == "CODE-15%"]]
    title = "Age Group Distribution"
    sub_title_list = ["SaMi-Trop", "PTB-XL", "Chagas Positive CODE-15%", "Chagas Negative CODE-15%"]#, "CODE-15%"]
    plot_demographics_count(demo_list, column="age_groups", sub_title_list = sub_title_list, title = title, file_path=path / "age_group.png")
    title = "Patient Gender Distribution"
    plot_demographics_count(demo_list, column="sex", sub_title_list = sub_title_list, title = title, file_path=path / "gender.png")

    # TODO add ECG plots mean std (time, freq) per database, etc...

if __name__ == "__main__":
    main()
