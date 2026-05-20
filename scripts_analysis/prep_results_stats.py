from pathlib import Path
import json

import pandas as pd
import numpy as np

filter_paths = ["arch_experiment.csv", "old", "aggregated", "stats_tests.txt"]
run_results = [path for path in Path(f"evaluations/").glob("*") if not any(filter_paths in path.name for filter_paths in filter_paths)]

run_evaluation_metrics_paths = {}
for run_path in run_results:
    run_evaluation_metrics_paths[run_path.stem] = list(
        sorted(
            run_path.glob("*/evaluation_metrics.json"),
            key=(lambda p: int(p.parent.__str__().split("_")[-1])),
        )
    )

run_evaluation_recalls = []
run_evaluation_f2s = []
run_names = []
for run_name, path_list in run_evaluation_metrics_paths.items():
    recalls = []
    f2s = []
    for path in path_list:
        with open(path, "r") as f:
            evaluation_metrics = json.load(f)
        recalls.append(evaluation_metrics["Recall"])
        try:
            f2s.append(evaluation_metrics["F2-Score"])
        except KeyError:            
            f2_k = ((2**2+1)*evaluation_metrics["Recall"]*evaluation_metrics["Precision"]) / (evaluation_metrics["Recall"] + 2**2*evaluation_metrics["Precision"])
            f2s.append(f2_k)
    run_names.append(run_name)
    run_evaluation_recalls.append(recalls)
    run_evaluation_f2s.append(f2s)

pd.DataFrame(np.array(run_evaluation_recalls).T, columns=run_names).sort_index(axis=1).to_csv("evaluations/experiments_recall.csv", index=False)
pd.DataFrame(np.array(run_evaluation_f2s).T, columns=run_names).sort_index(axis=1).to_csv("evaluations/experiments_f2.csv", index=False)