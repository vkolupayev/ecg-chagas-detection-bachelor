#!/bin/bash
#SBATCH -o ft-founder-tm.sh.out
#SBATCH -p gpu
#SBATCH -n 6
#SBATCH --gres gpu
#SBATCH --time=80:00:00

singularity exec --nv python_3_13.sif bash -c "
python -m venv venv &&
source venv/bin/activate &&
pip install -r requirements.txt &&
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m experiment_run \
--config ./configs/founder_fine_tune_tm.yaml
"