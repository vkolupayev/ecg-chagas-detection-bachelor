# Electrocardiogram Signal Analysis with Deep Learning Based Methods for Chagas Disease

**Bachelor Thesis Source Code for:**

**Electrocardiogram Signal Analysis with Deep Learning Based Methods for Chagas Disease**



## Project instructions

### Get resources:

1. Clone the project:
   
        git@github.com:vkolupayev/ecg-chagas-detection-bachelor.git

2. Download PTB-XL, CODE-15%, and SaMi-Trop to `data/`. Run:

        chmod +x ./download_data.sh
        ./download_data.sh

3. Download model weights. Run:

        chmod +x ./download_weights.sh
        ./download_weights.sh


### Preparation:

Note the development of the project was done on a Ubuntu 24.04.4 LTS machine. The experiments were ran on Vilnius University mathematics and informatics faculty high performance computing graphics processing unit cluster. See [link](https://mif.vu.lt/itwiki/en:hpc) for more information.

1. Prepare development environment:

        python -m venv venv
        source venv/bin/active
        pip install -r requirements.txt
        # If profiling is needed install requirements-profiler.txt

2. Prepare PTB-XL, CODE-15%, and SaMi-Trop data. Run:

        chmod +x ./prep_data.sh
        ./prep_data.sh

3. Transfer project to HPC with `scp`.
4. Follow these steps to set up a Python 3.13 container:


        singularity build --sandbox /tmp/python_3_13_vk \
        docker://python:3.13.10
        
        mkdir /tmp/python_3_13_vk/scratch
        
        singularity exec -H ~/work/test:$HOME -w /tmp \
        python_3_13_vk pip install -r requirements.txt
        
        singularity --verbose build python_3_13.sif \
        /tmp/python_3_13_vk

        rm -rf /tmp/python_3_13_vk
        
        # note that docker could be an alternative to singularity. 
        # If VU MIF HPC is not used.

### Experiment replication
`configs/` contain the configuration for all of the experiment runs in the bachelor thesis. See `example_slurm_job.sh`, on how to specify SLURM jobs for an experiment run.

To run a `SLURM` job on HPC:

1. sbatch example_slurm_job.sh

### Experiment results replication

With an acitve python virtual environment/container, that has the dependencies installed. Run:

        python -m scripts_analysis.prep_results_stats
        python -m scripts_analysis.results_analysis
        Rscript scripts_analysis/experiments_tests.r \
        > evaluations/stats_tests.txt
        python -m scripts_analysis.generate_assets
