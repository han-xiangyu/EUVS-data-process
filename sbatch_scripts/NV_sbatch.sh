
job_name="3DfoundationModelDataProcessNew"
training_logs_dir="/lustre/fsw/portfolios/nvr/users/ymingli/gaussian/logs/long_video_spatial05_sampling6000"

## GPU job

submit_job --gpu 1 --cpu 32 --nodes 1 --partition=grizzly,polar,polar3,polar4 --account=nvr_av_end2endav \
                --image=/lustre/fsw/portfolios/nvr/users/ymingli/dockers/cu118.sqsh  \
                --mounts=/lustre/:/lustre/,/lustre/fsw/portfolios/nvr/users/ymingli/miniconda3:/home/ymingli/miniconda3 \
                --duration 4 \
                --dependency=singleton \
                --name $job_name \
                --logdir $training_logs_dir \
                --notimestamp \
                --command  "bash /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/code/EUVS-data-process/sbatch_scripts/NV_run.sh"


# ## CPU only job

# submit_job --gpu 0 --cpu 24 --mem 64 \
#     --account nvr_av_foundations --partition cpu_long \
#     --tasks_per_node=1 --duration 168 \
#     --image=/lustre/fsw/portfolios/nvr/users/ymingli/dockers/cu118.sqsh \
#     --mounts=/lustre/:/lustre/,/lustre/fsw/portfolios/nvr/users/ymingli/miniconda3:/home/ymingli/miniconda3 \
#     --dependency=singleton \
#     --name $job_name\
#     --logdir  $training_logs_dir \
#     --notimestamp \
#     --command "bash /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/code/EUVS-data-process/sbatch_scripts/NV_run.sh"
