
job_name="3DfoundationModelDataProcess2"
training_logs_dir="/lustre/fsw/portfolios/nvr/users/ymingli/gaussian/logs/data_process_downsample0.2"

## GPU job

# submit_job --cpu 24 --nodes 1 --partition=grizzly,polar,polar3,polar4 --account=nvr_av_end2endav \
#                 --image=/lustre/fsw/portfolios/nvr/users/ymingli/dockers/cu118.sqsh  \
#                 --mounts=/lustre/:/lustre/,/lustre/fsw/portfolios/nvr/users/ymingli/miniconda3:/home/ymingli/miniconda3 \
#                 --duration 4 \
#                 --dependency=singleton \
#                 --name $job_name \
#                 --logdir $training_logs_dir \
#                 --notimestamp \
#                 --command  "bash NV_run.sh"


## CPU only job

submit_job --gpu 0 --cpu 24 --mem 64 \
    --account nvr_av_foundations --partition cpu_long \
    --tasks_per_node=1 --duration 168 \
    --image=/lustre/fsw/portfolios/nvr/users/ymingli/dockers/cu118.sqsh \
    --mounts=/lustre/:/lustre/,/lustre/fsw/portfolios/nvr/users/ymingli/miniconda3:/home/ymingli/miniconda3 \
    --dependency=singleton \
    --name $job_name\
    --logdir  $training_logs_dir \
    --notimestamp \
    --command "bash NV_run.sh"
