# NV cluster script

# source /home/ymingli/miniconda3/bin/activate
source /lustre/fs12/portfolios/nvr/users/ymingli/miniconda3/etc/profile.d/conda.sh
conda activate mars_data

# Force to shield site-packages
export PYTHONNOUSERSITE=1
# Clear PYTHONPATH and  LD_LIBRARY_PATH
unset PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/usr/lib/x86_64-linux-gnu


cd /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/code/EUVS-data-process

python -m city_gs_data_process.long_video