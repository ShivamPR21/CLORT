#!/bin/bash
#SBATCH --job-name="Dataloader-Testing@CLORT"
#SBATCH --partition=standard
#SBATCH --time=00:30:00

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=15
#SBATCH --mem=128gb

#SBATCH --chdir=/scratch/pandey/
#SBATCH --error=job.clort_model_train_verification_v1.err
#SBATCH --output=job.clort_model_train_verification_v1.out

module load compiler/gcc/7.3.0 compiler/intel compiler/cuda compiler/cudnn
module load python/conda-python/3.7 python/3.6
module load cmake

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/apps/cdac/DL-CondaPy3.7/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/apps/cdac/DL-CondaPy3.7/etc/profile.d/conda.sh" ]; then
        . "/home/apps/cdac/DL-CondaPy3.7/etc/profile.d/conda.sh"
    else
        export PATH="/home/apps/cdac/DL-CondaPy3.7/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate ShivamPR21
which python

data_dir=$HOME/argoverse_dataset/train_1/argoverse-tracking/train1
script_path=$HOME/research/CLORT/scripts

out_dir=$HOME/research/clort_results
mkdir -p $out_dir

#python $HOME/collect_env.py
srun python -m cProfile $script_path/dataloader_test.py $data_dir --batch_size 128 \
--n_epochs 10 --n_itr_logs 10 --log_id 0 --n_frames -1 --n_augs 5 --vis_loss_w 0.7 --pcl_loss_w 0.7 --enc_loss_w 1.0 \
--vis_lr 0.0007 --pcl_lr 0.0007 --enc_lr 0.0003 --itr_log_ln 10 --epoch_log_ln 1 --results_dir $out_dir --preload_model_path '' > $out_dir/dataloader_test.txt
