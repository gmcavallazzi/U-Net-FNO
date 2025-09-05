#!/bin/bash
#SBATCH -D /users/addh496/sharedscratch/U-Net-FNO
#SBATCH -J unet-FNO
#SBATCH --partition=preemptgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=R-%x.%j.out
#SBATCH --gres=gpu:2

# Get the git branch from command-line argument or use spectral-norm as default
GIT_BRANCH=${1:-master}
echo "Target git branch: $GIT_BRANCH"

# Load environment setup
flight env activate gridware
module load apps/nvhpc/23.9
module load libs/nvidia-cuda/11.1.1/bin
module load cudnn/8.5.0

# Git repository handling
if [ -d ".git" ]; then
    echo "----------------------------------------"
    echo "Git repository detected"
    current_branch=$(git rev-parse --abbrev-ref HEAD)
    
    if [ "$current_branch" != "$GIT_BRANCH" ]; then
        echo "Switching to branch: $GIT_BRANCH"
        git checkout $GIT_BRANCH
    fi
    
    echo "Running on branch: $(git rev-parse --abbrev-ref HEAD)"
    echo "Latest commit: $(git log -1 --oneline)"
    echo "----------------------------------------"
fi

# Conda setup
__conda_setup="$('/users/addh496/sharedscratch/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/users/addh496/sharedscratch/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/users/addh496/sharedscratch/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/users/addh496/sharedscratch/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup

# Activate torch_modern environment instead of cfd_gan
conda activate torch_modern

# Set up LD_PRELOAD for libstdc++ compatibility
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Set PYTHONPATH
export PYTHONPATH=$(python -c "import sys; print(':'.join(sys.path))")

# Print debug information
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Total tasks: $SLURM_NTASKS"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Current conda environment: $CONDA_PREFIX"
echo "LD_PRELOAD: $LD_PRELOAD"

# Check GPU allocation in Slurm
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_GPUS: $SLURM_GPUS"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"

# Check if NVIDIA drivers are working
nvidia-smi

# Verify PyTorch can see the GPU
echo "Checking PyTorch GPU access:"
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU count:', torch.cuda.device_count()); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# Run your main script
echo "Running main script..."
#python test_model.py
python main_unet_fno.py
#python main_unet_fno.py --resume ./results/run_*/models/best_model.pt
