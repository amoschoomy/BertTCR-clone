#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=predict_berttcr
#SBATCH --time=2:00:00
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --account=a_kelvin_tuong
#SBATCH -e run.error
#SBATCH -o run.out


eval "$('/scratch/user/uqachoo1/miniforge3/bin/conda' 'shell.bash' 'hook')"
conda activate 3.7
# python ./Codes/BERT_embedding.py
python ./Codes/BertTCR_prediction.py --model_file /scratch/project/tcr_ml/BertTCR/TrainedModels/Pretrained_multiple_cancer_detection.pth
