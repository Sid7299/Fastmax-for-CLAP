#!/bin/bash
#SBATCH --job-name=fclap_pretrain
#SBATCH --account=nexus
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=12:00:00
#SBATCH -J fastmax_clap_pretrain
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --output fclap_%j.out
#SBATCH --error=logs/fclap_%j.err

srun python -m src.laion_clap.training.main \
    --save-frequency 5 \
    --save-top-performance 3 \
    --save-most-recent \
    --dataset-type="webdataset" \
    --datasetpath="datasets" \
    --precision="fp32" \
    --batch-size=96 \
    --lr=1e-4 \
    --wd=0.0 \
    --epochs=45 \
    --workers=6 \
    --use-bn-sync \
    --amodel HTSAT-tiny \
    --tmodel transformer \
    --warmup 3200 \
    --datasetnames "Clotho" \
    --datasetinfos "train" \
    --top-k-checkpoint-select-dataset="Clotho-test" \
    --top-k-checkpoint-select-metric="mAP@10" \
    --logs 'logs' \
    --seed 3407 \
    --gather-with-grad \
    --optimizer "adam" \
    --data-filling "repeatpad" \
    --data-truncating "rand_trunc" \
    --pretrained-audio 'HTSAT-fullset-tiny-map=0.467.ckpt' \
    --prefetch-factor 2
