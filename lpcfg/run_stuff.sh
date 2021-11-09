#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_fast.py --temperature 1 --train_file data/ptb-train.pkl --val_file data/ptb-val.pkl --band 0 --save_path 30_60_2e3 --z_dim 0 --model_type 4th --num_epochs 15 --lr 2e-3 --num_features 16 --no_argmax True --nt_states 30 --t_states 60 --accumulate 4
CUDA_VISIBLE_DEVICES=0 python train_fast.py --temperature 1 --train_file data/ptb-train.pkl --val_file data/ptb-val.pkl --band 0 --save_path 30_60_rff_features_16_2e3 --z_dim 0 --model_type 19th --num_epochs 15 --lr 2e-3 --num_features 8 --no_argmax True --nt_states 30 --t_states 60 --accumulate 4
CUDA_VISIBLE_DEVICES=0 python train_fast.py --temperature 1 --train_file data/ptb-train.pkl --val_file data/ptb-val.pkl --band 0 --save_path 30_60_rff_features_16_2e3 --z_dim 0 --model_type 19th --num_epochs 15 --lr 2e-3 --num_features 16 --no_argmax True --nt_states 30 --t_states 60 --accumulate 4

CUDA_VISIBLE_DEVICES=0 python train_fast.py --temperature 1 --train_file data/ptb-train.pkl --val_file data/ptb-val.pkl --band 0 --save_path 60_120_2e3 --z_dim 0 --model_type 4th --num_epochs 15 --lr 2e-3 --num_features 16 --no_argmax True --nt_states 60 --t_states 120 --accumulate 4
CUDA_VISIBLE_DEVICES=0 python train_fast.py --temperature 1 --train_file data/ptb-train.pkl --val_file data/ptb-val.pkl --band 0 --save_path 60_120_rff_features_16_2e3 --z_dim 0 --model_type 19th --num_epochs 15 --lr 2e-3 --num_features 16 --no_argmax True --nt_states 60 --t_states 120 --accumulate 4
CUDA_VISIBLE_DEVICES=0 python train_fast.py --temperature 1 --train_file data/ptb-train.pkl --val_file data/ptb-val.pkl --band 0 --save_path 60_120_rff_features_32_2e3 --z_dim 0 --model_type 19th --num_epochs 15 --lr 2e-3 --num_features 32 --no_argmax True --nt_states 60 --t_states 120 --accumulate 4

CUDA_VISIBLE_DEVICES=0 python train_fast.py --temperature 1 --train_file data/ptb-train.pkl --val_file data/ptb-val.pkl --band 0 --save_path 100_200_2e3 --z_dim 0 --model_type 4th --num_epochs 15 --lr 2e-3 --num_features 32 --no_argmax True --nt_states 100 --t_states 200 --accumulate 4
CUDA_VISIBLE_DEVICES=0 python train_fast.py --temperature 1 --train_file data/ptb-train.pkl --val_file data/ptb-val.pkl --band 0 --save_path 100_200_rff_features_32_2e3 --z_dim 0 --model_type 19th --num_epochs 15 --lr 2e-3 --num_features 32 --no_argmax True --nt_states 100 --t_states 200 --accumulate 4
CUDA_VISIBLE_DEVICES=0 python train_fast.py --temperature 1 --train_file data/ptb-train.pkl --val_file data/ptb-val.pkl --band 0 --save_path 100_200_rff_features_64_2e3 --z_dim 0 --model_type 19th --num_epochs 15 --lr 2e-3 --num_features 64 --no_argmax True --nt_states 100 --t_states 200 --accumulate 4

