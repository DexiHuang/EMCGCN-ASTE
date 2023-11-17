#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python main.py --mode train --dataset res14 --batch_size 16 --epochs 100 --model_dir savemodel/ --seed 42 --pooling avg --prefix ../data/D2/ --encoder_model bert
CUDA_VISIBLE_DEVICES=0 python main.py --mode train --dataset res14 --batch_size 16 --epochs 100 --model_dir savemodel/ --seed 70 --pooling avg --prefix ../data/D2/ --encoder_model roberta --bert_lr 1e-3
# CUDA_VISIBLE_DEVICES=0 python main.py --mode test --dataset lap14 --batch_size 16 --epochs 100 --model_dir savemodel/ --seed 1000 --pooling avg --prefix ../data/D2/