#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python3 -u ./text_generation/run_pubmed_small.py --train_filepath "./data/pubmed/xxx.csv"
