#!/bin/bash

HOME=/bd_targaryen/users/kleelakunwet/
cd ~
source /bd_targaryen/users/kleelakunwet/miniconda3/bin/activate
conda activate dex
cd DEX

task=$1

python eval.py task=$task n_eval_episodes=20 seed=1 use_dcbf=False render_three_views=True ckpt_dir=./exp_local/$task/DEX/d100/s1/model &> log/$task/result/s1.out
python eval.py task=$task n_eval_episodes=20 seed=1 use_dcbf=True render_three_views=True ckpt_dir=./exp_local/$task/DEX/d100/s1/model &> log/$task/result/s1_cbf.out
python eval.py task=$task n_eval_episodes=20 seed=2 use_dcbf=False render_three_views=True ckpt_dir=./exp_local/$task/DEX/d100/s1/model &> log/$task/result/s2.out
python eval.py task=$task n_eval_episodes=20 seed=2 use_dcbf=True render_three_views=True ckpt_dir=./exp_local/$task/DEX/d100/s1/model &> log/$task/result/s2_cbf.out
python eval.py task=$task n_eval_episodes=20 seed=3 use_dcbf=False render_three_views=True ckpt_dir=./exp_local/$task/DEX/d100/s1/model &> log/$task/result/s3.out
python eval.py task=$task n_eval_episodes=20 seed=3 use_dcbf=True render_three_views=True ckpt_dir=./exp_local/$task/DEX/d100/s1/model &> log/$task/result/s3_cbf.out