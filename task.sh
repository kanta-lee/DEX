HOME=/bd_targaryen/users/kleelakunwet/
cd ~
source miniconda3/bin/activate
conda activate dex
cd DEX
python train.py task=$1 agent=dex use_wb=False seed=$2
