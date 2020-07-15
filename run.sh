# srun --gres=gpu:1 python -u main_nocl.py train --use_gpu=True --batch_size=128 --noise_ratio=0.4 > train_nocl.log &
# srun --gres=gpu:1 -w node02 python -u main_dclif.py train --use_gpu=True --batch_size=128 --noise_ratio=0.4 --curriculum_size=1000 > train.log &
# srun --gres=gpu:1 -w node02 python -u main_if.py train --use_gpu=True

srun --gres=gpu:1 -w node01 python -u main_if_boost.py main --use_gpu=True --batch_size=128 --lr=1e-3 --noise_ratio=0.2 --curriculum_size=10000

