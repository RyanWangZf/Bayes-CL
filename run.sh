srun --gres=gpu:1 python -u main_nocl.py train --use_gpu=True --batch_size=128 --noise_ratio=0.4 > train_nocl.log &
srun --gres=gpu:1 python -u main_dclif.py train --use_gpu=True --batch_size=128 --noise_ratio=0.4 --curriculum_size=1000 > train.log &

