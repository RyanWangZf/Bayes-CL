# srun --gres=gpu:1 python -u main_nocl.py train --use_gpu=True --batch_size=128 --noise_ratio=0.4 > train_nocl.log &
# srun --gres=gpu:1 -w node02 python -u main_dclif.py train --use_gpu=True --batch_size=128 --noise_ratio=0.4 --curriculum_size=1000 > train.log &
# srun --gres=gpu:1 -w node02 python -u main_if.py train --use_gpu=True
# srun --gres=gpu:1 -w node01 python -u main_if_boost.py main --use_gpu=True --batch_size=128 --lr=1e-3 --noise_ratio=0.0 --curriculum_size=10000
# srun --gres=gpu:1 -w node01 python -u main_if_onestep.py main --use_gpu=True --batch_size=128 --lr=1e-3 --noise_ratio=0.0 --curriculum_size=10000
# srun --gres=gpu:1 python -u main_spl.py main --use_gpu=True --batch_size=64 --lr=1e-3 --spl="splir"

# our uncertainty-guided bnn method
# srun --gres=gpu:1 python -u main_bnn_onestep.py main --use_gpu=True --batch_size=64 --lr=1e-3

# curriculum learning by transfer learning
# srun --gres=gpu:1 python -u main_cltl.py main --use_gpu=True --batch_size=64 --lr=1e-3

# mentornet, first one: train mentor; second: train student and evaluate student;
# srun --gres=gpu:1 python -u main_mentornet.py main_train_mentornet --use_gpu=True --batch_size=64 --lr=1e-3 --noise_ratio=0.2
# srun --gres=gpu:1 python -u main_mentornet.py main --use_gpu=True --batch_size=32 --lr=1e-3 --noise_ratio=0.0

# curriculumNet
# srun --gres=gpu:1 python -u main_curriculumNet.py main --use_gpu=True --batch_size=32 --lr=1e-3

# differentiable CL
srun --gres=gpu:1 python -u main_dataparam_cl.py main --use_gpu=True --batch_size=64 --lr=1e-3 --weight_decay=1e-4