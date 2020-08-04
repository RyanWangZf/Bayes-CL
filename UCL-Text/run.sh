# UCL
# srun --gres=gpu:1 python -u main_ucl.py main --data_name="20ng" --bnn="snr" --batch_size=64 --lr=1e-3
# srun --gres=gpu:1 python -u main_ucl.py main --data_name="R52" --bnn="snr" --batch_size=64 --lr=1e-3
# srun --gres=gpu:1 python -u main_ucl.py main --data_name="ohsumed" --bnn="mix" --batch_size=64 --lr=1e-3
# srun --gres=gpu:1 python -u main_ucl.py main --data_name="mr" --bnn="snr" --batch_size=64 --lr=1e-3

# UCL-TL
# srun --gres=gpu:1 python -u main_ucl_tl.py main --data_name="20ng" --bnn="snr" --batch_size=64 --lr=1e-3
# srun --gres=gpu:1 python -u main_ucl_tl.py main --data_name="R52" --bnn="mix" --batch_size=64 --lr=1e-3 --num_epoch=100
# srun --gres=gpu:1 python -u main_ucl_tl.py main --data_name="ohsumed" --bnn="snr" --batch_size=64 --lr=1e-3
# srun --gres=gpu:1 python -u main_ucl_tl.py main --data_name="mr" --bnn="snr" --batch_size=64 --lr=1e-3

# SPL
# srun --gres=gpu:1 python -u main_spl.py main --data_name="20ng" --batch_size=64 --lr=1e-3 --spl="spl"
# srun --gres=gpu:1 python -u main_spl.py main --data_name="R52" --batch_size=64 --lr=1e-3 --spl="spl"
# srun --gres=gpu:1 python -u main_spl.py main --data_name="ohsumed" --batch_size=64 --lr=1e-3 --spl="spl"
# srun --gres=gpu:1 python -u main_spl.py main --data_name="mr" --batch_size=64 --lr=1e-3 --spl="spl"

# SPCL
# srun --gres=gpu:1 python -u main_spl.py main --data_name="20ng" --batch_size=64 --lr=1e-3 --spl="spcl"
# srun --gres=gpu:1 python -u main_spl.py main --data_name="R52" --batch_size=64 --lr=1e-3 --spl="spcl"
# srun --gres=gpu:1 python -u main_spl.py main --data_name="ohsumed" --batch_size=64 --lr=1e-3 --spl="spcl"
# srun --gres=gpu:1 python -u main_spl.py main --data_name="mr" --batch_size=64 --lr=1e-3 --spl="spcl"

# SPL-IR
# srun --gres=gpu:1 python -u main_spl.py main --data_name="20ng" --batch_size=64 --lr=1e-3 --spl="splir"
# srun --gres=gpu:1 python -u main_spl.py main --data_name="R52" --batch_size=64 --lr=1e-3 --spl="splir"
# srun --gres=gpu:1 python -u main_spl.py main --data_name="ohsumed" --batch_size=64 --lr=1e-3 --spl="splir"
# srun --gres=gpu:1 python -u main_spl.py main --data_name="mr" --batch_size=64 --lr=1e-3 --spl="splir"

# CL-TL
# srun --gres=gpu:1 python -u main_cltl.py main --data_name="20ng" --batch_size=64 --lr=1e-3
# srun --gres=gpu:1 python -u main_cltl.py main --data_name="R52" --batch_size=32 --lr=1e-3
# srun --gres=gpu:1 python -u main_cltl.py main --data_name="ohsumed" --batch_size=64 --lr=1e-3
# srun --gres=gpu:1 python -u main_cltl.py main --data_name="mr" --batch_size=64 --lr=1e-3

# curriculumNet
# srun --gres=gpu:1 python -u main_curriculumNet.py main --data_name="20ng" --batch_size=64 --lr=1e-3
# srun --gres=gpu:1 python -u main_curriculumNet.py main --data_name="R52" --batch_size=64 --lr=1e-3
# srun --gres=gpu:1 python -u main_curriculumNet.py main --data_name="ohsumed" --batch_size=64 --lr=1e-3
# srun --gres=gpu:1 python -u main_curriculumNet.py main --data_name="mr" --batch_size=64 --lr=1e-3

# data param
# srun --gres=gpu:1 python -u main_dataparam.py main --data_name="20ng" --batch_size=64 --lr=1e-3
# srun --gres=gpu:1 python -u main_dataparam.py main --data_name="R52" --batch_size=64 --lr=1e-3
# srun --gres=gpu:1 python -u main_dataparam.py main --data_name="ohsumed" --batch_size=64 --lr=1e-3
# srun --gres=gpu:1 python -u main_dataparam.py main --data_name="mr" --batch_size=64 --lr=1e-3

# mentornet
# srun --gres=gpu:1 python -u main_mentornet.py main_train_mentornet --data_name="20ng" --use_gpu=True --batch_size=64 --lr=1e-3 --noise_ratio=0.2
# srun --gres=gpu:1 python -u main_mentornet.py main --data_name="20ng" --use_gpu=True --batch_size=64 --lr=1e-3 --noise_ratio=0.0

# srun --gres=gpu:1 python -u main_mentornet.py main_train_mentornet --data_name="R52" --use_gpu=True --batch_size=64 --lr=1e-3 --noise_ratio=0.2
# srun --gres=gpu:1 python -u main_mentornet.py main --data_name="R52" --use_gpu=True --batch_size=64 --lr=1e-3 --noise_ratio=0.0

# srun --gres=gpu:1 python -u main_mentornet.py main_train_mentornet --data_name="ohsumed" --use_gpu=True --batch_size=64 --lr=1e-3 --noise_ratio=0.2
# srun --gres=gpu:1 python -u main_mentornet.py main --data_name="ohsumed" --use_gpu=True --batch_size=64 --lr=1e-3 --noise_ratio=0.0 --weight_decay=1e-4

# srun --gres=gpu:1 python -u main_mentornet.py main_train_mentornet --data_name="mr" --use_gpu=True --batch_size=64 --lr=1e-3 --noise_ratio=0.2
srun --gres=gpu:1 python -u main_mentornet.py main --data_name="mr" --use_gpu=True --batch_size=64 --lr=1e-3 --noise_ratio=0.0
