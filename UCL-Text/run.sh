# UCL
# srun --gres=gpu:1 python -u main_ucl.py main --data_name="20ng" --bnn="snr" --batch_size=64 --lr=1e-3
# srun --gres=gpu:1 python -u main_ucl.py main --data_name="R52" --bnn="snr" --batch_size=64 --lr=1e-3
# srun --gres=gpu:1 python -u main_ucl.py main --data_name="ohsumed" --bnn="mix" --batch_size=64 --lr=1e-3
srun --gres=gpu:1 python -u main_ucl.py main --data_name="mr" --bnn="snr" --batch_size=64 --lr=1e-3


# SPL

# SPCL

# SPL-IR

