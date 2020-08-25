# srun --gres=gpu:1 python -u main_bnn_onestep.py main --use_gpu=True --bnn="alea" --batch_size=64 --lr=1e-3 
# srun --gres=gpu:1 python -u main_bnn_onestep.py main --data_name="cifar100" --bnn="alea" --use_gpu=True --batch_size=32 --lr=1e-3
# srun --gres=gpu:1 python -u main_bnn_onestep.py main --data_name="stl10" --bnn="alea" --use_gpu=True --batch_size=32 --lr=1e-3
# srun --gres=gpu:1 python -u main_bnn_onestep.py main --data_name="svhn" --bnn="alea" --use_gpu=True --batch_size=32 --lr=1e-3

# srun --gres=gpu:1 python -u main_bnn_onestep.py main --use_gpu=True --bnn="epis" --batch_size=64 --lr=1e-3 > cifar10.log &
# srun --gres=gpu:1 python -u main_bnn_onestep.py main --data_name="cifar100" --bnn="epis" --use_gpu=True --batch_size=32 --lr=1e-3 > cifar100.log &
# srun --gres=gpu:1 python -u main_bnn_onestep.py main --data_name="stl10" --bnn="epis" --use_gpu=True --batch_size=32 --lr=1e-3 > stl10.log &
# srun --gres=gpu:1 python -u main_bnn_onestep.py main --data_name="svhn" --bnn="epis" --use_gpu=True --batch_size=32 --lr=1e-3 > svhn.log &

# srun --gres=gpu:1 python -u main_bnn_onestep.py main --use_gpu=True --bnn="mix" --batch_size=64 --lr=1e-3 > cifar10.log &
# srun --gres=gpu:1 python -u main_bnn_onestep.py main --data_name="cifar100" --bnn="mix" --use_gpu=True --batch_size=32 --lr=1e-3 > cifar100.log &
# srun --gres=gpu:1 python -u main_bnn_onestep.py main --data_name="stl10" --bnn="mix" --use_gpu=True --batch_size=32 --lr=1e-3 > stl10.log &
# srun --gres=gpu:1 python -u main_bnn_onestep.py main --data_name="svhn" --bnn="mix" --use_gpu=True --batch_size=32 --lr=1e-3 > svhn.log &

# srun --gres=gpu:1 python -u main_bnn_onestep.py main --use_gpu=True --bnn="snr" --batch_size=64 --lr=1e-3 > cifar10.log &
# srun --gres=gpu:1 python -u main_bnn_onestep.py main --data_name="cifar100" --bnn="snr" --use_gpu=True --batch_size=32 --lr=1e-3 > cifar100.log &
# srun --gres=gpu:1 python -u main_bnn_onestep.py main --data_name="stl10" --bnn="snr" --use_gpu=True --batch_size=32 --lr=1e-3 > stl10.log &
# srun --gres=gpu:1 python -u main_bnn_onestep.py main --data_name="svhn" --bnn="snr" --use_gpu=True --batch_size=32 --lr=1e-3 > svhn.log &
