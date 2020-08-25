# our uncertainty-guided bnn method
# srun --gres=gpu:1 --qos=high python -u main_ugcl.py main --use_gpu=True --batch_size=128 --lr=1e-3 --bnn="snr" &
# srun --gres=gpu:1 --qos=high python -u main_ugcl.py main --use_gpu=True --batch_size=128 --lr=1e-3 --bnn="alea" &
# srun --gres=gpu:1 --qos=high python -u main_ugcl.py main --use_gpu=True --batch_size=128 --lr=1e-3 --bnn="epis" &
# srun --gres=gpu:1 --qos=high python -u main_ugcl.py main --use_gpu=True --batch_size=128 --lr=1e-3 --bnn="mix" &

# srun --gres=gpu:1 --qos=high python -u main_ugcl.py main --data_name="cifar100" --use_gpu=True --batch_size=256 --lr=1e-3 --bnn="snr" &
# srun --gres=gpu:1 --qos=high python -u main_ugcl.py main --data_name="cifar100" --use_gpu=True --batch_size=128 --lr=1e-3 --bnn="alea" &
# srun --gres=gpu:1 --qos=high python -u main_ugcl.py main --data_name="cifar100" --use_gpu=True --batch_size=128 --lr=1e-3 --bnn="epis" &
# srun --gres=gpu:1 --qos=high python -u main_ugcl.py main --data_name="cifar100" --use_gpu=True --batch_size=64 --lr=1e-3 --bnn="mix" &

# srun --gres=gpu:1 --qos=high python -u main_ugcl.py main --data_name="stl10" --use_gpu=True --batch_size=128 --lr=1e-3 --bnn="snr"  &
# srun --gres=gpu:1 --qos=high python -u main_ugcl.py main --data_name="stl10" --use_gpu=True --batch_size=128 --lr=1e-3 --bnn="alea" &
# srun --gres=gpu:1 --qos=high python -u main_ugcl.py main --data_name="stl10" --use_gpu=True --batch_size=128 --lr=1e-3 --bnn="epis" &
# srun --gres=gpu:1 --qos=high python -u main_ugcl.py main --data_name="stl10" --use_gpu=True --batch_size=128 --lr=1e-3 --bnn="mix"  &

# srun --gres=gpu:1 --qos=high python -u main_ugcl.py main --data_name="svhn" --use_gpu=True --batch_size=128 --lr=1e-3 --bnn="snr" 
# srun --gres=gpu:1 --qos=high python -u main_ugcl.py main --data_name="svhn" --use_gpu=True --batch_size=128 --lr=1e-3 --bnn="alea" &
# srun --gres=gpu:1 --qos=high python -u main_ugcl.py main --data_name="svhn" --use_gpu=True --batch_size=128 --lr=1e-3 --bnn="epis" &
# srun --gres=gpu:1 --qos=high python -u main_ugcl.py main --data_name="svhn" --use_gpu=True --batch_size=128 --lr=1e-3 --bnn="mix"  &

# our uncertainty-guided bnn based on transferable uncertainty
# srun --gres=gpu:1 --qos=high python -u main_ugcl_tl.py main  --data_name="cifar10" --use_gpu=True -bnn="snr" --batch_size=128 --lr=1e-3 &
# srun --gres=gpu:1 --qos=high python -u main_ugcl_tl.py main --data_name="cifar100" --bnn="epis" --use_gpu=True --batch_size=128 --lr=1e-3 &
# srun --gres=gpu:1 --qos=high python -u main_ugcl_tl.py main --data_name="stl10" --bnn="epis" --use_gpu=True --batch_size=128 --lr=1e-3 &
# srun --gres=gpu:1 --qos=high python -u main_ugcl_tl.py main --data_name="svhn" --bnn="snr" --use_gpu=True --batch_size=128 --lr=1e-3 &

# self-paced learning
# srun --gres=gpu:1 --qos=high python -u main_spl.py main --use_gpu=True --batch_size=128 --lr=1e-3 --spl="spl" &
# srun --gres=gpu:1 --qos=high python -u main_spl.py main --data_name="cifar100" --use_gpu=True --batch_size=128 --lr=1e-3 --spl="spl" &
# srun --gres=gpu:1 --qos=high python -u main_spl.py main --data_name="stl10" --use_gpu=True --batch_size=128 --lr=1e-3 --spl="spl" &
# srun --gres=gpu:1 --qos=high python -u main_spl.py main --data_name="svhn" --use_gpu=True --batch_size=128 --lr=1e-3 --spl="spl" &

# self-paced curriculum learning
# srun --gres=gpu:1 --qos=high python -u main_spl.py main --use_gpu=True --batch_size=128 --lr=1e-3 --spl="spcl" &
# srun --gres=gpu:1 --qos=high python -u main_spl.py main --data_name="cifar100" --use_gpu=True --batch_size=128 --lr=1e-3 --spl="spcl" &
# srun --gres=gpu:1 --qos=high python -u main_spl.py main --data_name="stl10" --use_gpu=True --batch_size=128 --lr=1e-3 --spl="spcl" &
# srun --gres=gpu:1 --qos=high python -u main_spl.py main --data_name="svhn" --use_gpu=True --batch_size=128 --lr=1e-3 --spl="spcl" &

# self-paced curriculum learning - implicit regularization
# srun --gres=gpu:1 --qos=high python -u main_spl.py main --use_gpu=True --batch_size=128 --lr=1e-3 --spl="splir" &
# srun --gres=gpu:1 --qos=high python -u main_spl.py main --data_name="cifar100" --use_gpu=True --batch_size=128 --lr=1e-3 --spl="splir" &
# srun --gres=gpu:1 --qos=high python -u main_spl.py main --data_name="stl10" --use_gpu=True --batch_size=128 --lr=1e-3 --spl="splir" &
# srun --gres=gpu:1 --qos=high python -u main_spl.py main --data_name="svhn" --use_gpu=True --batch_size=128 --lr=1e-3 --spl="splir" &

# curriculum learning by transfer learning
# srun --gres=gpu:1 --qos=high python -u main_cltl.py main --use_gpu=True --batch_size=128 --lr=1e-3 &
# srun --gres=gpu:1 --qos=high python -u main_cltl.py main --data_name="cifar100" --use_gpu=True --batch_size=128 --lr=1e-3 &
# srun --gres=gpu:1 --qos=high python -u main_cltl.py main --data_name="stl10" --use_gpu=True --batch_size=128 --lr=1e-3 &
# srun --gres=gpu:1 --qos=high python -u main_cltl.py main --data_name="svhn" --use_gpu=True --batch_size=128 --lr=1e-3 &

# mentornet, first one: train mentor; second: train student and evaluate student;
# srun --gres=gpu:1 python -u main_mentornet.py main_train_mentornet --use_gpu=True --batch_size=64 --lr=1e-3 --noise_ratio=0.2
# srun --gres=gpu:1 python -u main_mentornet.py main --use_gpu=True --batch_size=128 --lr=1e-3 --noise_ratio=0.0

# srun --gres=gpu:1 python -u main_mentornet.py main_train_mentornet --data_name="cifar100" --use_gpu=True --batch_size=64 --lr=1e-3 --noise_ratio=0.2
# srun --gres=gpu:1 python -u main_mentornet.py main --data_name="cifar100" --use_gpu=True --batch_size=128 --lr=1e-3 --noise_ratio=0.0

# srun --gres=gpu:1 python -u main_mentornet.py main_train_mentornet --data_name="stl10" --use_gpu=True --batch_size=64 --lr=1e-3 --noise_ratio=0.2
# srun --gres=gpu:1 python -u main_mentornet.py main --data_name="stl10" --use_gpu=True --batch_size=128 --lr=1e-3 --noise_ratio=0.0

# srun --gres=gpu:1 python -u main_mentornet.py main_train_mentornet --data_name="svhn" --use_gpu=True --batch_size=64 --lr=1e-3 --noise_ratio=0.2
# srun --gres=gpu:1 python -u main_mentornet.py main --data_name="svhn" --use_gpu=True --batch_size=128 --lr=1e-3 --noise_ratio=0.0

# curriculumNet
# srun --gres=gpu:1 --qos=high python -u main_curriculumNet.py main --use_gpu=True --batch_size=128 --lr=1e-3 --weight_decay=1e-4  &
# srun --gres=gpu:1 --qos=high python -u main_curriculumNet.py main --data_name="cifar100" --use_gpu=True --batch_size=128 --lr=1e-3 &
# srun --gres=gpu:1 --qos=high python -u main_curriculumNet.py main --data_name="stl10" --use_gpu=True --batch_size=128 --lr=1e-3 &
# srun --gres=gpu:1 --qos=high python -u main_curriculumNet.py main --data_name="svhn" --use_gpu=True --batch_size=128 --lr=1e-3 &

# differentiable CL
# srun --gres=gpu:1 python -u main_dataparam_cl.py main --use_gpu=True --batch_size=128 --lr=1e-3 --weight_decay=1e-4 &
# srun --gres=gpu:1 python -u main_dataparam_cl.py main --data_name="cifar100" --use_gpu=True --batch_size=128 --lr=1e-3 --weight_decay=1e-5
# srun --gres=gpu:1 python -u main_dataparam_cl.py main --data_name="stl10" --use_gpu=True --batch_size=128 --lr=1e-3 --weight_decay=1e-4 &
# srun --gres=gpu:1 python -u main_dataparam_cl.py main --data_name="svhn" --use_gpu=True --batch_size=128 --lr=1e-3 --weight_decay=1e-4 &

# test # baby steps
# srun --gres=gpu:1 --qos=high python -u main_ugcl_babystep.py main \
#     --baby_step=3 \
#     --data_name="cifar10" --bnn="snr" --use_gpu=True --batch_size=128 --lr=1e-3
# srun --gres=gpu:1 --qos=high python -u main_ugcl_babystep.py main \
#     --baby_step=4 \
#     --data_name="cifar10" --bnn="snr" --use_gpu=True --batch_size=128 --lr=1e-3
# srun --gres=gpu:1 --qos=high python -u main_ugcl_babystep.py main \
#     --baby_step=5 \
#     --data_name="cifar10" --bnn="snr" --use_gpu=True --batch_size=128 --lr=1e-3

# srun --gres=gpu:1 --qos=high python -u main_ugcl_babystep.py main \
#     --baby_step=3 \
#     --data_name="cifar100" --bnn="mix" --use_gpu=True --batch_size=128 --lr=1e-3
# srun --gres=gpu:1 --qos=high python -u main_ugcl_babystep.py main \
#     --baby_step=4 \
#     --data_name="cifar100" --bnn="mix" --use_gpu=True --batch_size=128 --lr=1e-3
# srun --gres=gpu:1 --qos=high python -u main_ugcl_babystep.py main \
#     --baby_step=5 \
#     --data_name="cifar100" --bnn="mix" --use_gpu=True --batch_size=128 --lr=1e-3

