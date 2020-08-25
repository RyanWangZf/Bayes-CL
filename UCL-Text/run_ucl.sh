# srun --qos=high --gres=gpu:1 python -u main_ucl.py main --data_name="20ng" --bnn="alea" --batch_size=64 --lr=1e-3 > 20ng_alea.log &
# srun --qos=high --gres=gpu:1 python -u main_ucl.py main --data_name="R52" --bnn="alea" --batch_size=64 --lr=1e-3  > r52_alea.log &
# srun  --qos=high --gres=gpu:1 python -u main_ucl.py main --data_name="ohsumed" --bnn="alea" --batch_size=64 --lr=1e-3  > ohsumed_alea.log &
# srun  --qos=high --gres=gpu:1 python -u main_ucl.py main --data_name="mr" --bnn="alea" --batch_size=64 --lr=1e-3  > mr_alea.log &

# srun --qos=high  --gres=gpu:1 python -u main_ucl.py main --data_name="20ng" --bnn="epis" --batch_size=64 --lr=1e-3 > 20ng_epis.log &
# srun --qos=high  --gres=gpu:1 python -u main_ucl.py main --data_name="R52" --bnn="epis" --batch_size=64 --lr=1e-3 > r52_epis.log &
# srun --qos=high  --gres=gpu:1 python -u main_ucl.py main --data_name="ohsumed" --bnn="epis" --batch_size=64 --lr=1e-3 > ohsumed_epis.log &
# srun --qos=high  --gres=gpu:1 python -u main_ucl.py main --data_name="mr" --bnn="epis" --batch_size=64 --lr=1e-3 > mr_epis.log &

# srun --qos=high  --gres=gpu:1 python -u main_ucl.py main --data_name="20ng" --bnn="mix" --batch_size=64 --lr=1e-3 > 20ng_mix.log &
# srun --qos=high  --gres=gpu:1 python -u main_ucl.py main --data_name="R52" --bnn="mix" --batch_size=64 --lr=1e-3 > r52_mix.log &
# srun --qos=high  --gres=gpu:1 python -u main_ucl.py main --data_name="ohsumed" --bnn="mix" --batch_size=64 --lr=1e-3 > ohsumed_mix.log &
# srun --qos=high  --gres=gpu:1 python -u main_ucl.py main --data_name="mr" --bnn="mix" --batch_size=64 --lr=1e-3 > mr_mix.log &
 
# srun --qos=high  --gres=gpu:1 python -u main_ucl.py main --data_name="20ng" --bnn="snr" --batch_size=64 --lr=1e-3 > 20ng_snr.log &
# srun --qos=high  --gres=gpu:1 python -u main_ucl.py main --data_name="R52" --bnn="snr" --batch_size=64 --lr=1e-3 > r52_snr.log &
# srun --qos=high  --gres=gpu:1 python -u main_ucl.py main --data_name="ohsumed" --bnn="snr" --batch_size=64 --lr=1e-3 > ohsumed_snr.log &
# srun --qos=high  --gres=gpu:1 python -u main_ucl.py main --data_name="mr" --bnn="snr" --batch_size=64 --lr=1e-3 > mr_snr.log &

# run once for each line
# srun --qos=high  --gres=gpu:1 python -u main_ucl.py main --data_name="ohsumed" --bnn="snr" --batch_size=32 --lr=1e-3
# srun --qos=high  --gres=gpu:1 python -u main_ucl.py main --data_name="ohsumed" --bnn="alea" --batch_size=32 --lr=1e-3
# srun --qos=high  --gres=gpu:1 python -u main_ucl.py main --data_name="ohsumed" --bnn="epis" --batch_size=32 --lr=1e-3
# srun --qos=high  --gres=gpu:1 python -u main_ucl.py main --data_name="ohsumed" --bnn="mix" --batch_size=32 --lr=1e-3

# srun --qos=high  --gres=gpu:1 python -u main_ucl.py main --data_name="R52" --bnn="alea" --batch_size=64 --lr=1e-3
# srun --qos=high  --gres=gpu:1 python -u main_ucl.py main --data_name="R52" --bnn="epis" --batch_size=64 --lr=1e-3
# srun --qos=high  --gres=gpu:1 python -u main_ucl.py main --data_name="R52" --bnn="mix" --batch_size=64 --lr=1e-3

# srun --qos=high  --gres=gpu:1 python -u main_ucl.py main --data_name="mr" --bnn="epis" --batch_size=64 --lr=1e-3
# srun --qos=high  --gres=gpu:1 python -u main_ucl.py main --data_name="mr" --bnn="mix" --batch_size=64 --lr=1e-3
# srun --qos=high  --gres=gpu:1 python -u main_ucl.py main --data_name="mr" --bnn="snr" --batch_size=64 --lr=1e-3

