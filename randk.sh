python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 2048 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.99 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1623024890_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.99__dataset-cifar10_unit-2048

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 2048 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.99 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1623101269_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.99__dataset-cifar10_unit-2048

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 2048 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.99 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1623166478_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.99__dataset-cifar10_unit-2048

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 2048 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.99 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1623229891_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.99__dataset-cifar10_unit-2048


##1024

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 1024 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.98 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1623026403_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.98__dataset-cifar10_unit-1024

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 1024 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.98 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1623102743_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.98__dataset-cifar10_unit-1024

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 1024 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.98 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1623167748_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.98__dataset-cifar10_unit-1024

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 1024 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.98 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1623231162_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.98__dataset-cifar10_unit-1024

## 512

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 512 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.96 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1623029037_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.96__dataset-cifar10_unit-512

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 512 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.96 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1623104612_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.96__dataset-cifar10_unit-512

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 512 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.96 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1623169263_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.96__dataset-cifar10_unit-512

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 512 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.96 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1623232874_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.96__dataset-cifar10_unit-512

#256

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 256 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.92 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1623236695_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.92__dataset-cifar10_unit-256

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 256 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.92 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1623033600_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.92__dataset-cifar10_unit-256

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 256 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.92 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1623109141_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.92__dataset-cifar10_unit-256

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 256 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.92 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1623173070_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.92__dataset-cifar10_unit-256

## 128

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 128 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.84 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1623036649_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.84__dataset-cifar10_unit-128

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 128 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.84 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1623112198_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.84__dataset-cifar10_unit-128

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 128 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.84 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1623175582_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.84__dataset-cifar10_unit-128

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 128 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.84 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1623239189_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.84__dataset-cifar10_unit-128







































#93
python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 2048 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.9375 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622834931_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.9375__dataset-cifar10_unit-2048

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 2048 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.9375 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622874052_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.9375__dataset-cifar10_unit-2048

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 2048 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.9375 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622911385_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.9375__dataset-cifar10_unit-2048

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 2048 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.9375 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622949284_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.9375__dataset-cifar10_unit-2048

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 2048 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.9375 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622986889_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.9375__dataset-cifar10_unit-2048

#875


python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 1024 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.875 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622851513_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.875__dataset-cifar10_unit-1024

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 1024 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.875 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622889851_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.875__dataset-cifar10_unit-1024

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 1024 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.875 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622927413_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.875__dataset-cifar10_unit-1024

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 1024 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.875 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622965158_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.875__dataset-cifar10_unit-1024

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 1024 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.875 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1623002978_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.875__dataset-cifar10_unit-1024


#75

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 512 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.75 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622860815_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.75__dataset-cifar10_unit-512

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 512 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.75 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622898212_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.75__dataset-cifar10_unit-512

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 512 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.75 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622936059_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.75__dataset-cifar10_unit-512

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 512 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.75 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622973525_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.75__dataset-cifar10_unit-512

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 512 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.75 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1623011501_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.75__dataset-cifar10_unit-512

## 256

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 256 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.50 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622866630_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.5__dataset-cifar10_unit-256

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 256 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.50 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622904104_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.5__dataset-cifar10_unit-256

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 256 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.50 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622942044_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.5__dataset-cifar10_unit-256

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 256 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.50 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622979522_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.5__dataset-cifar10_unit-256

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 256 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.50 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1623017607_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.5__dataset-cifar10_unit-256

## 128

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 128 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.0 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622870742_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.0__dataset-cifar10_unit-128

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 128 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.0 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622908149_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.0__dataset-cifar10_unit-128

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 128 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.0 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622946070_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.0__dataset-cifar10_unit-128

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 128 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.0 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622983623_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.0__dataset-cifar10_unit-128

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 128 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_random_k --consensus_stepsize 0.0375 --compress_ratio 0.0 --quantize_level 16 --is_biased True \
    --hostfile /home/aarao8/choco_2/ChocoSGD/dl_code/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1623021654_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.0__dataset-cifar10_unit-128

