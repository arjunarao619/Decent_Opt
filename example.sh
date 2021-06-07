# #######################################
# ##2048
# python run.py \
#     --arch mlp --optimizer parallel_choco_v \
#     --units 2048 \
#     --avg_model True --experiment test \
#     --data cifar10 --pin_memory False \
#     --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
#     --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
#     --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
#     --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
#     --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
#     --save_some_models 50,100,150,200,217,300 \
#     --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
#     --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.9375 --quantize_level 16 --is_biased True \
#     --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
#     --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
#     --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622523687_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.9375__dataset-cifar10_unit-2048

# python run.py \
#     --arch mlp --optimizer parallel_choco_v \
#     --units 2048 \
#     --avg_model True --experiment test \
#     --data cifar10 --pin_memory False \
#     --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
#     --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
#     --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
#     --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
#     --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
#     --save_some_models 50,100,150,200,217,300 \
#     --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
#     --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.9375 --quantize_level 16 --is_biased True \
#     --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
#     --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
#     --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622547514_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.9375__dataset-cifar10_unit-2048

# python run.py \
#     --arch mlp --optimizer parallel_choco_v \
#     --units 2048 \
#     --avg_model True --experiment test \
#     --data cifar10 --pin_memory False \
#     --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
#     --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
#     --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
#     --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
#     --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
#     --save_some_models 50,100,150,200,217,300 \
#     --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
#     --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.9375 --quantize_level 16 --is_biased True \
#     --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
#     --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
#     --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622571389_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.9375__dataset-cifar10_unit-2048

# python run.py \
#     --arch mlp --optimizer parallel_choco_v \
#     --units 2048 \
#     --avg_model True --experiment test \
#     --data cifar10 --pin_memory False \
#     --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
#     --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
#     --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
#     --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
#     --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
#     --save_some_models 50,100,150,200,217,300 \
#     --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
#     --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.9375 --quantize_level 16 --is_biased True \
#     --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
#     --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
#     --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622595434_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.9375__dataset-cifar10_unit-2048

# python run.py \
#     --arch mlp --optimizer parallel_choco_v \
#     --units 2048 \
#     --avg_model True --experiment test \
#     --data cifar10 --pin_memory False \
#     --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
#     --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
#     --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
#     --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
#     --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
#     --save_some_models 50,100,150,200,217,300 \
#     --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
#     --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.9375 --quantize_level 16 --is_biased True \
#     --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
#     --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
#     --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621757828_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.9375__dataset-cifar10_unit-2048


# #######1024

# python run.py \
#     --arch mlp --optimizer parallel_choco_v \
#     --units 1024 \
#     --avg_model True --experiment test \
#     --data cifar10 --pin_memory False \
#     --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
#     --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
#     --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
#     --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
#     --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
#     --save_some_models 50,100,150,200,217,300 \
#     --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
#     --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.875 --quantize_level 16 --is_biased True \
#     --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
#     --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
#     --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621765281_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.875__dataset-cifar10_unit-1024

# python run.py \
#     --arch mlp --optimizer parallel_choco_v \
#     --units 1024 \
#     --avg_model True --experiment test \
#     --data cifar10 --pin_memory False \
#     --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
#     --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
#     --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
#     --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
#     --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
#     --save_some_models 50,100,150,200,217,300 \
#     --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
#     --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.875 --quantize_level 16 --is_biased True \
#     --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
#     --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
#     --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622555154_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.875__dataset-cifar10_unit-1024

# python run.py \
#     --arch mlp --optimizer parallel_choco_v \
#     --units 1024 \
#     --avg_model True --experiment test \
#     --data cifar10 --pin_memory False \
#     --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
#     --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
#     --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
#     --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
#     --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
#     --save_some_models 50,100,150,200,217,300 \
#     --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
#     --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.875 --quantize_level 16 --is_biased True \
#     --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
#     --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
#     --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622531242_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.875__dataset-cifar10_unit-1024

# python run.py \
#     --arch mlp --optimizer parallel_choco_v \
#     --units 1024 \
#     --avg_model True --experiment test \
#     --data cifar10 --pin_memory False \
#     --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
#     --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
#     --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
#     --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
#     --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
#     --save_some_models 50,100,150,200,217,300 \
#     --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
#     --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.875 --quantize_level 16 --is_biased True \
#     --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
#     --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
#     --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622578975_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.875__dataset-cifar10_unit-1024

# python run.py \
#     --arch mlp --optimizer parallel_choco_v \
#     --units 1024 \
#     --avg_model True --experiment test \
#     --data cifar10 --pin_memory False \
#     --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
#     --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
#     --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
#     --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
#     --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
#     --save_some_models 50,100,150,200,217,300 \
#     --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
#     --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.875 --quantize_level 16 --is_biased True \
#     --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
#     --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
#     --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622603537_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.875__dataset-cifar10_unit-1024



# ##########512


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
    --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.75 --quantize_level 16 --is_biased True \
    --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622610476_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.75__dataset-cifar10_unit-512

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
    --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.75 --quantize_level 16 --is_biased True \
    --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621772113_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.75__dataset-cifar10_unit-512

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
    --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.75 --quantize_level 16 --is_biased True \
    --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622585877_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.75__dataset-cifar10_unit-512

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
    --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.75 --quantize_level 16 --is_biased True \
    --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622562059_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.75__dataset-cifar10_unit-512

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
    --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.75 --quantize_level 16 --is_biased True \
    --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622538194_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.75__dataset-cifar10_unit-512


### 256

# python run.py \
#     --arch mlp --optimizer parallel_choco_v \
#     --units 256 \
#     --avg_model True --experiment test \
#     --data cifar10 --pin_memory False \
#     --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
#     --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
#     --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
#     --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
#     --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
#     --save_some_models 50,100,150,200,217,300 \
#     --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
#     --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.60 --quantize_level 16 --is_biased True \
#     --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
#     --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
#     --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621703643_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.6__dataset-cifar10_unit-256

# python run.py \
#     --arch mlp --optimizer parallel_choco_v \
#     --units 256 \
#     --avg_model True --experiment test \
#     --data cifar10 --pin_memory False \
#     --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
#     --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
#     --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
#     --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
#     --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
#     --save_some_models 50,100,150,200,217,300 \
#     --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
#     --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.60 --quantize_level 16 --is_biased True \
#     --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
#     --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
#     --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622368712_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.6__dataset-cifar10_unit-256

# python run.py \
#     --arch mlp --optimizer parallel_choco_v \
#     --units 256 \
#     --avg_model True --experiment test \
#     --data cifar10 --pin_memory False \
#     --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
#     --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
#     --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
#     --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
#     --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
#     --save_some_models 50,100,150,200,217,300 \
#     --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
#     --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.60 --quantize_level 16 --is_biased True \
#     --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
#     --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
#     --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622391410_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.6__dataset-cifar10_unit-256

# python run.py \
#     --arch mlp --optimizer parallel_choco_v \
#     --units 256 \
#     --avg_model True --experiment test \
#     --data cifar10 --pin_memory False \
#     --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
#     --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
#     --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
#     --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
#     --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
#     --save_some_models 50,100,150,200,217,300 \
#     --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
#     --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.60 --quantize_level 16 --is_biased True \
#     --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
#     --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
#     --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622414922_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.6__dataset-cifar10_unit-256

# python run.py \
#     --arch mlp --optimizer parallel_choco_v \
#     --units 256 \
#     --avg_model True --experiment test \
#     --data cifar10 --pin_memory False \
#     --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
#     --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
#     --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
#     --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
#     --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
#     --save_some_models 50,100,150,200,217,300 \
#     --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
#     --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.60 --quantize_level 16 --is_biased True \
#     --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
#     --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
#     --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622439989_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.6__dataset-cifar10_unit-256

# ######128

# python run.py \
#     --arch mlp --optimizer parallel_choco_v \
#     --units 128 \
#     --avg_model True --experiment test \
#     --data cifar10 --pin_memory False \
#     --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
#     --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
#     --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
#     --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
#     --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
#     --save_some_models 50,100,150,200,217,300 \
#     --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
#     --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.20 --quantize_level 16 --is_biased True \
#     --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
#     --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
#     --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621706432_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.2__dataset-cifar10_unit-128
# python run.py \
#     --arch mlp --optimizer parallel_choco_v \
#     --units 128 \
#     --avg_model True --experiment test \
#     --data cifar10 --pin_memory False \
#     --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
#     --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
#     --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
#     --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
#     --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
#     --save_some_models 50,100,150,200,217,300 \
#     --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
#     --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.20 --quantize_level 16 --is_biased True \
#     --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
#     --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
#     --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622371545_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.2__dataset-cifar10_unit-128

# python run.py \
#     --arch mlp --optimizer parallel_choco_v \
#     --units 128 \
#     --avg_model True --experiment test \
#     --data cifar10 --pin_memory False \
#     --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
#     --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
#     --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
#     --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
#     --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
#     --save_some_models 50,100,150,200,217,300 \
#     --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
#     --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.20 --quantize_level 16 --is_biased True \
#     --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
#     --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
#     --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622394233_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.2__dataset-cifar10_unit-128

# python run.py \
#     --arch mlp --optimizer parallel_choco_v \
#     --units 128 \
#     --avg_model True --experiment test \
#     --data cifar10 --pin_memory False \
#     --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
#     --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
#     --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
#     --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
#     --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
#     --save_some_models 50,100,150,200,217,300 \
#     --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
#     --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.20 --quantize_level 16 --is_biased True \
#     --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
#     --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
#     --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622417739_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.2__dataset-cifar10_unit-128

# python run.py \
#     --arch mlp --optimizer parallel_choco_v \
#     --units 128 \
#     --avg_model True --experiment test \
#     --data cifar10 --pin_memory False \
#     --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
#     --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
#     --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
#     --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
#     --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
#     --save_some_models 50,100,150,200,217,300 \
#     --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
#     --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.20 --quantize_level 16 --is_biased True \
#     --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
#     --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
#     --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622442795_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.2__dataset-cifar10_unit-128