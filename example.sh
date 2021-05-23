python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 256 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0  --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,120,150,190,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.99 --quantize_level 16 --is_biased True \
    --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True
python mailme.py --message "done"

# python run.py \
#     --arch mlp --optimizer parallel_choco_v \
#     --units 1024 \
#     --avg_model True --experiment test \
#     --data cifar10 --pin_memory False \
#     --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
#     --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
#     --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0  --on_cuda False --use_ipc False --comm_device cpu \
#     --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
#     --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
#     --save_some_models 50,100,150,190,200,217,300 \
#     --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
#     --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.875 --quantize_level 16 --is_biased True \
#     --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
#     --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True
# python mailme.py --message "1024 875"


# python run.py \
#     --arch mlp --optimizer parallel_choco_v \
#     --units 512 \
#     --avg_model True --experiment test \
#     --data cifar10 --pin_memory False \
#     --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
#     --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
#     --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0  --on_cuda False --use_ipc False --comm_device cpu \
#     --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
#     --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
#     --save_some_models 50,100,150,190,200,217,300 \
#     --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
#     --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.75 --quantize_level 16 --is_biased True \
#     --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
#     --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True
# python mailme.py --message "512 75"


# python run.py \
#     --arch mlp --optimizer parallel_choco_v \
#     --units 256 \
#     --avg_model True --experiment test \
#     --data cifar10 --pin_memory False \
#     --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
#     --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
#     --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0  --on_cuda False --use_ipc False --comm_device cpu \
#     --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
#     --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
#     --save_some_models 50,100,150,190,200,217,300 \
#     --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
#     --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.5 --quantize_level 16 --is_biased True \
#     --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
#     --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True
# python mailme.py --message "256 50"


# python run.py \
#     --arch mlp --optimizer parallel_choco_v \
#     --units 128 \
#     --avg_model True --experiment test \
#     --data cifar10 --pin_memory False \
#     --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
#     --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
#     --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0  --on_cuda False --use_ipc False --comm_device cpu \
#     --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
#     --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
#     --save_some_models 50,100,150,190,200,217,300 \
#     --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
#     --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.0 --quantize_level 16 --is_biased True \
#     --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
#     --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True
# python mailme.py --message "done"


# # get 8 nodes 256 0.99 from data

# python run.py \
#     --arch mlp --optimizer parallel_choco_v \
#     --units 256 \
#     --avg_model True --experiment test \
#     --data cifar10 --pin_memory False \
#     --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
#     --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
#     --n_mpi_process 12 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
#     --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
#     --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
#     --save_some_models 50,100,150,200,217,300 \
#     --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
#     --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.99 --quantize_level 16 --is_biased True \
#     --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
#     --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True
# python mailme.py --message "256 12 nodes"

# python run.py \
#     --arch mlp --optimizer parallel_choco_v \
#     --units 256 \
#     --avg_model True --experiment test \
#     --data cifar10 --pin_memory False \
#     --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
#     --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
#     --n_mpi_process 16 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
#     --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
#     --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
#     --save_some_models 50,100,150,200,217,300 \
#     --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
#     --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.99 --quantize_level 16 --is_biased True \
#     --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
#     --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True
# python mailme.py --message "256 16 nodes"

# python run.py \
#     --arch mlp --optimizer parallel_choco_v \
#     --units 256 \
#     --avg_model True --experiment test \
#     --data cifar10 --pin_memory False \
#     --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
#     --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
#     --n_mpi_process 20 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
#     --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
#     --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
#     --save_some_models 50,100,150,200,217,300 \
#     --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
#     --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.99 --quantize_level 16 --is_biased True \
#     --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
#     --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True
# python mailme.py --message "256 20 nodes"