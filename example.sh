

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 256 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 16 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.8 --quantize_level 16 --is_biased True \
    --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/resume/1621987447_lr-0.1_epochs-300_batchsize-128_num_mpi_process_16_topology-ring_comm_info-compress_top_k-0.8__dataset-cifar10_unit-256


python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 256 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 20 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.8 --quantize_level 16 --is_biased True \
    --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/resume/1621990035_lr-0.1_epochs-300_batchsize-128_num_mpi_process_20_topology-ring_comm_info-compress_top_k-0.8__dataset-cifar10_unit-256

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 256 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 16 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.8 --quantize_level 16 --is_biased True \
    --hostfile hostfile --graph_topology torus --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/resume/1622000952_lr-0.1_epochs-300_batchsize-128_num_mpi_process_16_topology-torus_comm_info-compress_top_k-0.8__dataset-cifar10_unit-256


python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 256 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 20 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.8 --quantize_level 16 --is_biased True \
    --hostfile hostfile --graph_topology torus --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/resume/1622003595_lr-0.1_epochs-300_batchsize-128_num_mpi_process_20_topology-torus_comm_info-compress_top_k-0.8__dataset-cifar10_unit-256


python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 256 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 16 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.8 --quantize_level 16 --is_biased True \
    --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/resume/1622013453_lr-0.1_epochs-300_batchsize-128_num_mpi_process_16_topology-ring_comm_info-compress_top_k-0.8__dataset-cifar10_unit-256


python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 256 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 20 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.8 --quantize_level 16 --is_biased True \
    --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/resume/1622016076_lr-0.1_epochs-300_batchsize-128_num_mpi_process_20_topology-ring_comm_info-compress_top_k-0.8__dataset-cifar10_unit-256



python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 256 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 16 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.8 --quantize_level 16 --is_biased True \
    --hostfile hostfile --graph_topology torus --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/resume/1622026019_lr-0.1_epochs-300_batchsize-128_num_mpi_process_16_topology-torus_comm_info-compress_top_k-0.8__dataset-cifar10_unit-256


python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 256 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 20 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.8 --quantize_level 16 --is_biased True \
    --hostfile hostfile --graph_topology torus --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/resume/1622028727_lr-0.1_epochs-300_batchsize-128_num_mpi_process_20_topology-torus_comm_info-compress_top_k-0.8__dataset-cifar10_unit-256

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 256 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 16 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.8 --quantize_level 16 --is_biased True \
    --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/resume/1622038533_lr-0.1_epochs-300_batchsize-128_num_mpi_process_16_topology-ring_comm_info-compress_top_k-0.8__dataset-cifar10_unit-256


python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 256 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 20 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.8 --quantize_level 16 --is_biased True \
    --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/resume/1622041153_lr-0.1_epochs-300_batchsize-128_num_mpi_process_20_topology-ring_comm_info-compress_top_k-0.8__dataset-cifar10_unit-256

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 256 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 16 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.8 --quantize_level 16 --is_biased True \
    --hostfile hostfile --graph_topology torus --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/resume/1622050942_lr-0.1_epochs-300_batchsize-128_num_mpi_process_16_topology-torus_comm_info-compress_top_k-0.8__dataset-cifar10_unit-256


python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 256 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 20 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.8 --quantize_level 16 --is_biased True \
    --hostfile hostfile --graph_topology torus --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/resume/1622053472_lr-0.1_epochs-300_batchsize-128_num_mpi_process_20_topology-torus_comm_info-compress_top_k-0.8__dataset-cifar10_unit-256

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 256 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 16 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.8 --quantize_level 16 --is_biased True \
    --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/resume/1622063719_lr-0.1_epochs-300_batchsize-128_num_mpi_process_16_topology-ring_comm_info-compress_top_k-0.8__dataset-cifar10_unit-256


python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 256 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 20 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.8 --quantize_level 16 --is_biased True \
    --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/resume/1622066254_lr-0.1_epochs-300_batchsize-128_num_mpi_process_20_topology-ring_comm_info-compress_top_k-0.8__dataset-cifar10_unit-256

python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 256 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 16 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.8 --quantize_level 16 --is_biased True \
    --hostfile hostfile --graph_topology torus --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/resume/1622076890_lr-0.1_epochs-300_batchsize-128_num_mpi_process_16_topology-torus_comm_info-compress_top_k-0.8__dataset-cifar10_unit-256


python run.py \
    --arch mlp --optimizer parallel_choco_v \
    --units 256 \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory False \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 20 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cpu \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 100,150,200 \
    --save_some_models 50,100,150,200,217,300 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_top_k --consensus_stepsize 0.0375 --compress_ratio 0.8 --quantize_level 16 --is_biased True \
    --hostfile hostfile --graph_topology torus --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/dist/bin/python --mpi_path $HOME/.conda/envs/dist/bin/mpirun --evaluate_avg True \
    --resume /home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/resume/1622079471_lr-0.1_epochs-300_batchsize-128_num_mpi_process_20_topology-torus_comm_info-compress_top_k-0.8__dataset-cifar10_unit-256