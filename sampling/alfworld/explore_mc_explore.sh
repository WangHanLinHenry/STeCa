# 开始编写我的代码
gpu_nodes=(0 1 2 3)
sample_num_workers=4
save_path=outputs/
logs_path=${save_path}logs
save_dir=ckt/
cur_model_name=alfworld-sft
sample_node_num=4
task=alfworld
step_traj_save_path=outputs/alfworld/explore/explore_step_explore

# launch the FastChat controller
python -u -m fastchat.serve.controller >> ${logs_path}/model_worker.log 2>&1 &
fs_controller_pid=$!

monte_carlo_explore_model_name=${cur_model_name}-monte-carlo-explore

# Part 4: Estimate step-level rewards

for ((j=0;j<${sample_num_workers};j=j+1)); do
    if [ -d "${save_dir}${monte_carlo_explore_model_name}-${j}" ]; then
        echo "Link to model exists"
    else
        ln -s ${save_dir}${cur_model_name} ${save_dir}${monte_carlo_explore_model_name}-${j}
    fi
done
if [ -f "${logs_path}/worker_pid.txt" ]; then
    rm ${logs_path}/worker_pid.txt
fi


fs_worker_port=21012
worker_idx=0
for ((j=0;j<${sample_num_workers};j=j+1)); do
    echo "Launch the model worker on port ${fs_worker_port}"
    CUDA_VISIBLE_DEVICES=$((${worker_idx} % ${sample_num_workers})) python -u -m fastchat.serve.vllm_worker \
        --model-path ${save_dir}${monte_carlo_explore_model_name}-${j} \
        --port ${fs_worker_port} \
        --worker-address http://localhost:${fs_worker_port} >> ${logs_path}/model_worker-${j}.log 2>&1 &
    echo $! >> ${logs_path}/worker_pid.txt
    fs_worker_port=$(($fs_worker_port+1))
    worker_idx=$(($worker_idx+1))
    sleep 15
done
sleep 60

echo "Base agent starts monte carlo sampling"
if [ -f "${logs_path}/eval_pid.txt" ]; then
    rm ${logs_path}/eval_pid.txt
fi

sample_num=5
per_iteration_num=5
sample_workers=32
sample_iterations=$((sample_num/per_iteration_num))

for ((j=0;j<${sample_iterations};j=j+1));do
    for ((k=0;k<${per_iteration_num};k=k+1)); do
        # Part 3: sample trajectories
        monte_carlo_sample_save_path=${save_path}alfworld/explore/monte_carlo_sample_iteration/sampled_traj_$((j*per_iteration_num+k))
        for ((l=0;l<$sample_workers; l++)); do
            output_path=${monte_carlo_sample_save_path}/
            if [ -d ${output_path} ]; then
                rm -r ${output_path}
            fi
            mkdir -p ${output_path}
            python monte_carlo_sample_${task}.py --agent_config fastchat_explore --model_name ${monte_carlo_explore_model_name}-$((l%sample_num_workers)) --exp_config ${task} --part_num ${sample_workers} --part_idx ${l} --save_path ${output_path} --data_path ${step_traj_save_path} >> ${logs_path}/gen_response_worker-$((j*per_iteration_num+k))-${l}.log 2>&1 &
            echo $! >> ${logs_path}/eval_pid.txt
        done
    done
    wait $(cat ${logs_path}/eval_pid.txt)
    rm ${logs_path}/eval_pid.txt
    echo "Base agent has finished exploring ${j} iteration"
done


# kill the model worker
echo "Kill the model workers"
kill -9 $(cat ${logs_path}/worker_pid.txt)
rm ${logs_path}/worker_pid.txt