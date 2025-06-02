gpu_nodes=(3 4 5 6)
sample_num_workers=4
save_path=outputs/
logs_path=${save_path}logs
save_dir=ckt/
cur_model_name=virtualhome-sft
worker_num=32
sample_node_num=4
task=virtualhome

# launch the FastChat controller
python -u -m fastchat.serve.controller >> ${logs_path}/model_worker.log 2>&1 &
fs_controller_pid=$!

# Part 3: Base agent explore stage
# launch the fastchat model worker

explore_model_name=${cur_model_name}-explore-max20

for ((j=0;j<${sample_num_workers};j=j+1)); do
    if [ -d "${save_dir}${explore_model_name}-${j}" ]; then
        echo "Link to model exists"
    else
        ln -s ${save_dir}${cur_model_name} ${save_dir}${explore_model_name}-${j}
    fi
done
if [ -f "${logs_path}/worker_pid.txt" ]; then
    rm ${logs_path}/worker_pid.txt
fi

fs_worker_port=21012
worker_idx=0
for ((j=0;j<${sample_num_workers};j=j+1)); do
    echo "Launch the model worker on port ${fs_worker_port}"
    CUDA_VISIBLE_DEVICES=${gpu_nodes[$j]} python -u -m fastchat.serve.vllm_worker \
        --model-path ${save_dir}${explore_model_name}-${j} \
        --port ${fs_worker_port} \
        --worker-address http://localhost:${fs_worker_port} >> ${logs_path}/model_worker-${j}.log 2>&1 &
    echo $! >> ${logs_path}/worker_pid.txt
    fs_worker_port=$(($fs_worker_port+1))
    worker_idx=$(($worker_idx+1))
    sleep 15
done

sleep 60
sleep 300

# start explore on the same sft data
echo "Base agent starts exploring"
if [ -f "${logs_path}/eval_pid.txt" ]; then
    rm ${logs_path}/eval_pid.txt
fi

step_traj_save_path=outputs/virtualhome/expert/expert_step_explore

# 这里很重要！ 这里很重要！ 这里很重要！
if [ -d ${step_traj_save_path} ]; then
    rm -r ${step_traj_save_path}
fi
mkdir -p ${step_traj_save_path}

for (( j = 0; j < $worker_num; j++ )); do
    python3 generate_response_virtualhome_expert.py --exp_config ${task} --model_name ${explore_model_name}-$((j%sample_node_num)) --part_num $((worker_num)) --part_idx ${j} --save_path ${step_traj_save_path}  >> ${logs_path}/gen_response_worker-${j}.log 2>&1 &
    echo $! >> ${logs_path}/eval_pid.txt
done

wait $(cat ${logs_path}/eval_pid.txt)
rm ${logs_path}/eval_pid.txt
echo "Base agent has finished exploring"

# if failed, exit
if [ $? -ne 0 ]; then
    echo "base agent exploration failed"
    kill -9 $(cat ${logs_path}/worker_pid.txt)
    rm ${logs_path}/worker_pid.txt
    exit 1
fi

# kill the model worker
echo "Kill the model workers"
kill -9 $(cat ${logs_path}/worker_pid.txt)
rm ${logs_path}/worker_pid.txt