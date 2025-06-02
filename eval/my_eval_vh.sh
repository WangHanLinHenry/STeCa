save_path=outputs/eval/
logs_path=${save_path}logs_eval
task=virtualhome

# launch the FastChat controller
python -u -m fastchat.serve.controller >> ${logs_path}/model_worker.log 2>&1 &
fs_controller_pid=$!

# Part 2: Evaluate SFT agent
fs_worker_port=21012
CUDA_VISIBLE_DEVICES=7 python -u -m fastchat.serve.vllm_worker --model-path ckt/virtualhome-steca-rft --port ${fs_worker_port} --worker-address http://localhost:${fs_worker_port} >> ${logs_path}/model_worker.log 2>&1 &


fs_worker_pid=$!
sleep 60
sleep 300

# evaluate on the test set
python -m eval_agent.main_vh --agent_config fastchat --model_name virtualhome-steca-rft --exp_config ${task} --split test --override --output_path outputs/eval/virtualhome-steca-rft-test --test_path data/vh_seen_test.jsonl
python -m eval_agent.main_vh --agent_config fastchat --model_name virtualhome-steca-rft --exp_config ${task} --split dev --override --output_path outputs/eval/virtualhome-steca-rft-dev --test_path data/vh_unseen_test.jsonl

# if failed, exit
if [ $? -ne 0 ]; then
    echo "base agent evaluation failed"
    kill -9 $fs_worker_pid
    exit 1
fi

# kill the model worker
kill -9 $fs_worker_pid
# kill the controller
kill -9 $fs_controller_pid