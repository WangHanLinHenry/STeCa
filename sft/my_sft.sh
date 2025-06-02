# You can change the task and model_path to your own task and model_path
task=alfworld
model_path=Llama-2-7b-chat-hf  

node_num=4  # number of GPUs

save_dir=ckt/
sft_model_name=${task}-sft

# Part 1: SFT stage
sft_data_path="data/${task}_sft.json"
batch_size=16
micro_batch_size=2
accumulation_step=$((${batch_size}/${node_num}/${micro_batch_size}))

export CUDA_VISIBLE_DEVICES=4,5,6,7
export NCCL_P2P_LEVEL=NVL

python -m torch.distributed.run --nproc_per_node=${node_num} --master_port=20002 /home/hanlin/hlwang_projects/IPR/fastchat/train/train.py \
    --model_name_or_path ${model_path} \
    --data_path ${sft_data_path} \
    --bf16 True \
    --output_dir ${save_dir}${sft_model_name} \
    --num_train_epochs 3 \
    --per_device_train_batch_size ${micro_batch_size} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${accumulation_step} \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess False

# if failed, exit
if [ $? -ne 0 ]; then
    echo "SFT training failed"
    exit 1
fi
