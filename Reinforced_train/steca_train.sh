# This is a sample code for ALFWorld environment
# For VirtualHome environment, please modify the corresponding parameters

task=alfworld
batch_size=15
micro_batch_size=1
node_num=5
accumulation_step=$((${batch_size}/${node_num}/${micro_batch_size}))
beta=0.0
lr=3e-6

save_dir=ckt/
cur_model_name=${task}-sft
dpo_model_name=${task}-steca-rft

pm_data_path=outputs/alfworld/constructed_data/final_data_rft_train.json

export CUDA_VISIBLE_DEVICES=3,4,5,6,7
export NCCL_P2P_LEVEL=NVL

python -m torch.distributed.run --nproc_per_node=${node_num} --master_port=20002 fastchat/train/train_dpo_ours.py \
    --model_name_or_path ${save_dir}${cur_model_name} \
    --ref_model_name_or_path ${save_dir}${cur_model_name} \
    --data_path ${pm_data_path} \
    --bf16 True \
    --output_dir ${save_dir}${dpo_model_name} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${micro_batch_size} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${accumulation_step} \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --beta ${beta} \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 5 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 4096 \
    --max_prompt_length 512 \
    --max_target_length 3072 \
    --gradient_checkpointing True \
    --lazy_preprocess False
