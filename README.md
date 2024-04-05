# Timo Models Official Hub

## Summary

Welcome to the official hub for Timo models. Here, we provide access to two versions of the Timo model: a 7 billion parameter model and a 13 billion parameter model. 

## Training Timo

The training process for Timo is structured around three main steps:

- Supervised Fine-Tuning of Mathematical Models

- Generating Temporal Preference Pairs

- Temporal Direct Preference Optimization

## Enviroment

```bash
pip install -r requirements.txt
```

## Supervised fine-tuning omathematical models

```bash
torchrun --master_addr ${MASTER_ADDR} \
  --nproc_per_node=${NUM_GPUS} \
  --master_port=6009 \
  train.py \
  --model_name_or_path $MODEL_PATH \
  --data_path $DATA_PATH \
  --bf16 True \
  --output_dir ${OUTPUT_PATH}\
  --num_train_epochs 2 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 10000 \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
  --tf32 True
```

## Generating the temporal preference pairs

```bash
python generate.py \
    --model_path $model_path \
    --generate True \
    --train_data_path $train_data_path \
    --score True \
    --generate_data_path $generate_data_path \
    --save_path $save_path
```

## Temporal direct preference optimization 

```bash
deepspeed --include localhost:${DEVICES} --master_port 29502 dpo_train.py \
    --model_name_or_path ${MODELPATH} \
    --json_path ${JSONPATH} \
    --output_dir ${OUTPUTPATH}/${RUNNAME} \
    --num_train_epochs ${DPOEPOCH} \
    --beta 0.1 \
    --per_device_train_batch_size ${BSZPERDEV} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${GRADACC} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 1 \
    --learning_rate 5e-7 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "linear" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --report_to "wandb" \
    --run_name ${RUNNAME} \
    --bf16 True \
    --gradient_checkpointing True \
    --deepspeed ./ds_config/stage3_no_offloading_accelerate.json
```

