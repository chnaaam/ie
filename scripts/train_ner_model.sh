#!/bin/bash

cd ../
python3 -m nlp.train.named_entity_recognition train \
    --model_name "monologg/kocharelectra-base-discriminator" \
    --train_batch_size 48 \
    --valid_batch_size 72 \
    --num_epochs 5 \
    --learning_rate 3e-5 \
    --max_seq_length 256 \
    --num_train_workers 4 \
    --num_valid_workers 4 \
    --device_id 0 \
    --use_fp16 True \
    --betas "(0.9, 0.99)" \
    --weight_decay 0.01 \
    --seed 42 \
    
