#!/bin/bash

cd ../
python3 -m nlp.train.named_entity_recognition preprocess \
    --tokenizer_name "monologg/kocharelectra-base-discriminator" \
    --max_seq_length 256 \
    --ai_hub_dataset_path "/mnt/drive_d/Datasets/ner(ver1_0)" \
    --klue_dataset_path "/mnt/drive_d/Datasets/klue-ner-v1.1" \
    --reload True \
    