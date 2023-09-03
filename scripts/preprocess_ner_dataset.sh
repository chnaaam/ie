#!/bin/bash

cd ../
python3 -m nlp.train.named_entity_recognition preprocess \
    --korean_corpus_dataset_path "/mnt/drive_d/Datasets/NIKL_NE_2022_v1.0" \
    --klue_dataset_path "/mnt/drive_d/Datasets/klue-ner-v1.1" \
    --reload True \
    