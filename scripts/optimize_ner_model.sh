#!/bin/bash

cd ../

python3 -m nlp.optimizer ner \
    --model_path ~/.nlp_projects/ner/outputs/score-83.07 \
    --output_path ~/.nlp_projects/ner/outputs/onnx-83.07/