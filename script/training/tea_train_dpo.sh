#!/bin/bash
set -e

cd ../..
echo "Current directory: $(pwd)"
cd src/prefkd
echo "Current directory: $(pwd)"

python3 -u train.py \
  model=dpo \
  model.policy_name_or_path=mistralai/Mistral-7B-v0.3 \
  model.reference_name_or_path=mistralai/Mistral-7B-v0.3 \
  model.teacher_tokenizer_name_or_path=mistralai/Mistral-7B-v0.3 \
  model.student_tokenizer_name_or_path=mistralai/Mistral-7B-v0.3 \
  model.policy_block_name=MistralDecoderLayer \
  model.reference_block_name=MistralDecoderLayer \
  loss=dpo \
  policy_mode=teacher \
  reference_mode=teacher \
  gradient_accumulation_steps=2 batch_size=32 eval_batch_size=32 \
  trainer=FSDPTrainer sample_during_eval=false \
  save_repo=tonyshelby/Mistral_v03_7B_DPO_plus




