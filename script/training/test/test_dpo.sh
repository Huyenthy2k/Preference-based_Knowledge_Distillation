#!/bin/bash
set -e

cd ../../..
echo "Current directory: $(pwd)"
cd src/prefkd
echo "Current directory: $(pwd)"

/usr/local/envs/prefKD_train/bin/python -u train.py \
  model=dpo \
  model.policy_name_or_path=tonyshelby/Qwen2.5_0.5B_SFT_sample \
  model.reference_name_or_path=tonyshelby/Qwen2.5_0.5B_SFT_sample \
  model.teacher_tokenizer_name_or_path=tonyshelby/Qwen2.5_0.5B_SFT_sample \
  model.student_tokenizer_name_or_path=tonyshelby/Qwen2.5_0.5B_SFT_sample \
  model.policy_block_name=Qwen2DecoderLayer \
  model.reference_block_name=Qwen2DecoderLayer \
  loss=dpo \
  eval_every=5 \
  policy_mode=student \
  reference_mode=student \
  datasets=tonyshelby/sample \
  gradient_accumulation_steps=2 batch_size=4 eval_batch_size=4 \
  trainer=FSDPTrainer sample_during_eval=false \
  debug=false \
  save_repo=tonyshelby/Qwen2.5_0.5B_DPO_sample
