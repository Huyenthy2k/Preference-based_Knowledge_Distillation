#!/bin/bash
set -e

cd ..
echo "Current directory: $(pwd)"
cd src/prefkd
echo "Current directory: $(pwd)"

python3 -u train.py \
  model=sft \
  model.policy_name_or_path=Qwen/Qwen2.5-1.5B \
  model.reference_name_or_path=Qwen/Qwen2.5-1.5B \
  model.teacher_tokenizer_name_or_path=Qwen/Qwen2.5-1.5B \
  model.student_tokenizer_name_or_path=Qwen/Qwen2.5-1.5B \
  model.policy_block_name=Qwen2DecoderLayer \
  model.reference_block_name=Qwen2DecoderLayer \
  loss=sft \
  policy_mode=student \
  gradient_accumulation_steps=2 batch_size=32 eval_batch_size=32 \
  trainer=FSDPTrainer sample_during_eval=false \
  save_repo=tonyshelby/Qwen2.5_1.5B_SFT
