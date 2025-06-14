#!/bin/bash
set -e

# cd ../..
echo "Current directory: $(pwd)"
cd src/prefkd
echo "Current directory: $(pwd)"

python3 -u train.py \
  model=sft \
  model.policy_name_or_path=Qwen/Qwen2.5-0.5B \
  model.reference_name_or_path=Qwen/Qwen2.5-0.5B \
  model.teacher_tokenizer_name_or_path=Qwen/Qwen2.5-0.5B \
  model.student_tokenizer_name_or_path=Qwen/Qwen2.5-0.5B \
  model.policy_block_name=Qwen2DecoderLayer \
  model.reference_block_name=Qwen2DecoderLayer \
  loss=sft \
  eval_every=5 \
  policy_mode=student \
  datasets=tonyshelby/sample \
  gradient_accumulation_steps=2 batch_size=16 eval_batch_size=16 \
  trainer=FSDPTrainer sample_during_eval=false \
  debug=false \
  save_repo=tonyshelby/Qwen2.5_1.5B_SFT_sample_log
